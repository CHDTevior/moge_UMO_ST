"""Unified single-actor and two-actor HY273 rectified-flow model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn

from .hy273_actor_exchange import ActorExchangeFrameMotionTextDiT
from .hy273_multitask_condition import ConditionBatch, NUM_TASKS, TaskId
from .hy273_slices import DIM_HY273, ROOT_DIM
from .kimodo_context_flow_dit import HY273KimodoContextFlow
from .text_condition import RawTextCondition


@dataclass
class UnifiedActorFlowOutput:
    prediction: torch.Tensor
    root_prediction_raw: torch.Tensor
    local_root: torch.Tensor
    context_root: torch.Tensor
    context_body: torch.Tensor
    text_tokens: torch.Tensor
    text_pooled: torch.Tensor
    actor_count: int


def _repeat_text_condition(
    condition: RawTextCondition, actor_count: int
) -> RawTextCondition:
    actor_count = int(actor_count)
    local_tokens = (
        None
        if condition.local_tokens is None
        else condition.local_tokens.repeat_interleave(actor_count, dim=0)
    )
    local_padding_mask = (
        None
        if condition.local_padding_mask is None
        else condition.local_padding_mask.repeat_interleave(actor_count, dim=0)
    )
    return RawTextCondition(
        tokens=condition.tokens.repeat_interleave(actor_count, dim=0),
        pooled=condition.pooled.repeat_interleave(actor_count, dim=0),
        padding_mask=condition.padding_mask.repeat_interleave(actor_count, dim=0),
        local_tokens=local_tokens,
        local_padding_mask=local_padding_mask,
    )


class HY273UnifiedActorFlow(HY273KimodoContextFlow):
    """Shared K-Encoder/root-body DiT with an explicit target actor axis.

    Single-person generation and editing run with ``A=1`` and therefore retain
    the existing compute graph. Text-to-interaction runs with ``A=2``; both
    actors share scene text/timestep/gauge while carrying independent noise and
    exchanging hidden states after every transformer block.
    """

    max_actors = 2

    def __init__(
        self,
        *args,
        actor_exchange_dim: int = 256,
        actor_exchange_heads: int = 4,
        **kwargs,
    ) -> None:
        if kwargs.get("source_fusion_mode", "additive") != "additive":
            raise ValueError("Unified actor v1 supports additive source context only")
        kwargs["source_context_num_tasks"] = NUM_TASKS
        kwargs["actor_exchange"] = True
        kwargs["actor_exchange_dim"] = int(actor_exchange_dim)
        kwargs["actor_exchange_heads"] = int(actor_exchange_heads)
        super().__init__(*args, **kwargs)
        if not isinstance(self.root_backbone, ActorExchangeFrameMotionTextDiT):
            raise RuntimeError("Unified actor root backbone was not actor-aware")
        if not isinstance(self.body_backbone, ActorExchangeFrameMotionTextDiT):
            raise RuntimeError("Unified actor body backbone was not actor-aware")
        self.task_condition_embed = nn.Embedding(NUM_TASKS, self.hidden_dim)
        nn.init.zeros_(self.task_condition_embed.weight)

    @staticmethod
    def _zero_parameter_anchor(
        module: nn.Module,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return sum(
            (parameter.reshape(-1)[0] * 0.0 for parameter in module.parameters()),
            reference.new_zeros(()),
        ).to(dtype=reference.dtype)

    @staticmethod
    def _canonicalize_actor_inputs(
        model_in: torch.Tensor,
        length_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        squeeze_actor = model_in.ndim == 3
        if squeeze_actor:
            model_in = model_in.unsqueeze(1)
        if model_in.ndim != 4:
            raise ValueError("model_in must have shape [B,T,D] or [B,A,T,D]")
        if model_in.shape[-1] != HY273UnifiedActorFlow.model_in_dim:
            raise ValueError(
                f"Expected model input dim {HY273UnifiedActorFlow.model_in_dim}, "
                f"got {model_in.shape[-1]}"
            )
        batch, actors, frames = model_in.shape[:3]
        if actors not in {1, 2}:
            raise ValueError("Unified actor v1 supports A=1 or A=2")
        if length_mask is None:
            valid = torch.ones(
                batch, actors, frames, device=model_in.device, dtype=torch.bool
            )
        else:
            valid = length_mask.to(device=model_in.device, dtype=torch.bool)
            if valid.ndim == 2 and actors == 1:
                valid = valid.unsqueeze(1)
            if valid.shape != (batch, actors, frames):
                raise ValueError("length_mask must have shape [B,T] or [B,A,T]")
        if not bool(valid.any(dim=-1).all()):
            raise ValueError("Every active actor must contain at least one valid frame")
        return model_in, valid, squeeze_actor

    def forward(
        self,
        model_in: torch.Tensor,
        t: torch.Tensor,
        c_dir: Optional[torch.Tensor] = None,
        text: Optional[Iterable[str]] = None,
        length_mask: Optional[torch.Tensor] = None,
        x_self_cond: Optional[torch.Tensor] = None,
        text_drop_prob: float = 0.0,
        force_drop_text: bool = False,
        return_details: bool = False,
        condition: ConditionBatch | None = None,
        *,
        task_id: torch.Tensor | None = None,
        text_profiles: tuple[str, ...] | None = None,
    ) -> torch.Tensor | UnifiedActorFlowOutput:
        model_in, actor_valid, squeeze_actor = self._canonicalize_actor_inputs(
            model_in, length_mask
        )
        batch, actors, frames, _ = model_in.shape
        device, dtype = model_in.device, model_in.dtype
        if condition is not None and actors != 1:
            raise ValueError("Source context is currently defined only for A=1 tasks")
        if condition is not None and condition.batch_size != batch:
            raise ValueError("Source condition batch differs from the scene batch")
        if condition is not None and bool(condition.ease_present.any()):
            raise ValueError(
                "Unified actor v1 does not implement Ease conditioning"
            )
        if task_id is None:
            if condition is None:
                raise ValueError("task_id is required when condition is absent")
            task_id = condition.task_id
        task_id = task_id.to(device=device, dtype=torch.long).reshape(-1)
        if task_id.shape != (batch,):
            raise ValueError("task_id must have shape [B]")
        if bool(((task_id < 0) | (task_id >= NUM_TASKS)).any()):
            raise ValueError("task_id contains an unknown task")
        interaction_task = task_id == int(TaskId.INTERACTION)
        if actors == 2 and not bool(interaction_task.all()):
            raise ValueError("A=2 batches must use TaskId.INTERACTION")
        if actors == 1 and bool(interaction_task.any()):
            raise ValueError("TaskId.INTERACTION requires an explicit A=2 target")
        if condition is not None and not torch.equal(
            condition.task_id.to(device=device, dtype=torch.long),
            task_id,
        ):
            raise ValueError("ConditionBatch task_id differs from the model task_id")

        t = t.expand(batch) if t.ndim == 0 else t
        t = t.to(device=device, dtype=dtype).reshape(batch)
        if c_dir is None:
            c_dir = torch.zeros(batch, 2, device=device, dtype=dtype)
            c_dir[:, 0] = 1.0
        else:
            c_dir = c_dir.to(device=device, dtype=dtype).reshape(batch, 2)
        text_list = tuple([""] * batch if text is None else list(text))
        if len(text_list) != batch:
            raise ValueError(f"Expected {batch} scene texts, got {len(text_list)}")
        profiles = (
            condition.text_encoding_profile
            if text_profiles is None and condition is not None
            else text_profiles
        )
        if profiles is not None and len(profiles) != batch:
            raise ValueError(f"Expected {batch} text profiles, got {len(profiles)}")
        scene_text = self._profiled_text_condition(
            text_list,
            profiles,
            batch,
            device,
            dtype,
            float(text_drop_prob),
            bool(force_drop_text),
        )
        flat_text = _repeat_text_condition(scene_text, actors)

        flat_t = t.repeat_interleave(actors)
        flat_dir = c_dir.repeat_interleave(actors, dim=0)
        cond = self._compose_global_condition(
            flat_t,
            flat_dir,
            flat_text.pooled,
            dtype=dtype,
        )
        task_bias = self.task_condition_embed(task_id).to(dtype=dtype)
        cond = cond + task_bias.repeat_interleave(actors, dim=0)

        flat_valid = actor_valid.reshape(batch * actors, frames)
        flat_model_in = model_in.reshape(batch * actors, frames, -1)
        lengths = flat_valid.sum(dim=-1).clamp_min(1)
        pos = (
            torch.arange(frames, device=device, dtype=torch.long)
            .view(1, frames, 1)
            .expand(batch * actors, frames, 1)
        )
        if condition is None:
            context_root = flat_model_in.new_zeros(
                batch * actors, frames, self.hidden_dim
            )
            context_body = flat_model_in.new_zeros(
                batch * actors, frames, self.hidden_dim
            )
            source_anchor = self._zero_parameter_anchor(
                self.source_context,
                flat_model_in,
            )
            context_root = context_root + source_anchor
            context_body = context_body + source_anchor
        else:
            context = self.source_context(condition, target_frames=frames, dtype=dtype)
            context_root = context.root.to(device=device, dtype=dtype)
            context_body = context.body.to(device=device, dtype=dtype)

        flat_self_cond = None
        if x_self_cond is not None:
            if squeeze_actor and x_self_cond.ndim == 3:
                x_self_cond = x_self_cond.unsqueeze(1)
            if x_self_cond.shape[:3] != (batch, actors, frames):
                raise ValueError("x_self_cond shape differs from model_in")
            flat_self_cond = x_self_cond.reshape(batch * actors, frames, -1)

        state = flat_model_in[..., :DIM_HY273]
        motion_mask = flat_model_in[..., DIM_HY273:]
        root_hidden = self.root_input_proj(flat_model_in) + context_root
        if self.self_conditioning and flat_self_cond is not None:
            root_hidden = root_hidden + self.self_cond_scale * self.root_self_cond_proj(
                flat_self_cond[..., :ROOT_DIM].to(device=device, dtype=dtype)
            )
        root_hidden = self.root_backbone(
            motion=root_hidden,
            text=flat_text.tokens,
            cond=cond,
            motion_valid=flat_valid,
            text_padding_mask=flat_text.padding_mask,
            motion_pos_ids=pos,
            local_text=flat_text.local_tokens,
            local_text_padding_mask=flat_text.local_padding_mask,
            scene_batch_size=batch,
            actor_count=actors,
        )
        root_prediction = self.root_output_proj(root_hidden)

        bridge_root = (
            root_prediction.detach()
            if self.training and self.detach_root_bridge
            else root_prediction
        )
        local_root = self.root_conditioner(bridge_root, lengths)
        body_in = torch.cat([local_root, state[..., ROOT_DIM:], motion_mask], dim=-1)
        body_hidden = self.body_input_proj(body_in) + context_body
        if self.self_conditioning and flat_self_cond is not None:
            body_hidden = body_hidden + self.self_cond_scale * self.body_self_cond_proj(
                flat_self_cond[..., ROOT_DIM:].to(device=device, dtype=dtype)
            )
        body_hidden = self.body_backbone(
            motion=body_hidden,
            text=flat_text.tokens,
            cond=cond,
            motion_valid=flat_valid,
            text_padding_mask=flat_text.padding_mask,
            motion_pos_ids=pos,
            local_text=flat_text.local_tokens,
            local_text_padding_mask=flat_text.local_padding_mask,
            scene_batch_size=batch,
            actor_count=actors,
        )
        body_prediction = self.body_output_proj(body_hidden)
        prediction = torch.cat([root_prediction, body_prediction], dim=-1)

        prediction = prediction.reshape(batch, actors, frames, DIM_HY273)
        root_prediction = root_prediction.reshape(batch, actors, frames, ROOT_DIM)
        local_root = local_root.reshape(batch, actors, frames, -1)
        context_root = context_root.reshape(batch, actors, frames, self.hidden_dim)
        context_body = context_body.reshape(batch, actors, frames, self.hidden_dim)
        if squeeze_actor:
            prediction = prediction[:, 0]
            root_prediction = root_prediction[:, 0]
            local_root = local_root[:, 0]
            context_root = context_root[:, 0]
            context_body = context_body[:, 0]
        if return_details:
            return UnifiedActorFlowOutput(
                prediction=prediction,
                root_prediction_raw=root_prediction,
                local_root=local_root,
                context_root=context_root,
                context_body=context_body,
                text_tokens=scene_text.tokens,
                text_pooled=scene_text.pooled,
                actor_count=actors,
            )
        return prediction
