"""Single-target HY273 model shared by T2M, Edit, and actor-conditioned Reaction."""

from __future__ import annotations

from typing import Any

import torch

from .hy273_multitask_condition import ConditionBatch, NUM_TASKS
from .kimodo_context_flow_dit import HY273KimodoContextFlow


class HY273UnifiedReactionFlow(HY273KimodoContextFlow):
    """Keep one denoising target while changing only typed source context."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("actor_exchange", False):
            raise ValueError("Reaction model cannot enable two-actor exchange")
        if kwargs.get("source_fusion_mode", "token_block") != "token_block":
            raise ValueError("Reaction model requires an independent source token block")
        kwargs["actor_exchange"] = False
        kwargs["source_fusion_mode"] = "token_block"
        kwargs["source_context_num_tasks"] = NUM_TASKS
        kwargs["global_task_conditioning"] = True
        super().__init__(*args, **kwargs)

    def forward(
        self,
        *args: Any,
        condition: ConditionBatch | None = None,
        task_id: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        text_profiles = kwargs.pop("text_profiles", None)
        if text_profiles is not None:
            if condition is None:
                raise ValueError("text_profiles require a typed ConditionBatch")
            if tuple(text_profiles) != tuple(condition.text_encoding_profile):
                raise ValueError(
                    "text_profiles differ from ConditionBatch.text_encoding_profile"
                )
        if task_id is not None:
            if condition is None:
                raise ValueError("task_id requires a typed ConditionBatch")
            expected = condition.task_id.to(
                device=task_id.device,
                dtype=torch.long,
            )
            actual = task_id.to(dtype=torch.long).reshape(-1)
            if not torch.equal(actual, expected):
                raise ValueError("task_id differs from ConditionBatch.task_id")
        return super().forward(*args, condition=condition, **kwargs)
