"""Part-structured frame-token CodeFlow.

This is the canonical PS-CF path: one DiT token per RVQ frame, six grouped
part-specific input/output paths, and terminal projection tied to the frozen
part codebooks.
"""

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit_blocks import FinalLayer, FrameMotionTextDiT, TimestepEmbedder
from .inpaint_protocols import (
    PARTGRID_OP_EDIT,
    PARTGRID_OP_GENERATE,
    PARTGRID_OP_PRESERVE,
    PARTGRID_TASK_GLOBAL_EDIT,
    PARTGRID_TASK_TEMPORAL,
    PARTGRID_TASK_T2M,
)
from .kv_control import build_joint_control_condition, masked_joint_position_loss
from .motionstreamer272 import decode_motionstreamer272_joints_from_embeddings
from .motion_code_flow import MotionCodeFlow, MotionCodeFlowConfig, lengths_to_mask, sample_timesteps
from .text_encoder import FrozenCLIPTextEncoder, TextCondition
from .vq_tokenizers import build_codeflow_tokenizer


class PartStructuredMotionCodeFlow(MotionCodeFlow):
    """Rectified flow over structured six-part RVQ frame embeddings."""

    def __init__(self, config: MotionCodeFlowConfig) -> None:
        nn.Module.__init__(self)
        if config.representation != "part_structured":
            raise ValueError("PartStructuredMotionCodeFlow requires representation='part_structured'")
        if config.coupling_mode != "frame_grouped":
            raise ValueError("PartStructuredMotionCodeFlow requires coupling_mode='frame_grouped'")
        if config.time_patch != 1:
            raise ValueError("PartStructuredMotionCodeFlow uses time_patch=1")
        if config.use_self_condition:
            raise ValueError("PartStructuredMotionCodeFlow canonical path disables self-conditioning")
        if float(config.clean_loss_weight) != 0.0:
            raise ValueError("PartStructuredMotionCodeFlow canonical objective uses clean_loss_weight=0")
        part_dim = int(config.part_hidden_dim) if int(config.part_hidden_dim) > 0 else int(config.code_dim)
        if config.hidden_size != config.num_parts * part_dim:
            raise ValueError(
                "PartStructuredMotionCodeFlow hidden size must match the grouped latent width: "
                f"hidden_size must be num_parts*part_hidden_dim={config.num_parts * part_dim}, "
                f"got {config.hidden_size}"
            )
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(f"hidden_size {config.hidden_size} must be divisible by num_heads={config.num_heads}")
        if config.terminal_mode not in {"nearest", "residual_nearest", "tied_logits", "learned_head"}:
            raise ValueError(f"Unsupported terminal_mode: {config.terminal_mode}")
        if config.latent_norm_mode not in {"none", "codebook", "empirical"}:
            raise ValueError(f"Unsupported latent_norm_mode: {config.latent_norm_mode}")
        if float(config.latent_offset) != 0.0:
            raise ValueError("PartStructuredMotionCodeFlow uses latent_offset=0 to preserve raw codebook metric")
        if config.sampling_schedule not in {"uniform", "logit_normal"}:
            raise ValueError(f"Unsupported sampling_schedule: {config.sampling_schedule}")
        if config.sampling_method not in {"ode", "sde"}:
            raise ValueError(f"Unsupported sampling_method: {config.sampling_method}")
        if config.decode_mode not in {"nearest", "ids", "continuous"}:
            raise ValueError(f"Unsupported decode_mode: {config.decode_mode}")
        if config.terminal_tau_mode not in {"fixed", "codebook_nn"}:
            raise ValueError(f"Unsupported terminal_tau_mode: {config.terminal_tau_mode}")
        self.config = config

        self.tokenizer = build_codeflow_tokenizer(
            backend=config.vq_backend,
            kv_root=config.kv_root,
            checkpoint_path=config.vq_checkpoint,
            partition_path=config.vq_partition,
            opt_path=config.vq_opt_path,
            rvq_target_mode=config.rvq_target_mode,
        )
        if self.tokenizer.num_parts != config.num_parts:
            raise ValueError(f"Config num_parts={config.num_parts}, tokenizer has {self.tokenizer.num_parts}")
        if self.tokenizer.num_codes != config.num_codes:
            raise ValueError(f"Config num_codes={config.num_codes}, tokenizer has {self.tokenizer.num_codes}")
        if self.tokenizer.code_dim != config.code_dim:
            raise ValueError(f"Config code_dim={config.code_dim}, tokenizer has {self.tokenizer.code_dim}")
        self._init_latent_stats()
        self._init_terminal_tau()

        self.text_encoder = FrozenCLIPTextEncoder(
            clip_version=config.clip_version,
            clip_path=config.clip_path,
            kv_root=config.kv_root,
        )

        self.part_input_norms = nn.ModuleList([
            nn.LayerNorm(config.code_dim, elementwise_affine=True, eps=1e-6)
            for _ in range(config.num_parts)
        ])
        self.part_inputs = nn.ModuleList([
            nn.Linear(config.code_dim, part_dim)
            for _ in range(config.num_parts)
        ])
        if config.enable_partgrid_inpainting:
            if int(config.partgrid_num_ops) <= PARTGRID_OP_EDIT:
                raise ValueError("partgrid_num_ops must include PRESERVE, GENERATE, and EDIT operations")
            if int(config.partgrid_num_tasks) < 3:
                raise ValueError("partgrid_num_tasks must include T2M, temporal, and part-grid tasks")
            self.partgrid_op_embed = nn.Embedding(int(config.partgrid_num_ops), part_dim)
            self.partgrid_task_embed = nn.Embedding(int(config.partgrid_num_tasks), part_dim)
            self.partgrid_source_fuse_norms = nn.ModuleList([
                nn.LayerNorm(part_dim, elementwise_affine=True, eps=1e-6)
                for _ in range(config.num_parts)
            ])
            self.partgrid_source_fuses = nn.ModuleList([
                nn.Linear(part_dim, part_dim)
                for _ in range(config.num_parts)
            ])
            for fuse in self.partgrid_source_fuses:
                nn.init.zeros_(fuse.weight)
                nn.init.zeros_(fuse.bias)
        else:
            self.partgrid_op_embed = None
            self.partgrid_task_embed = None
            self.partgrid_source_fuse_norms = None
            self.partgrid_source_fuses = None

        self.global_edit_similarity_norm = None
        self.global_edit_similarity_head = None
        if bool(config.enable_partgrid_inpainting) and float(config.global_edit_similarity_loss_weight) > 0.0:
            if int(config.global_edit_similarity_num_classes) < 2:
                raise ValueError("global_edit_similarity_num_classes must be at least 2")
            self.global_edit_similarity_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=True,
                eps=1e-6,
            )
            self.global_edit_similarity_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, int(config.global_edit_similarity_num_classes)),
            )

        self.timestep_embed = TimestepEmbedder(config.hidden_size)
        self.text_token_proj = nn.Linear(self.text_encoder.width, config.hidden_size)
        self.text_pooled_proj = nn.Sequential(
            nn.Linear(self.text_encoder.output_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        head_dim = config.hidden_size // config.num_heads
        self.backbone = FrameMotionTextDiT(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            depth_double=config.depth_double,
            depth_single=config.depth_single,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            rope_axes_dims=[head_dim],
            control_input_dim=int(config.kv_control_input_dim) if bool(config.enable_kv_control) else 0,
            control_rank=int(config.kv_control_rank),
            control_encoder_width=int(config.kv_control_encoder_width),
            control_attn_bias_init=float(config.kv_control_attn_bias_init),
        )
        self.part_outputs = nn.ModuleList([
            FinalLayer(config.hidden_size, config.code_dim)
            for _ in range(config.num_parts)
        ])

        if config.terminal_mode == "learned_head":
            self.learned_heads = nn.ModuleList([
                nn.Linear(config.code_dim, config.num_codes)
                for _ in range(config.num_parts)
            ])
        else:
            self.learned_heads = None

        if bool(config.enable_kv_control) and bool(config.kv_control_train_adapter_only):
            self._freeze_non_control_parameters()

    @property
    def holder_output(self):
        # Kept as a compatibility shim for older training health checks.
        return self.part_outputs[0]

    @property
    def core_output_weight(self) -> torch.Tensor:
        return self.part_outputs[0].linear.weight

    @property
    def partgrid_parameter_prefixes(self) -> Tuple[str, ...]:
        return (
            "partgrid_op_embed.",
            "partgrid_task_embed.",
            "partgrid_source_fuse_norms.",
            "partgrid_source_fuses.",
            "global_edit_similarity_norm.",
            "global_edit_similarity_head.",
        )

    @property
    def kv_control_parameter_prefixes(self) -> Tuple[str, ...]:
        return (
            "backbone.control_encoder.",
            "backbone.control_kv_down.",
            "backbone.control_kv_up_k.",
            "backbone.control_kv_up_v.",
            "backbone.control_attn_bias.",
        )

    @property
    def compatible_missing_prefixes(self) -> Tuple[str, ...]:
        prefixes: Tuple[str, ...] = ()
        if bool(self.config.enable_partgrid_inpainting):
            prefixes = prefixes + self.partgrid_parameter_prefixes
        if bool(self.config.enable_kv_control):
            prefixes = prefixes + self.kv_control_parameter_prefixes
        return prefixes

    def _freeze_non_control_parameters(self) -> None:
        prefixes = self.kv_control_parameter_prefixes
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith(prefixes))

    def _init_terminal_tau(self) -> None:
        cfg = self.config
        if cfg.terminal_tau_mode == "codebook_nn":
            if self.tokenizer.codebooks.shape[0] != cfg.num_parts:
                raise ValueError(
                    "terminal_tau_mode='codebook_nn' requires one codebook per model part; "
                    f"got codebook parts={self.tokenizer.codebooks.shape[0]} and num_parts={cfg.num_parts}."
                )
            values: List[torch.Tensor] = []
            for part_idx in range(cfg.num_parts):
                codebook = self.tokenizer.codebooks[part_idx].float()
                dist_sq = torch.cdist(codebook, codebook, p=2.0).square()
                dist_sq.fill_diagonal_(float("inf"))
                nearest = dist_sq.min(dim=1).values
                tau = torch.median(nearest[torch.isfinite(nearest)])
                values.append(tau.clamp_min(float(cfg.terminal_tau_floor)))
            tau_parts = torch.stack(values).float()
        else:
            tau_parts = torch.full(
                (cfg.num_parts,),
                max(float(cfg.terminal_tau), float(cfg.terminal_tau_floor)),
                dtype=torch.float32,
            )
        self.register_buffer("terminal_tau_parts", tau_parts, persistent=False)

    def _text_condition(
        self,
        texts: Iterable[str],
        drop_prob: float = 0.0,
        force_drop: bool = False,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> TextCondition:
        cond = self.text_encoder(texts, drop_prob=drop_prob, force_drop=force_drop, drop_mask=drop_mask)
        return TextCondition(
            pooled=self.text_pooled_proj(cond.pooled),
            tokens=self.text_token_proj(cond.tokens),
            padding_mask=cond.padding_mask,
        )

    def _pack_motion(
        self,
        x: torch.Tensor,
        token_lengths: torch.Tensor,
        source_latent: Optional[torch.Tensor] = None,
        op_ids: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        preserve_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        bsz, latent_len, num_parts, dim = x.shape
        if num_parts != cfg.num_parts or dim != cfg.code_dim:
            raise RuntimeError(
                f"Expected motion latent [B,T,{cfg.num_parts},{cfg.code_dim}], got {tuple(x.shape)}"
            )
        motion_valid = lengths_to_mask(token_lengths, latent_len)
        use_partgrid = bool(cfg.enable_partgrid_inpainting)
        if use_partgrid:
            if source_latent is None:
                source_latent = torch.zeros_like(x)
            else:
                source_latent = source_latent.to(device=x.device, dtype=x.dtype)
                if source_latent.shape != x.shape:
                    raise RuntimeError(
                        f"Expected source_latent shape {tuple(x.shape)}, got {tuple(source_latent.shape)}"
                    )
            if op_ids is None:
                op_ids = torch.full(
                    (bsz, latent_len, cfg.num_parts),
                    PARTGRID_OP_GENERATE,
                    device=x.device,
                    dtype=torch.long,
                )
            else:
                op_ids = op_ids.to(device=x.device, dtype=torch.long)
                if op_ids.shape != x.shape[:3]:
                    raise RuntimeError(f"Expected op_ids shape {tuple(x.shape[:3])}, got {tuple(op_ids.shape)}")
                if bool((op_ids < 0).any()) or bool((op_ids >= int(cfg.partgrid_num_ops)).any()):
                    raise RuntimeError(
                        f"op_ids must be in [0, {int(cfg.partgrid_num_ops) - 1}] "
                        "for the configured meta-operation table"
                    )
            if preserve_mask is not None:
                preserve_mask = preserve_mask.to(device=x.device, dtype=torch.bool)
                if preserve_mask.shape != x.shape[:3]:
                    raise RuntimeError(
                        f"Expected preserve_mask shape {tuple(x.shape[:3])}, got {tuple(preserve_mask.shape)}"
                    )
            source_visible = op_ids == PARTGRID_OP_EDIT
            if preserve_mask is not None:
                source_visible = source_visible | preserve_mask
            else:
                source_visible = source_visible | (op_ids == PARTGRID_OP_PRESERVE)
            source_latent = source_latent * source_visible[:, :, :, None].to(source_latent.dtype)
            if task_ids is None:
                task_ids = torch.full((bsz,), PARTGRID_TASK_T2M, device=x.device, dtype=torch.long)
            else:
                task_ids = task_ids.to(device=x.device, dtype=torch.long)
                if task_ids.shape != (bsz,):
                    raise RuntimeError(f"Expected task_ids shape ({bsz},), got {tuple(task_ids.shape)}")
                if bool((task_ids < 0).any()) or bool((task_ids >= int(cfg.partgrid_num_tasks)).any()):
                    raise RuntimeError(
                        f"task_ids must be in [0, {int(cfg.partgrid_num_tasks) - 1}] "
                        "for the configured task embedding table"
                    )
            task_ctx = self.partgrid_task_embed(task_ids)[:, None, :]

        part_chunks = []
        for part_idx in range(cfg.num_parts):
            part_x = self.part_input_norms[part_idx](x[:, :, part_idx])
            part_chunk = self.part_inputs[part_idx](part_x)
            if use_partgrid:
                source_x = self.part_input_norms[part_idx](source_latent[:, :, part_idx])
                source_chunk = self.part_inputs[part_idx](source_x)
                ctx = source_chunk + self.partgrid_op_embed(op_ids[:, :, part_idx]) + task_ctx
                ctx = self.partgrid_source_fuse_norms[part_idx](ctx)
                part_chunk = part_chunk + self.partgrid_source_fuses[part_idx](ctx)
            part_chunks.append(part_chunk)
        tokens = torch.cat(part_chunks, dim=-1)
        time_ids = torch.arange(latent_len, device=x.device, dtype=torch.float32)
        pos = time_ids.view(1, latent_len, 1).expand(bsz, -1, -1)
        return tokens, motion_valid, pos

    def forward(
        self,
        z: torch.Tensor,
        timesteps: torch.Tensor,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        x_self_cond: Optional[torch.Tensor] = None,
        text_drop_prob: float = 0.0,
        force_text_drop: bool = False,
        text_drop_mask: Optional[torch.Tensor] = None,
        source_latent: Optional[torch.Tensor] = None,
        op_ids: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        preserve_mask: Optional[torch.Tensor] = None,
        control_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del x_self_cond
        cfg = self.config
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(z.shape[0])
        timesteps = timesteps.to(device=z.device, dtype=z.dtype)
        token_lengths = token_lengths.to(z.device).long().clamp(min=1, max=z.shape[1])

        text_cond = self._text_condition(
            texts,
            drop_prob=text_drop_prob,
            force_drop=force_text_drop,
            drop_mask=text_drop_mask,
        )
        motion_tokens, motion_valid, motion_pos = self._pack_motion(
            z,
            token_lengths,
            source_latent=source_latent,
            op_ids=op_ids,
            task_ids=task_ids,
            preserve_mask=preserve_mask,
        )
        cond = self.timestep_embed(timesteps.float()) + text_cond.pooled
        hidden = self.backbone(
            motion=motion_tokens,
            text=text_cond.tokens,
            cond=cond,
            motion_valid=motion_valid,
            text_padding_mask=text_cond.padding_mask,
            motion_pos_ids=motion_pos,
            control_cond=control_cond,
        )
        parts = [head(hidden, cond) for head in self.part_outputs]
        pred = torch.stack(parts, dim=2)
        valid = lengths_to_mask(token_lengths, z.shape[1]).to(pred.dtype)
        return pred * valid[:, :, None, None]

    def _global_edit_similarity_logits(
        self,
        source_model: torch.Tensor,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        text_drop_prob: float,
    ) -> torch.Tensor:
        if self.global_edit_similarity_head is None or self.global_edit_similarity_norm is None:
            raise RuntimeError("Global edit similarity head is not initialized")
        cfg = self.config
        text_cond = self._text_condition(texts, drop_prob=text_drop_prob)
        part_chunks = []
        for part_idx in range(cfg.num_parts):
            part_x = self.part_input_norms[part_idx](source_model[:, :, part_idx])
            part_chunks.append(self.part_inputs[part_idx](part_x))
        source_tokens = torch.cat(part_chunks, dim=-1)
        source_tokens = self.global_edit_similarity_norm(source_tokens + text_cond.pooled[:, None, :])
        return self.global_edit_similarity_head(source_tokens)

    def _global_edit_similarity_targets(
        self,
        source_model: torch.Tensor,
        target_model: torch.Tensor,
        valid_frame_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        num_classes = int(cfg.global_edit_similarity_num_classes)
        bsz, latent_len = source_model.shape[:2]
        device = source_model.device
        valid = valid_frame_mask.to(device=device, dtype=torch.bool)
        source_flat = source_model.detach().float().reshape(bsz, latent_len, -1)
        target_flat = target_model.detach().float().reshape(bsz, latent_len, -1)
        normalized = source_flat.new_zeros((bsz, latent_len))
        labels = torch.zeros((bsz, latent_len), device=device, dtype=torch.long)
        usable_samples = torch.zeros((bsz,), device=device, dtype=torch.bool)
        snr_values = source_flat.new_zeros((bsz,))
        radius = max(0, int(cfg.global_edit_similarity_window))
        topk_cfg = max(1, int(cfg.global_edit_similarity_snr_topk))
        min_range = float(cfg.global_edit_similarity_min_range)
        min_snr = float(cfg.global_edit_similarity_min_snr)

        for batch_idx in range(bsz):
            valid_ids = torch.nonzero(valid[batch_idx], as_tuple=False).flatten()
            valid_count = int(valid_ids.numel())
            if valid_count <= 1:
                continue
            src = source_flat[batch_idx, valid_ids]
            tgt = target_flat[batch_idx, valid_ids]
            dist = torch.cdist(src, tgt, p=2.0)
            pos = torch.arange(valid_count, device=device)
            window = (pos[:, None] - pos[None, :]).abs() <= radius
            dist = dist.masked_fill(~window, float("inf"))
            nearest = dist.min(dim=1).values
            raw = -nearest
            finite = torch.isfinite(raw)
            if not bool(finite.all()):
                raw = torch.where(finite, raw, torch.zeros_like(raw))
            raw_min = raw.min()
            raw_max = raw.max()
            raw_range = raw_max - raw_min
            if not torch.isfinite(raw_range) or float(raw_range.detach().cpu()) < min_range:
                continue
            curve = ((raw - raw_min) / raw_range.clamp_min(1e-8)).clamp(0.0, 1.0)
            topk = min(topk_cfg, valid_count)
            top_mean = torch.topk(curve, k=topk, largest=True).values.mean()
            bottom_mean = torch.topk(curve, k=topk, largest=False).values.mean()
            snr = top_mean / bottom_mean.clamp_min(1e-6)
            if min_snr > 0.0 and float(snr.detach().cpu()) < min_snr:
                continue
            normalized[batch_idx, valid_ids] = curve
            labels[batch_idx, valid_ids] = torch.clamp(
                (curve * float(num_classes)).long(),
                min=0,
                max=num_classes - 1,
            )
            usable_samples[batch_idx] = True
            snr_values[batch_idx] = snr

        usable_frames = valid & usable_samples[:, None]
        return labels, usable_frames, snr_values

    def _global_edit_similarity_ddp_zero(self, ref: torch.Tensor) -> torch.Tensor:
        zero = ref.new_zeros(())
        for module in (self.global_edit_similarity_norm, self.global_edit_similarity_head):
            if module is None:
                continue
            for param in module.parameters():
                if param.numel() > 0:
                    zero = zero + param.reshape(-1)[0].float() * 0.0
        return zero

    def terminal_logits(self, clean_pred: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        mode = mode or self.config.terminal_mode
        if mode in {"nearest", "tied_logits"}:
            return self.tokenizer.codebook_tied_logits(clean_pred, tau=self.terminal_tau_parts)
        if mode == "learned_head":
            if self.learned_heads is None:
                raise RuntimeError("learned_head terminal mode was not initialized")
            logits = []
            for part_idx, head in enumerate(self.learned_heads):
                logits.append(head(clean_pred[:, :, part_idx]))
            return torch.stack(logits, dim=2)
        raise ValueError(f"Unknown terminal mode: {mode}")

    def compute_losses(
        self,
        target_embeddings: torch.Tensor,
        target_ids: torch.Tensor,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        include_geometry_metrics: bool = False,
        geometry_severe_quantile: float = 0.75,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        x_self_cond: Optional[torch.Tensor] = None,
        allow_internal_self_condition: bool = True,
        source_embeddings: Optional[torch.Tensor] = None,
        op_ids: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        preserve_mask: Optional[torch.Tensor] = None,
        control_target_joints: Optional[torch.Tensor] = None,
        control_target_mask: Optional[torch.Tensor] = None,
        control_mean: Optional[torch.Tensor] = None,
        control_std: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del x_self_cond, allow_internal_self_condition
        cfg = self.config
        text_list = list(texts)
        bsz, latent_len, num_parts, _ = target_embeddings.shape
        if getattr(self.tokenizer, "target_mode", "stage") == "sum":
            expected_quantizers = int(getattr(self.tokenizer, "num_quantizers", target_ids.shape[-1]))
            if num_parts != 1 or target_ids.ndim != 3 or target_ids.shape[-1] != expected_quantizers:
                raise ValueError(
                    "MoMask RVQ sum target expects target_embeddings [B,T,1,D] and "
                    f"target_ids [B,T,{expected_quantizers}], got "
                    f"target_embeddings={tuple(target_embeddings.shape)} target_ids={tuple(target_ids.shape)}"
                )
        token_lengths = token_lengths.to(target_embeddings.device).long().clamp(min=1, max=latent_len)
        valid = lengths_to_mask(token_lengths, latent_len)
        valid_parts = valid[:, :, None].expand(bsz, latent_len, num_parts)
        valid_ids = valid[:, :, None].expand_as(target_ids).bool()
        valid_float = valid_parts.to(target_embeddings.dtype)
        target_model = self.raw_to_model_latent(target_embeddings)
        source_model = None
        if source_embeddings is not None:
            source_embeddings = source_embeddings.to(device=target_embeddings.device, dtype=target_embeddings.dtype)
            if source_embeddings.shape != target_embeddings.shape:
                raise RuntimeError(
                    f"Expected source_embeddings shape {tuple(target_embeddings.shape)}, "
                    f"got {tuple(source_embeddings.shape)}"
                )
            source_model = self.raw_to_model_latent(source_embeddings)

        preserve_bool = None
        if preserve_mask is not None:
            preserve_bool = preserve_mask.to(device=target_embeddings.device, dtype=torch.bool)
            if preserve_bool.shape != target_embeddings.shape[:3]:
                raise RuntimeError(
                    f"Expected preserve_mask shape {tuple(target_embeddings.shape[:3])}, got {tuple(preserve_bool.shape)}"
                )
            preserve_bool = preserve_bool & valid_parts.bool()
        if bool(cfg.enable_partgrid_inpainting) and preserve_bool is not None:
            generated_bool = valid_parts.bool() & ~preserve_bool
            preserve_weight = float(cfg.inpaint_preserve_loss_weight)
            if task_ids is not None:
                task_ids_for_weight = task_ids.to(device=target_embeddings.device, dtype=torch.long)
                if bool((task_ids_for_weight == PARTGRID_TASK_GLOBAL_EDIT).any()):
                    global_weight = torch.where(
                        (task_ids_for_weight == PARTGRID_TASK_GLOBAL_EDIT)[:, None, None],
                        torch.full_like(preserve_bool, float(cfg.global_edit_preserve_loss_weight), dtype=target_embeddings.dtype),
                        torch.full_like(preserve_bool, preserve_weight, dtype=target_embeddings.dtype),
                    )
                else:
                    global_weight = torch.full_like(preserve_bool, preserve_weight, dtype=target_embeddings.dtype)
            else:
                global_weight = torch.full_like(preserve_bool, preserve_weight, dtype=target_embeddings.dtype)
            cell_weight = (
                generated_bool.to(target_embeddings.dtype) * float(cfg.inpaint_generated_loss_weight)
                + preserve_bool.to(target_embeddings.dtype) * global_weight
            )
        else:
            generated_bool = valid_parts.bool()
            cell_weight = valid_float
        loss_denom = cell_weight.float().sum().clamp_min(1.0)

        if noise is None:
            noise = torch.randn_like(target_model) * cfg.noise_scale
        else:
            noise = noise.to(device=target_model.device, dtype=target_model.dtype)
        if timesteps is None:
            t = sample_timesteps(
                bsz,
                target_embeddings.device,
                cfg.time_schedule,
                cfg.denoiser_p_mean,
                cfg.denoiser_p_std,
            ).to(target_embeddings.dtype)
        else:
            t = timesteps.to(device=target_embeddings.device, dtype=target_embeddings.dtype)
            if t.ndim == 0:
                t = t.expand(bsz)
        t_view = t[:, None, None, None]
        z_t = t_view * target_model + (1.0 - t_view) * noise
        velocity_target = target_model - noise
        z_t = z_t * valid_float[:, :, :, None]

        control_active = (
            bool(cfg.enable_kv_control)
            and control_target_joints is not None
            and control_target_mask is not None
        )
        control_cond = None
        control_target_joints_f = None
        control_target_mask_b = None
        shared_text_drop_mask = None
        if control_active and float(cfg.cond_drop_prob) > 0.0:
            shared_text_drop_mask = torch.rand(
                bsz,
                device=target_embeddings.device,
            ) < float(cfg.cond_drop_prob)
        if control_active:
            if control_mean is None or control_std is None:
                raise RuntimeError("KV control loss requires control_mean and control_std tensors")
            with torch.no_grad():
                velocity_base = self.forward(
                    z_t,
                    t,
                    text_list,
                    token_lengths,
                    x_self_cond=None,
                    text_drop_prob=cfg.cond_drop_prob,
                    text_drop_mask=shared_text_drop_mask,
                    source_latent=source_model,
                    op_ids=op_ids,
                    task_ids=task_ids,
                    preserve_mask=preserve_bool,
                    control_cond=None,
                )
                clean_base = self.predict_clean_from_velocity(z_t.float(), t.float(), velocity_base.float())
                clean_base_raw = self.model_to_raw_latent(clean_base)
                current_joints = decode_motionstreamer272_joints_from_embeddings(
                    self.tokenizer,
                    clean_base_raw,
                    mean=control_mean,
                    std=control_std,
                )
            control_target_joints_f = control_target_joints.to(
                device=target_embeddings.device,
                dtype=current_joints.dtype,
            )
            control_target_mask_b = control_target_mask.to(device=target_embeddings.device, dtype=torch.bool)
            if control_target_joints_f.shape != current_joints.shape:
                if control_target_joints_f.shape[0] != current_joints.shape[0] or control_target_joints_f.shape[2:] != current_joints.shape[2:]:
                    raise ValueError(
                        "KV control target joints must match batch/joint dims: "
                        f"target={tuple(control_target_joints_f.shape)} current={tuple(current_joints.shape)}"
                    )
                aligned_target = torch.zeros_like(current_joints)
                aligned_mask = torch.zeros_like(current_joints, dtype=torch.bool)
                common_frames = min(control_target_joints_f.shape[1], current_joints.shape[1])
                aligned_target[:, :common_frames] = control_target_joints_f[:, :common_frames]
                aligned_mask[:, :common_frames] = control_target_mask_b[:, :common_frames]
                control_target_joints_f = aligned_target
                control_target_mask_b = aligned_mask
            control_cond = build_joint_control_condition(
                current_joints,
                control_target_joints_f,
                control_target_mask_b,
            )

        velocity_pred = self.forward(
            z_t,
            t,
            text_list,
            token_lengths,
            x_self_cond=None,
            text_drop_prob=cfg.cond_drop_prob,
            text_drop_mask=shared_text_drop_mask,
            source_latent=source_model,
            op_ids=op_ids,
            task_ids=task_ids,
            preserve_mask=preserve_bool,
            control_cond=control_cond,
        )
        velocity_pred_f = velocity_pred.float()
        velocity_target_f = velocity_target.float()
        target_model_f = target_model.float()
        source_model_f = source_model.float() if source_model is not None else None
        valid_float_f = valid_float.float()
        z_t_f = z_t.float()
        t_f = t.float()
        cell_weight_f = cell_weight.float()

        per_part_flow = (velocity_pred_f - velocity_target_f).square().mean(dim=-1)
        flow_loss = (per_part_flow * cell_weight_f).sum() / loss_denom

        clean_pred = self.predict_clean_from_velocity(z_t_f, t_f, velocity_pred_f)
        clean_loss = flow_loss.new_zeros(())
        control_loss = flow_loss.new_zeros(())
        control_mask_frac = flow_loss.new_zeros(())
        global_edit_clean_loss = flow_loss.new_zeros(())
        global_edit_similarity_loss = flow_loss.new_zeros(())
        global_edit_similarity_acc = flow_loss.new_zeros(())
        global_edit_similarity_valid_frac = flow_loss.new_zeros(())
        global_edit_similarity_snr = flow_loss.new_zeros(())
        global_edit_similarity_ddp_zero = self._global_edit_similarity_ddp_zero(flow_loss)
        clean_pred_raw = self.model_to_raw_latent(clean_pred)

        if control_active and control_target_joints_f is not None and control_target_mask_b is not None:
            pred_joints = decode_motionstreamer272_joints_from_embeddings(
                self.tokenizer,
                clean_pred_raw.to(target_embeddings.dtype),
                mean=control_mean,
                std=control_std,
            )
            if pred_joints.shape != control_target_joints_f.shape:
                common_frames = min(pred_joints.shape[1], control_target_joints_f.shape[1])
                pred_joints = pred_joints[:, :common_frames]
                control_target_joints_f = control_target_joints_f[:, :common_frames]
                control_target_mask_b = control_target_mask_b[:, :common_frames]
            control_loss = masked_joint_position_loss(
                pred_joints,
                control_target_joints_f,
                control_target_mask_b,
                loss_type=str(cfg.kv_control_loss_type),
            ).to(flow_loss.dtype)
            control_mask_frac = (
                control_target_mask_b.float().mean().to(device=flow_loss.device, dtype=flow_loss.dtype)
            )

        if task_ids is not None:
            task_ids_for_loss = task_ids.to(device=target_embeddings.device, dtype=torch.long)
            global_edit_sample = task_ids_for_loss == PARTGRID_TASK_GLOBAL_EDIT
        else:
            global_edit_sample = torch.zeros((bsz,), device=target_embeddings.device, dtype=torch.bool)
        global_edit_cell = valid_parts.bool() & global_edit_sample[:, None, None]

        if bool(global_edit_sample.any()) and float(cfg.global_edit_clean_loss_weight) > 0.0:
            per_part_clean = (clean_pred - target_model_f).square().mean(dim=-1)
            edit_float = global_edit_cell.to(per_part_clean.dtype)
            global_edit_clean_loss = (per_part_clean * edit_float).sum() / edit_float.sum().clamp_min(1.0)

        if (
            source_model_f is not None
            and bool(global_edit_sample.any())
            and float(cfg.global_edit_similarity_loss_weight) > 0.0
            and self.global_edit_similarity_head is not None
        ):
            valid_frame = valid.bool() & global_edit_sample[:, None]
            sim_labels, sim_valid, sim_snr = self._global_edit_similarity_targets(source_model_f, target_model_f, valid_frame)
            sim_logits = self._global_edit_similarity_logits(
                source_model,
                text_list,
                token_lengths,
                text_drop_prob=cfg.cond_drop_prob,
            ).float()
            num_classes = int(cfg.global_edit_similarity_num_classes)
            sim_count = sim_valid.sum().float()
            if bool(sim_valid.any()):
                flat_labels = sim_labels[sim_valid]
                class_counts = torch.bincount(flat_labels, minlength=num_classes).to(sim_logits.dtype)
                class_weights = class_counts.sum().clamp_min(1.0) / class_counts.clamp_min(1.0)
                class_weights = class_weights / class_weights.mean().clamp_min(1e-8)
                sim_ce = F.cross_entropy(
                    sim_logits.reshape(-1, num_classes),
                    sim_labels.reshape(-1),
                    weight=class_weights,
                    reduction="none",
                ).view(bsz, latent_len)
                sim_valid_f = sim_valid.to(sim_ce.dtype)
                global_edit_similarity_loss = (sim_ce * sim_valid_f).sum() / sim_count.clamp_min(1.0)
                sim_pred = sim_logits.argmax(dim=-1)
                global_edit_similarity_acc = (
                    ((sim_pred == sim_labels) & sim_valid).sum().float() / sim_count.clamp_min(1.0)
                )
                global_edit_similarity_valid_frac = sim_count / valid.bool().sum().float().clamp_min(1.0)
                usable_sample = sim_valid.any(dim=1)
                if bool(usable_sample.any()):
                    global_edit_similarity_snr = sim_snr[usable_sample].mean()

        terminal_loss = flow_loss.new_zeros(())
        code_weight = (
            (t_f >= float(cfg.code_ce_t_min))
            & (t_f <= float(cfg.code_ce_t_max))
        ).to(valid_float_f.dtype) * t_f.clamp_min(0.0).pow(float(cfg.code_ce_gamma))
        if cfg.terminal_mode in {"tied_logits", "learned_head"} and cfg.terminal_loss_weight > 0.0:
            if target_ids.shape[-1] != num_parts:
                raise ValueError(
                    "Token CE terminal loss requires target_ids to match the model part axis; "
                    f"got target_ids={tuple(target_ids.shape)} and target_embeddings={tuple(target_embeddings.shape)}. "
                    "Use terminal_mode='residual_nearest' with terminal_loss_weight=0 for summed RVQ targets."
                )
            with torch.cuda.amp.autocast(enabled=False):
                logits = self.terminal_logits(clean_pred_raw.float()).float()
                ce = F.cross_entropy(
                    logits.reshape(-1, cfg.num_codes),
                    target_ids.reshape(-1).long(),
                    reduction="none",
                ).view(bsz, latent_len, num_parts)
                if cfg.code_ce_normalize:
                    ce = ce / math.log(float(cfg.num_codes))
                ce = ce * code_weight[:, None, None]
            terminal_loss = (ce * cell_weight_f).sum() / loss_denom

        with torch.no_grad():
            pred_ids = self.terminal_ids(clean_pred_raw)
            acc = ((pred_ids == target_ids.long()) & valid_ids).sum().float() / valid_ids.sum().float().clamp_min(1.0)
            nn_ids = self.tokenizer.nearest_ids(clean_pred_raw)
            nn_acc = ((nn_ids == target_ids.long()) & valid_ids).sum().float() / valid_ids.sum().float().clamp_min(1.0)
            generated_ids_bool = generated_bool if generated_bool.shape == target_ids.shape else valid_ids
            generated_count = generated_ids_bool.sum().float().clamp_min(1.0)
            generated_acc = (
                ((pred_ids == target_ids.long()) & generated_ids_bool).sum().float() / generated_count
            )
            if preserve_bool is not None:
                preserve_ids_bool = preserve_bool if preserve_bool.shape == target_ids.shape else torch.zeros_like(valid_ids)
                preserved_count = preserve_ids_bool.sum().float().clamp_min(1.0)
                preserved_acc = ((pred_ids == target_ids.long()) & preserve_ids_bool).sum().float() / preserved_count
                preserved_frac = preserve_bool.sum().float() / valid_parts.sum().float().clamp_min(1.0)
            else:
                preserved_acc = acc.new_zeros(())
                preserved_frac = acc.new_zeros(())
            generated_frac = generated_bool.sum().float() / valid_parts.sum().float().clamp_min(1.0)

        total = (
            cfg.flow_loss_weight * flow_loss
            + cfg.terminal_loss_weight * terminal_loss
            + cfg.kv_control_loss_weight * control_loss
            + cfg.global_edit_clean_loss_weight * global_edit_clean_loss
            + cfg.global_edit_similarity_loss_weight * global_edit_similarity_loss
            + global_edit_similarity_ddp_zero
        )
        out = {
            "loss": total,
            "flow_loss": flow_loss,
            "terminal_loss": terminal_loss,
            "clean_loss": clean_loss,
            "kv_control_loss": control_loss,
            "kv_control_mask_frac": control_mask_frac,
            "global_edit_clean_loss": global_edit_clean_loss,
            "global_edit_similarity_loss": global_edit_similarity_loss,
            "global_edit_similarity_acc": global_edit_similarity_acc,
            "global_edit_similarity_valid_frac": global_edit_similarity_valid_frac,
            "global_edit_similarity_snr": global_edit_similarity_snr,
            "token_acc": acc,
            "nearest_acc": nn_acc,
            "code_ce_weight": code_weight.mean(),
            "generated_token_acc": generated_acc,
            "preserved_token_acc": preserved_acc,
            "generated_cell_frac": generated_frac,
            "preserved_cell_frac": preserved_frac,
            "cell_loss_weight": cell_weight_f.sum() / valid_float_f.sum().clamp_min(1.0),
        }
        if include_geometry_metrics:
            with torch.no_grad():
                code_dist, rank_pct = self.tokenizer.code_id_distances(target_ids.long(), pred_ids.long())
                valid_bool = valid_ids.bool()
                wrong = (pred_ids != target_ids.long()) & valid_bool
                severe = wrong & (rank_pct >= float(geometry_severe_quantile))
                valid_count = valid_bool.sum().float().clamp_min(1.0)
                wrong_count = wrong.sum().float()
                wrong_denom = wrong_count.clamp_min(1.0)
                out.update({
                    "geom_code_dist": (code_dist * valid_bool.to(code_dist.dtype)).sum() / valid_count,
                    "geom_rank_pct": (rank_pct * valid_bool.to(rank_pct.dtype)).sum() / valid_count,
                    "geom_wrong_code_dist": (code_dist * wrong.to(code_dist.dtype)).sum() / wrong_denom,
                    "geom_wrong_rank_pct": (rank_pct * wrong.to(rank_pct.dtype)).sum() / wrong_denom,
                    "geom_wrong_rate": wrong_count / valid_count,
                    "geom_severe_rate": severe.sum().float() / valid_count,
                    "geom_wrong_severe_frac": severe.sum().float() / wrong_denom,
                })
        return out

    @torch.no_grad()
    def sample_embeddings(
        self,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        steps: int = 32,
        cond_scale: float = 3.0,
        use_self_condition: bool = True,
    ) -> torch.Tensor:
        del use_self_condition
        cfg = self.config
        text_list = list(texts)
        bsz = len(text_list)
        token_lengths = token_lengths.to(self.device).long()
        latent_len = int(token_lengths.max().item())
        z = torch.randn(
            bsz,
            latent_len,
            cfg.num_parts,
            cfg.code_dim,
            device=self.device,
        ) * cfg.noise_scale
        valid = lengths_to_mask(token_lengths, latent_len).to(z.dtype)
        z = z * valid[:, :, None, None]
        grid = self._sampling_grid(int(steps), self.device).to(z.dtype)

        def forward_guided(z_in: torch.Tensor, t_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            if cond_scale == 1.0:
                v_out = self.forward(
                    z_in,
                    t_in,
                    text_list,
                    token_lengths,
                    text_drop_prob=0.0,
                )
            else:
                z_cat = torch.cat([z_in, z_in], dim=0)
                lengths_cat = torch.cat([token_lengths, token_lengths], dim=0)
                texts_cat = [""] * bsz + text_list
                v_all = self.forward(
                    z_cat,
                    torch.cat([t_in, t_in], dim=0),
                    texts_cat,
                    lengths_cat,
                    text_drop_prob=0.0,
                )
                v_uncond, v_cond = v_all.chunk(2, dim=0)
                v_out = v_uncond + float(cond_scale) * (v_cond - v_uncond)
            clean_out = self.predict_clean_from_velocity(z_in, t_in, v_out)
            return v_out, clean_out

        for idx in range(steps):
            t_cur_scalar = grid[idx]
            t_next_scalar = grid[idx + 1]
            dt = t_next_scalar - t_cur_scalar
            z_eval = z
            t_eval_scalar = t_cur_scalar
            if cfg.sampling_method == "sde" and float(cfg.sde_gamma) > 0.0:
                alpha_value = max(0.0, min(1.0, 1.0 - float(cfg.sde_gamma) * float(dt.item())))
                eps = torch.randn_like(z) * cfg.noise_scale
                z_eval = alpha_value * z + (1.0 - alpha_value) * eps
                t_eval_scalar = t_cur_scalar * alpha_value
                dt = t_next_scalar - t_eval_scalar

            t_eval = t_eval_scalar.expand(bsz)
            v, _ = forward_guided(z_eval, t_eval)
            z = z_eval + dt * v
            z = z * valid[:, :, None, None]

        raw = self.model_to_raw_latent(z)
        return raw * valid[:, :, None, None]

    @torch.no_grad()
    def generate_ids(
        self,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        steps: int = 32,
        cond_scale: float = 3.0,
        terminal_mode: Optional[str] = None,
    ) -> torch.Tensor:
        clean = self.sample_embeddings(
            texts,
            token_lengths=token_lengths,
            steps=steps,
            cond_scale=cond_scale,
        )
        return self.terminal_ids(clean, mode=terminal_mode)

    @torch.no_grad()
    def generate_motion(
        self,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        steps: int = 32,
        cond_scale: float = 3.0,
        terminal_mode: Optional[str] = None,
        decode_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.sample_embeddings(
            texts,
            token_lengths=token_lengths,
            steps=steps,
            cond_scale=cond_scale,
        )
        mode = decode_mode or self.config.decode_mode
        if mode not in {"nearest", "ids", "continuous"}:
            raise ValueError(f"Unsupported decode_mode: {mode}")
        ids = self.terminal_ids(clean, mode=terminal_mode)
        if mode == "continuous":
            motion = self.tokenizer.decode_embeddings(clean)
        else:
            motion = self.tokenizer.decode_ids(ids)
        return motion, ids

    @torch.no_grad()
    def sample_inpaint_embeddings(
        self,
        texts: Iterable[str],
        source_embeddings: torch.Tensor,
        preserve_mask: torch.Tensor,
        token_lengths: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        steps: int = 32,
        cond_scale: float = 3.0,
    ) -> torch.Tensor:
        if not bool(self.config.enable_partgrid_inpainting):
            raise RuntimeError("PartGrid inpainting is disabled in this model config")
        cfg = self.config
        text_list = list(texts)
        bsz = len(text_list)
        if source_embeddings.ndim != 4:
            raise ValueError(f"Expected source_embeddings [B,T,P,D], got {tuple(source_embeddings.shape)}")
        if source_embeddings.shape[0] != bsz:
            raise ValueError(f"source batch {source_embeddings.shape[0]} does not match text batch {bsz}")
        source_embeddings = source_embeddings.to(self.device)
        latent_len = source_embeddings.shape[1]
        token_lengths = token_lengths.to(self.device).long().clamp(min=1, max=latent_len)
        preserve_mask = preserve_mask.to(self.device, dtype=torch.bool)
        if preserve_mask.shape != source_embeddings.shape[:3]:
            raise ValueError(
                f"Expected preserve_mask shape {tuple(source_embeddings.shape[:3])}, got {tuple(preserve_mask.shape)}"
            )
        valid = lengths_to_mask(token_lengths, latent_len)
        valid_parts = valid[:, :, None].expand_as(preserve_mask)
        preserve_mask = preserve_mask & valid_parts
        op_ids = torch.full_like(preserve_mask, PARTGRID_OP_GENERATE, dtype=torch.long)
        op_ids = torch.where(
            preserve_mask,
            torch.full_like(op_ids, PARTGRID_OP_PRESERVE),
            op_ids,
        )
        if task_ids is None:
            task_ids = torch.full((bsz,), PARTGRID_TASK_TEMPORAL, device=self.device, dtype=torch.long)
        else:
            task_ids = task_ids.to(self.device, dtype=torch.long)
        source_model = self.raw_to_model_latent(source_embeddings)
        fixed_noise = torch.randn_like(source_model) * cfg.noise_scale
        z = torch.randn_like(source_model) * cfg.noise_scale
        valid_float = valid.to(z.dtype)
        z = z * valid_float[:, :, None, None]
        grid = self._sampling_grid(int(steps), self.device).to(z.dtype)

        def clamp_to_source_path(z_in: torch.Tensor, t_value: torch.Tensor) -> torch.Tensor:
            if t_value.ndim == 0:
                t_view = t_value.view(1, 1, 1, 1).expand(bsz, 1, 1, 1)
            else:
                t_view = t_value.view(bsz, 1, 1, 1)
            source_path = t_view * source_model + (1.0 - t_view) * fixed_noise
            return torch.where(preserve_mask[:, :, :, None], source_path, z_in)

        def forward_guided(z_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            if cond_scale == 1.0:
                return self.forward(
                    z_in,
                    t_in,
                    text_list,
                    token_lengths,
                    text_drop_prob=0.0,
                    source_latent=source_model,
                    op_ids=op_ids,
                    task_ids=task_ids,
                    preserve_mask=preserve_mask,
                )
            z_cat = torch.cat([z_in, z_in], dim=0)
            lengths_cat = torch.cat([token_lengths, token_lengths], dim=0)
            texts_cat = [""] * bsz + text_list
            source_cat = torch.cat([source_model, source_model], dim=0)
            op_cat = torch.cat([op_ids, op_ids], dim=0)
            task_cat = torch.cat([task_ids, task_ids], dim=0)
            preserve_cat = torch.cat([preserve_mask, preserve_mask], dim=0)
            v_all = self.forward(
                z_cat,
                torch.cat([t_in, t_in], dim=0),
                texts_cat,
                lengths_cat,
                text_drop_prob=0.0,
                source_latent=source_cat,
                op_ids=op_cat,
                task_ids=task_cat,
                preserve_mask=preserve_cat,
            )
            v_uncond, v_cond = v_all.chunk(2, dim=0)
            return v_uncond + float(cond_scale) * (v_cond - v_uncond)

        z = clamp_to_source_path(z, grid[0])
        for idx in range(steps):
            t_cur_scalar = grid[idx]
            t_next_scalar = grid[idx + 1]
            dt = t_next_scalar - t_cur_scalar
            z_eval = clamp_to_source_path(z, t_cur_scalar)
            t_eval = t_cur_scalar.expand(bsz)
            v = forward_guided(z_eval, t_eval)
            z = z_eval + dt * v
            z = z * valid_float[:, :, None, None]
            z = clamp_to_source_path(z, t_next_scalar)

        raw = self.model_to_raw_latent(z)
        raw = torch.where(preserve_mask[:, :, :, None], source_embeddings, raw)
        return raw * valid_float[:, :, None, None]

    @torch.no_grad()
    def generate_inpaint_motion(
        self,
        texts: Iterable[str],
        source_embeddings: torch.Tensor,
        preserve_mask: torch.Tensor,
        token_lengths: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        steps: int = 32,
        cond_scale: float = 3.0,
        terminal_mode: Optional[str] = None,
        decode_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.sample_inpaint_embeddings(
            texts,
            source_embeddings=source_embeddings,
            preserve_mask=preserve_mask,
            token_lengths=token_lengths,
            task_ids=task_ids,
            steps=steps,
            cond_scale=cond_scale,
        )
        mode = decode_mode or self.config.decode_mode
        if mode not in {"nearest", "ids", "continuous"}:
            raise ValueError(f"Unsupported decode_mode: {mode}")
        ids = self.terminal_ids(clean, mode=terminal_mode)
        if mode == "continuous":
            motion = self.tokenizer.decode_embeddings(clean)
        else:
            motion = self.tokenizer.decode_ids(ids)
        return motion, ids

    @torch.no_grad()
    def sample_global_edit_embeddings(
        self,
        texts: Iterable[str],
        source_embeddings: torch.Tensor,
        token_lengths: torch.Tensor,
        steps: int = 32,
        cond_scale: float = 3.0,
        preserve_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not bool(self.config.enable_partgrid_inpainting):
            raise RuntimeError("Global edit requires PartGrid conditioning modules")
        cfg = self.config
        text_list = list(texts)
        bsz = len(text_list)
        if source_embeddings.ndim != 4:
            raise ValueError(f"Expected source_embeddings [B,T,P,D], got {tuple(source_embeddings.shape)}")
        if source_embeddings.shape[0] != bsz:
            raise ValueError(f"source batch {source_embeddings.shape[0]} does not match text batch {bsz}")
        source_embeddings = source_embeddings.to(self.device)
        latent_len = source_embeddings.shape[1]
        token_lengths = token_lengths.to(self.device).long().clamp(min=1, max=latent_len)
        valid = lengths_to_mask(token_lengths, latent_len)
        valid_parts = valid[:, :, None].expand_as(source_embeddings[..., 0])
        task_ids = torch.full((bsz,), PARTGRID_TASK_GLOBAL_EDIT, device=self.device, dtype=torch.long)
        if preserve_mask is None:
            preserve_mask = torch.zeros_like(valid_parts, dtype=torch.bool)
        else:
            preserve_mask = preserve_mask.to(self.device, dtype=torch.bool)
            if preserve_mask.shape != valid_parts.shape:
                raise ValueError(f"Expected preserve_mask shape {tuple(valid_parts.shape)}, got {tuple(preserve_mask.shape)}")
            preserve_mask = preserve_mask & valid_parts
        op_ids = torch.full(valid_parts.shape, PARTGRID_OP_EDIT, device=self.device, dtype=torch.long)
        op_ids = torch.where(
            preserve_mask,
            torch.full_like(op_ids, PARTGRID_OP_PRESERVE),
            op_ids,
        )
        source_model = self.raw_to_model_latent(source_embeddings)
        fixed_noise = torch.randn_like(source_model) * cfg.noise_scale
        z = torch.randn_like(source_model) * cfg.noise_scale
        valid_float = valid.to(z.dtype)
        z = z * valid_float[:, :, None, None]
        grid = self._sampling_grid(int(steps), self.device).to(z.dtype)

        def clamp_to_source_path(z_in: torch.Tensor, t_value: torch.Tensor) -> torch.Tensor:
            if t_value.ndim == 0:
                t_view = t_value.view(1, 1, 1, 1).expand(bsz, 1, 1, 1)
            else:
                t_view = t_value.view(bsz, 1, 1, 1)
            source_path = t_view * source_model + (1.0 - t_view) * fixed_noise
            return torch.where(preserve_mask[:, :, :, None], source_path, z_in)

        def forward_guided(z_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            if cond_scale == 1.0:
                return self.forward(
                    z_in,
                    t_in,
                    text_list,
                    token_lengths,
                    text_drop_prob=0.0,
                    source_latent=source_model,
                    op_ids=op_ids,
                    task_ids=task_ids,
                    preserve_mask=preserve_mask,
                )
            z_cat = torch.cat([z_in, z_in], dim=0)
            lengths_cat = torch.cat([token_lengths, token_lengths], dim=0)
            texts_cat = [""] * bsz + text_list
            source_cat = torch.cat([source_model, source_model], dim=0)
            op_cat = torch.cat([op_ids, op_ids], dim=0)
            task_cat = torch.cat([task_ids, task_ids], dim=0)
            preserve_cat = torch.cat([preserve_mask, preserve_mask], dim=0)
            t_cat = torch.cat([t_in, t_in], dim=0)
            v_all = self.forward(
                z_cat,
                t_cat,
                texts_cat,
                lengths_cat,
                text_drop_prob=0.0,
                source_latent=source_cat,
                op_ids=op_cat,
                task_ids=task_cat,
                preserve_mask=preserve_cat,
            )
            v_uncond, v_cond = v_all.chunk(2, dim=0)
            return v_uncond + float(cond_scale) * (v_cond - v_uncond)

        z = clamp_to_source_path(z, grid[0])
        for idx in range(steps):
            t_cur_scalar = grid[idx]
            t_next_scalar = grid[idx + 1]
            dt = t_next_scalar - t_cur_scalar
            z_eval = clamp_to_source_path(z, t_cur_scalar)
            t_eval = t_cur_scalar.expand(bsz)
            v = forward_guided(z_eval, t_eval)
            z = z_eval + dt * v
            z = z * valid_float[:, :, None, None]
            z = clamp_to_source_path(z, t_next_scalar)

        raw = self.model_to_raw_latent(z)
        raw = torch.where(preserve_mask[:, :, :, None], source_embeddings, raw)
        return raw * valid_float[:, :, None, None]

    @torch.no_grad()
    def generate_global_edit_motion(
        self,
        texts: Iterable[str],
        source_embeddings: torch.Tensor,
        token_lengths: torch.Tensor,
        steps: int = 32,
        cond_scale: float = 3.0,
        terminal_mode: Optional[str] = None,
        decode_mode: Optional[str] = None,
        preserve_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.sample_global_edit_embeddings(
            texts,
            source_embeddings=source_embeddings,
            token_lengths=token_lengths,
            steps=steps,
            cond_scale=cond_scale,
            preserve_mask=preserve_mask,
        )
        mode = decode_mode or self.config.decode_mode
        if mode not in {"nearest", "ids", "continuous"}:
            raise ValueError(f"Unsupported decode_mode: {mode}")
        ids = self.terminal_ids(clean, mode=terminal_mode)
        if mode == "continuous":
            motion = self.tokenizer.decode_embeddings(clean)
        else:
            motion = self.tokenizer.decode_ids(ids)
        return motion, ids
