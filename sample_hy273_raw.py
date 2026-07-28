"""Sample HY273 raw-space flow checkpoints with step-wise source-domain clamping."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from data.kimodo273_datasets import Kimodo273TextDataset, collate_kimodo273_text
from models.raw_motion.asset_integrity import verify_asset_manifest
from models.raw_motion.flow_schedule import make_ode_grid
from models.raw_motion.hy273_constraints import build_synthetic_control_batch
from models.raw_motion.hy273_normalizer import HY273Normalizer, apply_kimodo_training_transform
from models.raw_motion.hy273_slices import CONTACT_SLICE, CONT_DIM, HEADING_SLICE, ROOT_SLICE
from train_hy273_raw_flow import (
    create_model,
    predict_clean_cont,
    prediction_velocity_cont,
    resolve_representation_loss_space,
    validate_normalizer_contract,
)


def verify_checkpoint_assets(train_args: argparse.Namespace) -> None:
    manifest_path = str(getattr(train_args, "asset_manifest_path", ""))
    if manifest_path:
        verify_asset_manifest(
            manifest_path,
            expected_manifest_sha256=str(
                getattr(train_args, "asset_manifest_sha256", "")
            ),
        )
    elif str(getattr(train_args, "architecture", "one_stage")) == "redenoise_kimodo_like":
        raise RuntimeError("redenoise_kimodo_like checkpoint has no pinned asset manifest")


def apply_checkpoint_path_override(
    train_args: argparse.Namespace,
    field: str,
    override: str,
) -> None:
    if not override:
        return
    current = str(getattr(train_args, field, ""))
    if (
        str(getattr(train_args, "architecture", "one_stage")) == "redenoise_kimodo_like"
        and Path(current).expanduser().resolve() != Path(override).expanduser().resolve()
    ):
        raise RuntimeError(
            f"Cannot override pinned {field} for redenoise_kimodo_like: "
            f"checkpoint={current!r}, requested={override!r}"
        )
    setattr(train_args, field, override)


def checkpoint_weight_state(
    checkpoint: dict[str, Any],
    weight_source: str,
    checkpoint_path: str,
) -> tuple[dict[str, torch.Tensor], str]:
    resolved = weight_source
    if resolved == "auto":
        resolved = "ema" if "ema" in checkpoint else "model"
    if resolved == "ema" and "ema" not in checkpoint:
        raise ValueError(f"EMA requested but checkpoint has no EMA: {checkpoint_path}")
    return checkpoint[resolved], resolved


def checkpoint_normalizer(
    checkpoint: dict[str, Any],
    train_args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: str,
) -> HY273Normalizer:
    normalizer = HY273Normalizer.from_data_root(
        train_args.data_root,
        stats_dir=getattr(train_args, "motion_stats_dir", "") or None,
        variance_eps=float(getattr(train_args, "stats_variance_eps", 0.0)),
    ).to(device)
    if "normalizer" in checkpoint:
        validate_normalizer_contract(normalizer, checkpoint["normalizer"], checkpoint_path)
    elif str(getattr(train_args, "architecture", "one_stage")) == "redenoise_kimodo_like":
        raise RuntimeError(f"redenoise_kimodo_like checkpoint lacks normalizer state: {checkpoint_path}")
    return normalizer


def resolve_sampling_cfg_scales(
    train_args: argparse.Namespace,
    text_scale: float | None,
    control_scale: float | None,
) -> tuple[float, float]:
    kimodo_like = (
        str(getattr(train_args, "architecture", "one_stage"))
        == "redenoise_kimodo_like"
    )
    return (
        float(text_scale if text_scale is not None else (2.0 if kimodo_like else 1.0)),
        float(control_scale if control_scale is not None else (2.0 if kimodo_like else 1.0)),
    )


@dataclass
class ODESampleOutput:
    raw_motion: torch.Tensor
    exact_clamped_motion: torch.Tensor
    final_clean_prediction: torch.Tensor
    final_branch_predictions: dict[str, torch.Tensor]


@torch.no_grad()
def sample_ode(
    model: torch.nn.Module,
    normalizer: HY273Normalizer,
    lengths: torch.Tensor,
    texts: list[str],
    observed_un: torch.Tensor,
    motion_mask: torch.Tensor,
    c_dir: torch.Tensor,
    num_steps: int = 32,
    self_conditioning: bool = False,
    cfg_scale: float = 1.0,
    control_cfg_scale: float = 1.0,
    contact_init: str = "random",
    contact_feedback: str = "blend",
    cfg_apply_contacts: bool = False,
    prediction_type: str = "velocity",
    velocity_t_eps: float = 1e-4,
    return_details: bool = False,
    generator: torch.Generator | None = None,
    initial_continuous_noise: torch.Tensor | None = None,
    initial_contact_noise: torch.Tensor | None = None,
) -> torch.Tensor | ODESampleOutput:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    observed = normalizer.normalize(observed_un.to(device=device, dtype=torch.float32)).to(dtype=dtype)
    motion_mask = motion_mask.to(device=device, dtype=dtype)
    lengths = lengths.to(device=device)
    bsz, frames, _ = observed.shape
    valid = torch.arange(frames, device=device)[None, :] < lengths[:, None]
    expected_cont_shape = (bsz, frames, CONT_DIM)
    expected_contact_shape = (bsz, frames, 4)
    if initial_continuous_noise is None:
        z_cont = torch.randn(
            *expected_cont_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        if initial_continuous_noise.shape != expected_cont_shape:
            raise ValueError(
                "initial_continuous_noise must have shape "
                f"{expected_cont_shape}, got {tuple(initial_continuous_noise.shape)}"
            )
        z_cont = initial_continuous_noise.to(device=device, dtype=dtype).clone()
    if initial_contact_noise is not None:
        if initial_contact_noise.shape != expected_contact_shape:
            raise ValueError(
                f"initial_contact_noise must have shape {expected_contact_shape}, "
                f"got {tuple(initial_contact_noise.shape)}"
            )
        contact_noise = initial_contact_noise.to(device=device, dtype=dtype).clone()
    elif contact_init == "zeros":
        contact_noise = torch.zeros(bsz, frames, 4, device=device, dtype=dtype)
    elif contact_init == "half":
        contact_noise = torch.full((bsz, frames, 4), 0.5, device=device, dtype=dtype)
    elif contact_init == "random":
        contact_noise = torch.rand(
            bsz,
            frames,
            4,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        raise ValueError(f"Unknown contact_init: {contact_init}")
    if contact_feedback not in {"blend", "prob", "fixed"}:
        raise ValueError(f"Unknown contact_feedback: {contact_feedback}")
    contact_aux = contact_noise.clone()
    x_self_cond: Optional[torch.Tensor] = None
    branch_self_cond: dict[str, torch.Tensor] = {}
    final_clean = torch.cat([z_cont, contact_aux], dim=-1)
    final_branches: dict[str, torch.Tensor] = {}
    grid = make_ode_grid(num_steps, device=device).to(dtype=dtype)
    for i in range(num_steps):
        t = grid[i].expand(bsz)
        dt = grid[i + 1] - grid[i]
        state = torch.cat([z_cont, contact_aux], dim=-1)
        zero_mask = torch.zeros_like(motion_mask)
        controlled_state = state * (1.0 - motion_mask) + observed * motion_mask
        input_free = torch.cat([state, zero_mask], dim=-1)
        input_control = torch.cat([controlled_state, motion_mask], dim=-1)
        has_control = bool(motion_mask.bool().any().item())

        if has_control:
            # Joint/text/control/empty branches share the same unclamped ODE state.
            # Overwrite exists only in the joint and control denoiser inputs.
            branch_input = torch.cat([input_control, input_free, input_control, input_free], dim=0)
            branch_text = list(texts) + list(texts) + [""] * bsz + [""] * bsz
            branch_t = t.repeat(4)
            branch_c_dir = c_dir.to(device=device, dtype=dtype).repeat(4, 1)
            branch_valid = valid.repeat(4, 1)
            branch_sc = None
            if self_conditioning and branch_self_cond:
                branch_sc = torch.cat(
                    [
                        branch_self_cond["joint"],
                        branch_self_cond["text"],
                        branch_self_cond["control"],
                        branch_self_cond["empty"],
                    ],
                    dim=0,
                )
            pred_all = model(
                branch_input,
                t=branch_t,
                c_dir=branch_c_dir,
                text=branch_text,
                length_mask=branch_valid,
                x_self_cond=branch_sc,
                text_drop_prob=0.0,
            )
            pred_joint, pred_text, pred_control, pred_empty = pred_all.chunk(4, dim=0)
            pred_guided = (
                pred_empty
                + float(cfg_scale) * (pred_text - pred_empty)
                + float(control_cfg_scale) * (pred_control - pred_empty)
            )
            pred = pred_guided if cfg_apply_contacts else pred_joint.clone()
            if not cfg_apply_contacts:
                pred[..., :CONT_DIM] = pred_guided[..., :CONT_DIM]
            final_branches = {
                "joint": pred_joint,
                "text": pred_text,
                "control": pred_control,
                "empty": pred_empty,
            }
            if self_conditioning:
                next_branch_sc: dict[str, torch.Tensor] = {}
                for name, value in final_branches.items():
                    branch_x0_cont = predict_clean_cont(
                        state[..., :CONT_DIM], t, value[..., :CONT_DIM], prediction_type
                    )
                    branch_x0 = torch.cat(
                        [branch_x0_cont, torch.sigmoid(value[..., CONTACT_SLICE])], dim=-1
                    )
                    if name in {"joint", "control"}:
                        branch_x0 = branch_x0 * (1.0 - motion_mask) + observed * motion_mask
                    next_branch_sc[name] = branch_x0.detach()
                branch_self_cond = next_branch_sc
        elif abs(float(cfg_scale) - 1.0) > 1e-6:
            branch_input = torch.cat([input_free, input_free], dim=0)
            branch_sc = None
            if self_conditioning and branch_self_cond:
                branch_sc = torch.cat(
                    [branch_self_cond["text"], branch_self_cond["empty"]], dim=0
                )
            pred_all = model(
                branch_input,
                t=t.repeat(2),
                c_dir=c_dir.to(device=device, dtype=dtype).repeat(2, 1),
                text=list(texts) + [""] * bsz,
                length_mask=valid.repeat(2, 1),
                x_self_cond=branch_sc,
                text_drop_prob=0.0,
            )
            pred_text, pred_empty = pred_all.chunk(2, dim=0)
            pred_guided = pred_empty + float(cfg_scale) * (pred_text - pred_empty)
            pred = pred_guided if cfg_apply_contacts else pred_text.clone()
            if not cfg_apply_contacts:
                pred[..., :CONT_DIM] = pred_guided[..., :CONT_DIM]
            final_branches = {"text": pred_text, "empty": pred_empty}
            if self_conditioning:
                branch_self_cond = {
                    name: torch.cat(
                        [
                            predict_clean_cont(
                                state[..., :CONT_DIM], t, value[..., :CONT_DIM], prediction_type
                            ),
                            torch.sigmoid(value[..., CONTACT_SLICE]),
                        ],
                        dim=-1,
                    ).detach()
                    for name, value in final_branches.items()
                }
                x_self_cond = branch_self_cond["text"]
        else:
            pred = model(
                input_free,
                t=t,
                c_dir=c_dir.to(device=device, dtype=dtype),
                text=texts,
                length_mask=valid,
                x_self_cond=x_self_cond,
                text_drop_prob=0.0,
            )
            final_branches = {"text": pred}
        pred_cont = pred[..., :CONT_DIM]
        contact_prob = torch.sigmoid(pred[..., CONTACT_SLICE])
        x0_hat_cont = predict_clean_cont(state[..., :CONT_DIM], t, pred_cont, prediction_type)
        v_cont = prediction_velocity_cont(
            state[..., :CONT_DIM],
            t,
            pred_cont,
            x0_hat_cont,
            prediction_type,
            velocity_t_eps=velocity_t_eps,
        )
        x0_hat = torch.cat([x0_hat_cont, contact_prob], dim=-1)
        x0_hat_clamped = x0_hat * (1.0 - motion_mask) + observed * motion_mask
        final_clean = x0_hat
        z_cont = state[..., :CONT_DIM] + dt * v_cont
        if contact_feedback == "blend":
            t_next = grid[i + 1].view(1, 1, 1).to(device=device, dtype=dtype)
            contact_aux = t_next * contact_prob + (1.0 - t_next) * contact_noise
        elif contact_feedback == "prob":
            contact_aux = contact_prob
        else:
            contact_aux = contact_noise
        if self_conditioning and not has_control:
            x_self_cond = x0_hat.detach()
    final = torch.cat([z_cont, contact_aux], dim=-1)
    final_exact = final * (1.0 - motion_mask) + observed * motion_mask
    out = normalizer.denormalize(final.float())
    out_exact = normalizer.denormalize(final_exact.float())
    out_clean = normalizer.denormalize(final_clean.float())
    branch_outputs = {
        name: normalizer.denormalize(
            torch.cat([value[..., :CONT_DIM], torch.sigmoid(value[..., CONTACT_SLICE])], dim=-1).float()
        )
        for name, value in final_branches.items()
    }
    for batch_idx in range(bsz):
        length = int(lengths[batch_idx].clamp(min=1, max=frames).item())
        if length < frames:
            out[batch_idx, length:] = out[batch_idx, length - 1 : length]
            out_exact[batch_idx, length:] = out_exact[batch_idx, length - 1 : length]
            out_clean[batch_idx, length:] = out_clean[batch_idx, length - 1 : length]
            for value in branch_outputs.values():
                value[batch_idx, length:] = value[batch_idx, length - 1 : length]
    if return_details:
        return ODESampleOutput(
            raw_motion=out,
            exact_clamped_motion=out_exact,
            final_clean_prediction=out_clean,
            final_branch_predictions=branch_outputs,
        )
    return out


def anchor_first_frame_convention(
    observed_un: torch.Tensor,
    motion_mask: torch.Tensor,
    c_dir: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor the coordinate convention learned from root_origin_shift/c_dir."""
    observed_un = observed_un.clone()
    motion_mask = motion_mask.clone()
    observed_un[:, 0, ROOT_SLICE.start] = 0.0
    observed_un[:, 0, ROOT_SLICE.start + 2] = 0.0
    motion_mask[:, 0, ROOT_SLICE.start] = True
    motion_mask[:, 0, ROOT_SLICE.start + 2] = True
    observed_un[:, 0, HEADING_SLICE] = c_dir.to(device=observed_un.device, dtype=observed_un.dtype)
    motion_mask[:, 0, HEADING_SLICE] = True
    return observed_un, motion_mask


def resolve_endpoint_protocol(
    train_args: argparse.Namespace,
    endpoint_preset: str = "",
    endpoint_subset_mode: str = "",
    endpoint_root_ref_mode: str = "",
    max_control_keyframes: int = 0,
    include_endpoint_rotations: bool | None = None,
) -> dict[str, object]:
    root_ref_mode = endpoint_root_ref_mode or str(
        getattr(train_args, "endpoint_root_ref_mode", "kimodo_hidden_root")
    )
    configured_modes = {
        item.strip()
        for item in str(getattr(train_args, "control_modes", "")).split(",")
        if item.strip()
    }
    return {
        "endpoint_preset": endpoint_preset
        or str(getattr(train_args, "endpoint_preset", "kimodo_ee")),
        "endpoint_subset_mode": endpoint_subset_mode
        or str(getattr(train_args, "endpoint_subset_mode", "random_nonempty")),
        "endpoint_root_ref_mode": root_ref_mode,
        "max_control_keyframes": int(
            max_control_keyframes
            or int(getattr(train_args, "max_control_keyframes", 8))
        ),
        "include_root_ref_for_endpoints": root_ref_mode == "kimodo_hidden_root",
        "include_endpoint_rotations": (
            bool(getattr(train_args, "control_include_endpoint_rotations", False))
            if include_endpoint_rotations is None
            else bool(include_endpoint_rotations)
        ),
        "include_contact_pattern": "contact" in configured_modes,
    }


def resolve_sampling_control_modes(
    train_args: argparse.Namespace, requested_modes: str = ""
) -> tuple[str, ...]:
    raw_modes = requested_modes or str(getattr(train_args, "control_modes", ""))
    modes = tuple(
        item.strip() for item in raw_modes.split(",") if item.strip() and item.strip() != "none"
    )
    if modes:
        return modes
    return ("root_sparse", "root_dense", "endpoints", "fullpose", "mixed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--text_root", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", default="generation/hy273_raw_flow")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument(
        "--indices",
        default="",
        help="Comma-separated dataset indices to sample. Defaults to the first --num_samples items.",
    )
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--control_cfg_scale", type=float, default=None)
    parser.add_argument("--contact_init", choices=["random", "zeros", "half"], default="random")
    parser.add_argument("--contact_feedback", choices=["blend", "prob", "fixed"], default="blend")
    parser.add_argument(
        "--cfg_apply_contacts",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--weight_source", choices=["auto", "model", "ema"], default="auto")
    parser.add_argument(
        "--control_modes",
        default="",
        help="Comma-separated modes. Empty reuses the checkpoint protocol without its none mode.",
    )
    parser.add_argument(
        "--control_mode_schedule",
        default="",
        help="Optional comma-separated mode per selected sample; disables random mode selection.",
    )
    parser.add_argument("--endpoint_preset", choices=["kimodo_ee", "five_point"], default="")
    parser.add_argument(
        "--endpoint_subset_mode",
        choices=["all", "random_nonempty"],
        default="",
    )
    parser.add_argument(
        "--endpoint_root_ref_mode",
        choices=["kimodo_hidden_root", "none"],
        default="",
    )
    parser.add_argument("--max_control_keyframes", type=int, default=0)
    parser.add_argument("--c_dir_mode", choices=["dataset", "forward"], default="dataset")
    parser.add_argument("--anchor_first_frame", action="store_true")
    parser.add_argument("--text_encoder", choices=["clip", "hy_cache", "hytext_cache", "qwen_clip_cache", "none"], default="")
    parser.add_argument("--hytext_cache_dir", default="")
    parser.add_argument("--hytext_ctxt_dim", type=int, default=0)
    parser.add_argument("--hytext_vtxt_dim", type=int, default=0)
    parser.add_argument("--hytext_max_open_shards", type=int, default=0)
    parser.add_argument("--hytext_allow_cache_miss", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    # Training checkpoints also contain optimizer and EMA states. Memory-map the
    # zip archive so sampling only pages in tensors selected below instead of
    # eagerly materializing every multi-gigabyte storage.
    ckpt = torch.load(args.checkpoint, map_location="cpu", mmap=True)
    train_args = argparse.Namespace(**ckpt.get("args", {}))
    cfg_scale, control_cfg_scale = resolve_sampling_cfg_scales(
        train_args, args.cfg_scale, args.control_cfg_scale
    )
    apply_checkpoint_path_override(train_args, "data_root", args.data_root)
    apply_checkpoint_path_override(train_args, "text_root", args.text_root)
    if args.text_encoder:
        train_args.text_encoder = args.text_encoder
    apply_checkpoint_path_override(
        train_args, "hytext_cache_dir", args.hytext_cache_dir
    )
    if args.hytext_ctxt_dim > 0:
        train_args.hytext_ctxt_dim = args.hytext_ctxt_dim
    if args.hytext_vtxt_dim > 0:
        train_args.hytext_vtxt_dim = args.hytext_vtxt_dim
    if args.hytext_max_open_shards > 0:
        train_args.hytext_max_open_shards = args.hytext_max_open_shards
    if args.hytext_allow_cache_miss:
        if str(getattr(train_args, "architecture", "one_stage")) == "redenoise_kimodo_like":
            raise RuntimeError("Pinned redenoise_kimodo_like assets require strict HYText cache lookup")
        train_args.hytext_allow_cache_miss = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    verify_checkpoint_assets(train_args)
    model = create_model(train_args).to(device)
    state_dict, weight_source = checkpoint_weight_state(
        ckpt, args.weight_source, args.checkpoint
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    normalizer = checkpoint_normalizer(ckpt, train_args, device, args.checkpoint)
    dataset = Kimodo273TextDataset(
        train_args.data_root,
        split=args.split,
        text_root=train_args.text_root or None,
        max_frames=train_args.max_frames,
        random_crop=False,
        deterministic_text=True,
    )
    if args.indices:
        sample_indices = [int(part.strip()) for part in args.indices.split(",") if part.strip()]
        if not sample_indices:
            raise ValueError("--indices was provided but no valid indices were parsed")
        for idx in sample_indices:
            if idx < 0 or idx >= len(dataset):
                raise IndexError(f"Dataset index {idx} out of range for split {args.split} with {len(dataset)} items")
    else:
        sample_indices = list(range(min(args.num_samples, len(dataset))))
    samples = [dataset[i] for i in sample_indices]
    batch = collate_kimodo273_text(samples)
    motion = batch["motion"].to(device)
    lengths = batch["lengths"].to(device)
    transform = apply_kimodo_training_transform(motion, random_heading=False, root_shift=True)
    motion = transform.motion
    endpoint_protocol = resolve_endpoint_protocol(
        train_args,
        endpoint_preset=args.endpoint_preset,
        endpoint_subset_mode=args.endpoint_subset_mode,
        endpoint_root_ref_mode=args.endpoint_root_ref_mode,
        max_control_keyframes=args.max_control_keyframes,
    )
    sampling_control_modes = resolve_sampling_control_modes(train_args, args.control_modes)
    mode_schedule = None
    if args.control_mode_schedule:
        mode_schedule = tuple(
            item.strip()
            for item in args.control_mode_schedule.split(",")
            if item.strip()
        )
        if len(mode_schedule) != motion.shape[0]:
            raise ValueError(
                "--control_mode_schedule must contain exactly one mode per selected sample"
            )
        sampling_control_modes = tuple(dict.fromkeys(mode_schedule))
    include_contact_pattern = "contact" in sampling_control_modes
    controls = build_synthetic_control_batch(
        motion,
        lengths,
        modes=sampling_control_modes,
        endpoint_preset=str(endpoint_protocol["endpoint_preset"]),
        endpoint_subset_mode=str(endpoint_protocol["endpoint_subset_mode"]),
        max_keyframes=int(endpoint_protocol["max_control_keyframes"]),
        include_root_ref_for_endpoints=bool(
            endpoint_protocol["include_root_ref_for_endpoints"]
        ),
        include_endpoint_rotations=bool(
            endpoint_protocol["include_endpoint_rotations"]
        ),
        include_contact_pattern=include_contact_pattern,
        dense_min_fraction=float(
            getattr(train_args, "control_dense_min_fraction", 0.25)
        ),
        mode_schedule=mode_schedule,
    )
    if args.c_dir_mode == "forward":
        c_dir = torch.zeros(motion.shape[0], 2, device=device, dtype=motion.dtype)
        c_dir[:, 0] = 1.0
    else:
        c_dir = transform.c_dir.to(device)
    if args.anchor_first_frame:
        controls.observed_motion, controls.motion_mask = anchor_first_frame_convention(
            controls.observed_motion,
            controls.motion_mask,
            c_dir,
        )
    prediction_type = str(getattr(train_args, "prediction_type", "velocity"))
    representation_loss_space = resolve_representation_loss_space(
        prediction_type,
        str(getattr(train_args, "representation_loss_space", "auto")),
    )
    # The JiT epsilon caps the training loss weight only. ODE integration must
    # retain the rectified-flow vector field and uses only a numerical floor.
    sampling_velocity_t_eps = 1e-4
    sample_output = sample_ode(
        model,
        normalizer,
        lengths,
        batch["texts"],
        controls.observed_motion,
        controls.motion_mask,
        c_dir,
        num_steps=args.num_steps,
        self_conditioning=bool(getattr(train_args, "self_conditioning", False)),
        cfg_scale=cfg_scale,
        control_cfg_scale=control_cfg_scale,
        contact_init=args.contact_init,
        contact_feedback=args.contact_feedback,
        cfg_apply_contacts=(
            include_contact_pattern
            if args.cfg_apply_contacts is None
            else bool(args.cfg_apply_contacts)
        ),
        prediction_type=prediction_type,
        velocity_t_eps=sampling_velocity_t_eps,
        return_details=True,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert isinstance(sample_output, ODESampleOutput)
    np.save(out_dir / "samples.npy", sample_output.raw_motion.cpu().numpy())
    np.save(out_dir / "samples_exact_clamped.npy", sample_output.exact_clamped_motion.cpu().numpy())
    np.save(out_dir / "final_clean_prediction.npy", sample_output.final_clean_prediction.cpu().numpy())
    for branch_name, branch_value in sample_output.final_branch_predictions.items():
        np.save(out_dir / f"final_branch_{branch_name}.npy", branch_value.cpu().numpy())
    np.save(out_dir / "observed.npy", controls.observed_motion.cpu().numpy())
    np.save(out_dir / "mask.npy", controls.motion_mask.cpu().numpy())
    lengths_np = lengths.detach().cpu().numpy().astype(np.int64, copy=False)
    np.save(out_dir / "lengths.npy", lengths_np)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "num_steps": args.num_steps,
                "cfg_scale": cfg_scale,
                "control_cfg_scale": control_cfg_scale,
                "ode_state_persistent_clamp": False,
                "primary_output": "samples.npy (raw, pre exact-clamp)",
                "exact_output": "samples_exact_clamped.npy",
                "contact_init": args.contact_init,
                "contact_feedback": args.contact_feedback,
                "cfg_apply_contacts": (
                    include_contact_pattern
                    if args.cfg_apply_contacts is None
                    else bool(args.cfg_apply_contacts)
                ),
                "seed": int(args.seed),
                "weight_source": weight_source,
                "prediction_type": str(getattr(train_args, "prediction_type", "velocity")),
                "representation_loss_space": representation_loss_space,
                "velocity_loss_t_eps": float(
                    getattr(train_args, "velocity_loss_t_eps", 0.05)
                ),
                "sampling_velocity_t_eps": sampling_velocity_t_eps,
                "lengths": lengths_np.tolist(),
                "texts": batch["texts"],
                "rel_paths": batch["rel_paths"],
                "indices": sample_indices,
                "control_modes": controls.mode_ids,
                "requested_control_mode_schedule": mode_schedule,
                "endpoint_protocol": endpoint_protocol,
                "c_dir_mode": args.c_dir_mode,
                "anchor_first_frame": bool(args.anchor_first_frame),
            },
            indent=2,
        )
    )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
