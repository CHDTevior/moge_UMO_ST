"""Rectified-flow utilities for HY273 raw-space training and sampling."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .hy273_slices import CONTACT_SLICE, CONT_DIM, DIM_HY273


LEGACY_SPLIT_CONTACT_PROTOCOL = "split_contact_logits_v1"
UNIFIED_273_CONTACT_PROTOCOL = "unified_273_clean_flow_v1"


def uses_unified_273_flow(contact_protocol: str) -> bool:
    protocol = str(contact_protocol)
    if protocol not in {
        LEGACY_SPLIT_CONTACT_PROTOCOL,
        UNIFIED_273_CONTACT_PROTOCOL,
    }:
        raise ValueError(f"Unknown HY273 contact protocol: {protocol!r}")
    return protocol == UNIFIED_273_CONTACT_PROTOCOL


def lengths_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    lengths = lengths.long()
    if max_len is None:
        max_len = int(lengths.max().item())
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    schedule: str = "logit_normal",
    p_mean: float = 0.0,
    p_std: float = 1.0,
    eps: float = 1e-4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if schedule == "uniform":
        return torch.rand(batch_size, device=device, generator=generator).clamp(eps, 1.0 - eps)
    if schedule == "logit_normal":
        normal = torch.randn(batch_size, device=device, generator=generator)
        return torch.sigmoid(normal * float(p_std) + float(p_mean)).clamp(eps, 1.0 - eps)
    raise ValueError(f"Unknown timestep schedule: {schedule}")


def make_ode_grid(num_steps: int, device: torch.device) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    return torch.linspace(0.0, 1.0, num_steps + 1, device=device)


def clean_x0_euler_step(
    state: torch.Tensor,
    clean_x0: torch.Tensor,
    *,
    timestep: torch.Tensor | float,
    dt: torch.Tensor | float,
    velocity_t_eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance an x0-prediction rectified-flow state by one Euler step."""

    if state.shape != clean_x0.shape or state.ndim < 2:
        raise ValueError("state and clean_x0 must have the same batched shape")
    if velocity_t_eps <= 0:
        raise ValueError("velocity_t_eps must be positive")

    def _schedule_view(value: torch.Tensor | float, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=state.device, dtype=state.dtype)
        if tensor.ndim == 0:
            return tensor.view(*([1] * state.ndim))
        if tensor.shape == (state.shape[0],):
            return tensor.view(state.shape[0], *([1] * (state.ndim - 1)))
        expected = (state.shape[0],) + (1,) * (state.ndim - 1)
        if tensor.shape == expected:
            return tensor
        raise ValueError(
            f"{name} must be scalar, [B], or {expected}; got {tuple(tensor.shape)}"
        )

    t_view = _schedule_view(timestep, "timestep")
    dt_view = _schedule_view(dt, "dt")
    if not bool(torch.isfinite(t_view).all() and torch.isfinite(dt_view).all()):
        raise ValueError("timestep and dt must be finite")
    denominator = (1.0 - t_view).clamp_min(float(velocity_t_eps))
    velocity = (clean_x0 - state) / denominator
    return state + dt_view * velocity, velocity


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


def mse_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((pred - target).square(), mask)


def smooth_l1_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    return masked_mean(loss, mask)


def bce_logits_masked(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return masked_mean(loss, mask)


def clean_from_velocity(z_cont_imp: torch.Tensor, t: torch.Tensor, v_pred_cont: torch.Tensor) -> torch.Tensor:
    t_view = t.view(-1, 1, 1).to(device=z_cont_imp.device, dtype=z_cont_imp.dtype)
    return z_cont_imp + (1.0 - t_view) * v_pred_cont


def build_flow_state(
    x0: torch.Tensor,
    observed: torch.Tensor,
    motion_mask: torch.Tensor,
    t: torch.Tensor,
    noise_cont: Optional[torch.Tensor] = None,
    contact_aux: Optional[torch.Tensor] = None,
    noise_contact: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    if x0.shape[-1] != DIM_HY273:
        raise ValueError(f"Expected x0 [B,T,{DIM_HY273}], got {tuple(x0.shape)}")
    t_view = t.view(-1, 1, 1).to(device=x0.device, dtype=x0.dtype)
    x0_cont = x0[..., :CONT_DIM]
    x0_contact = x0[..., CONTACT_SLICE]
    if noise_cont is None:
        noise_cont = torch.randn_like(x0_cont)
    if contact_aux is None:
        # Contacts are not an ODE state. Keep the auxiliary input in probability
        # space so train-time contact channels match sampling-time feedback.
        contact_aux = torch.rand_like(x0_contact) if noise_contact is None else noise_contact.sigmoid()
    z_cont = t_view * x0_cont + (1.0 - t_view) * noise_cont
    z_contact = t_view * x0_contact + (1.0 - t_view) * contact_aux.clamp(0.0, 1.0)
    v_target_cont = x0_cont - noise_cont
    mask_cont = motion_mask[..., :CONT_DIM].to(dtype=x0.dtype)
    mask_contact = motion_mask[..., CONTACT_SLICE].to(dtype=x0.dtype)
    z_cont_imp = z_cont * (1.0 - mask_cont) + observed[..., :CONT_DIM] * mask_cont
    z_contact_imp = z_contact * (1.0 - mask_contact) + observed[..., CONTACT_SLICE] * mask_contact
    z_imp = torch.cat([z_cont_imp, z_contact_imp], dim=-1)
    model_in = torch.cat([z_imp, motion_mask.to(dtype=x0.dtype)], dim=-1)
    return {
        "model_in": model_in,
        "z_imp": z_imp,
        "z_cont_imp": z_cont_imp,
        "v_target_cont": v_target_cont,
        "x0_cont": x0_cont,
        "x0_contact": x0_contact,
    }


def build_unified_273_flow_state(
    x0: torch.Tensor,
    observed: torch.Tensor,
    motion_mask: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Build one coherent Gaussian rectified-flow state over all HY273 channels."""

    if x0.shape[-1] != DIM_HY273:
        raise ValueError(f"Expected x0 [B,T,{DIM_HY273}], got {tuple(x0.shape)}")
    if observed.shape != x0.shape or motion_mask.shape != x0.shape:
        raise ValueError("observed and motion_mask must match x0 [B,T,273]")
    if t.shape != (x0.shape[0],):
        raise ValueError(f"Expected t [{x0.shape[0]}], got {tuple(t.shape)}")
    if noise is None:
        noise = torch.randn_like(x0)
    elif noise.shape != x0.shape:
        raise ValueError(f"Expected noise {tuple(x0.shape)}, got {tuple(noise.shape)}")

    t_view = t.view(-1, 1, 1).to(device=x0.device, dtype=x0.dtype)
    z = t_view * x0 + (1.0 - t_view) * noise
    mask = motion_mask.to(device=x0.device, dtype=x0.dtype)
    z_imp = z * (1.0 - mask) + observed * mask
    return {
        "model_in": torch.cat([z_imp, mask], dim=-1),
        "z_imp": z_imp,
        "noise": noise,
        "v_target": x0 - noise,
        "x0": x0,
    }
