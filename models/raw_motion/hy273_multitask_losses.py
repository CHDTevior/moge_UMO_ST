"""Frozen HY273 multitask loss contract with auditable ratio reductions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

import torch
import torch.nn.functional as F

from .hy273_normalizer import HY273Normalizer
from .hy273_slices import (
    CONTACT_JOINTS,
    CONTACT_SLICE,
    CONT_DIM,
    DIM_HY273,
    GLOBAL_ROT_SLICE,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    ROOT_SLICE,
    VELOCITY_SLICE,
    fk_positions_from_global_rot6d,
    reconstruct_global_joints_from_features,
)


SEMANTIC_WEIGHTS: Mapping[str, float] = {
    "repr_root_xyz": 10.0,
    "repr_heading": 2.0,
    "repr_joint_position": 10.0,
    "repr_global_rot6d": 10.0,
    "repr_velocity": 3.0,
}
SEMANTIC_SLICES = {
    "repr_root_xyz": ROOT_SLICE,
    "repr_heading": HEADING_SLICE,
    "repr_joint_position": JOINT_POS_SLICE,
    "repr_global_rot6d": GLOBAL_ROT_SLICE,
    "repr_velocity": VELOCITY_SLICE,
}
SEMANTIC_WEIGHT_SUM = float(sum(SEMANTIC_WEIGHTS.values()))
KIMODO_CONTACT_SEMANTIC_WEIGHT = 4.0
REPRESENTATION_LOSS_SPACES = (
    "velocity_mse",
    "clean_x0_mse",
    "clean_x0_smooth_l1",
)
R13_UNIFIED_CONTACT_WEIGHT = (
    0.09397019716051493
    * KIMODO_CONTACT_SEMANTIC_WEIGHT
    / SEMANTIC_WEIGHT_SUM
)
R13_UNIFIED_CONTROL_CONTACT_WEIGHT = (
    0.25 * KIMODO_CONTACT_SEMANTIC_WEIGHT / SEMANTIC_WEIGHT_SUM
)


@dataclass(frozen=True)
class HY273MultitaskLossWeights:
    """The legacy R11 coefficients; R13 overrides contact after unifying the flow."""

    representation_scale: float = 0.09397019716051493
    contact: float = 0.10
    clean_root_velocity: float = 0.01
    clean_joint_velocity: float = 0.01
    foot_lock: float = 0.01
    fk_consistency: float = 0.07
    control_continuous: float = 0.25
    control_contact: float = 0.05
    velocity_t_eps: float = 0.05
    fk_warmup_steps: int = 5_000
    fk_scale_m: float = 0.05
    fps: float = 30.0
    contact_threshold: float = 0.5

    def validate(self) -> None:
        values = {
            "representation_scale": self.representation_scale,
            "contact": self.contact,
            "clean_root_velocity": self.clean_root_velocity,
            "clean_joint_velocity": self.clean_joint_velocity,
            "foot_lock": self.foot_lock,
            "fk_consistency": self.fk_consistency,
            "control_continuous": self.control_continuous,
            "control_contact": self.control_contact,
            "velocity_t_eps": self.velocity_t_eps,
            "fk_scale_m": self.fk_scale_m,
            "fps": self.fps,
        }
        if any(float(value) < 0.0 for value in values.values()):
            raise ValueError(f"Loss coefficients must be non-negative: {values}")
        if self.velocity_t_eps <= 0.0 or self.fk_scale_m <= 0.0 or self.fps <= 0.0:
            raise ValueError("velocity_t_eps, fk_scale_m, and fps must be positive")
        if int(self.fk_warmup_steps) < 0:
            raise ValueError("fk_warmup_steps must be non-negative")

    def fk_weight(self, global_step: int) -> tuple[float, float]:
        if self.fk_warmup_steps <= 0:
            return float(self.fk_consistency), 1.0
        factor = min(
            max((int(global_step) + 1) / float(self.fk_warmup_steps), 0.0),
            1.0,
        )
        return float(self.fk_consistency) * factor, factor


@dataclass
class RatioLossTerm:
    """One local ratio used by backward and its explicit accounting fields."""

    name: str
    group: str
    numerator: torch.Tensor
    denominator: torch.Tensor
    weight: float
    backward_denominator: torch.Tensor | None = None
    backward_numerator_scale: float = 1.0

    @property
    def raw(self) -> torch.Tensor:
        denominator = (
            self.denominator
            if self.backward_denominator is None
            else self.backward_denominator
        )
        return (
            self.numerator
            * float(self.backward_numerator_scale)
            / denominator.clamp_min(1.0)
        )

    @property
    def weighted(self) -> torch.Tensor:
        return self.raw * float(self.weight)


@dataclass(frozen=True)
class _ElementLoss:
    name: str
    group: str
    values: torch.Tensor
    mask: torch.Tensor
    weight: float


@dataclass
class HY273MultitaskLossBundle:
    total: torch.Tensor
    terms: dict[str, RatioLossTerm]
    fk_warmup_factor: float
    fk_distance_cm: RatioLossTerm
    _elements: dict[str, _ElementLoss]
    _fk_distance_element: _ElementLoss

    def terms_for_samples(self, sample_selector: torch.Tensor) -> dict[str, RatioLossTerm]:
        """Build detached monitoring ratios for one capability without changing backward."""

        if sample_selector.ndim != 1:
            raise ValueError("sample_selector must have shape [B]")
        return {
            name: _ratio_term(spec, sample_selector=sample_selector)
            for name, spec in self._elements.items()
        }

    def group_contributions(self) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        for term in self.terms.values():
            output[term.group] = output.get(term.group, self.total * 0.0) + term.weighted
        return output

    def fk_distance_for_samples(
        self, sample_selector: torch.Tensor
    ) -> RatioLossTerm:
        return _ratio_term(
            self._fk_distance_element, sample_selector=sample_selector
        )


def _expanded_mask(mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    result = mask.to(device=values.device, dtype=torch.bool)
    while result.ndim < values.ndim:
        result = result.unsqueeze(-1)
    try:
        return result.expand_as(values)
    except RuntimeError as exc:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} cannot cover values {tuple(values.shape)}"
        ) from exc


def _ratio_term(
    spec: _ElementLoss,
    *,
    sample_selector: torch.Tensor | None = None,
) -> RatioLossTerm:
    mask = _expanded_mask(spec.mask, spec.values)
    if sample_selector is not None:
        selector = sample_selector.to(device=spec.values.device, dtype=torch.bool)
        if selector.shape != (spec.values.shape[0],):
            raise ValueError(
                f"Expected sample_selector [{spec.values.shape[0]}], got {tuple(selector.shape)}"
            )
        selector = selector.view(selector.shape[0], *([1] * (spec.values.ndim - 1)))
        mask = mask & selector
    mask_f = mask.to(dtype=spec.values.dtype)
    return RatioLossTerm(
        name=spec.name,
        group=spec.group,
        numerator=(spec.values * mask_f).sum(),
        denominator=mask_f.sum(),
        weight=float(spec.weight),
    )


def _smooth_l1_values(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, target, reduction="none", beta=1.0)


def compute_hy273_multitask_loss(
    *,
    x0_hat_cont: torch.Tensor,
    contact_logits: torch.Tensor,
    z_cont_imputed: torch.Tensor,
    x0_target_norm: torch.Tensor,
    x0_target_physical: torch.Tensor,
    hard_observed_norm: torch.Tensor,
    hard_mask: torch.Tensor,
    target_valid: torch.Tensor,
    timesteps: torch.Tensor,
    normalizer: HY273Normalizer,
    global_step: int,
    weights: HY273MultitaskLossWeights | None = None,
    unified_contact_flow: bool = False,
    representation_loss_space: str = "velocity_mse",
    contact_loss_space: str | None = None,
    representation_multiplier: float = 1.0,
    contact_multiplier: float = 1.0,
) -> HY273MultitaskLossBundle:
    """Compute the R11 target loss using rank-local ratio-of-sums.

    The returned numerators and denominators are intentionally not all-reduced.
    DDP must average gradients of each rank-local ratio. Monitoring code may sum
    detached numerators/denominators across ranks to report an unbiased ratio.
    """

    weights = weights or HY273MultitaskLossWeights()
    weights.validate()
    representation_loss_space = str(representation_loss_space)
    if representation_loss_space not in REPRESENTATION_LOSS_SPACES:
        raise ValueError(
            "representation_loss_space must be one of "
            f"{REPRESENTATION_LOSS_SPACES}, got {representation_loss_space!r}"
        )
    contact_loss_space = (
        representation_loss_space
        if contact_loss_space is None
        else str(contact_loss_space)
    )
    if contact_loss_space not in REPRESENTATION_LOSS_SPACES:
        raise ValueError(
            "contact_loss_space must be one of "
            f"{REPRESENTATION_LOSS_SPACES}, got {contact_loss_space!r}"
        )
    representation_multiplier = float(representation_multiplier)
    contact_multiplier = float(contact_multiplier)
    if not math.isfinite(representation_multiplier) or representation_multiplier < 0:
        raise ValueError("representation_multiplier must be finite and non-negative")
    if not math.isfinite(contact_multiplier) or contact_multiplier < 0:
        raise ValueError("contact_multiplier must be finite and non-negative")
    expected_cont = x0_target_norm[..., :CONT_DIM]
    expected_contact = x0_target_norm[..., CONTACT_SLICE]
    if x0_hat_cont.shape != expected_cont.shape:
        raise ValueError(
            f"x0_hat_cont shape {tuple(x0_hat_cont.shape)} != {tuple(expected_cont.shape)}"
        )
    if contact_logits.shape != expected_contact.shape:
        raise ValueError("contact prediction shape does not match target contacts")
    if hard_mask.shape != x0_target_norm.shape or hard_observed_norm.shape != x0_target_norm.shape:
        raise ValueError("hard control tensors must match target [B,T,273]")
    if target_valid.shape != x0_target_norm.shape[:2]:
        raise ValueError("target_valid must have shape [B,T]")

    valid = target_valid.to(device=x0_hat_cont.device, dtype=torch.bool)
    hard_mask = hard_mask.to(device=x0_hat_cont.device, dtype=torch.bool)
    t_view = timesteps.to(device=x0_hat_cont.device, dtype=x0_hat_cont.dtype).view(-1, 1, 1)
    velocity_denom = (1.0 - t_view).clamp_min(float(weights.velocity_t_eps))
    velocity_prediction = (x0_hat_cont - z_cont_imputed) / velocity_denom
    velocity_target = (expected_cont - z_cont_imputed) / velocity_denom
    velocity_sq = (velocity_prediction - velocity_target).square()
    if representation_loss_space == "velocity_mse":
        representation_values = velocity_sq
    elif representation_loss_space == "clean_x0_mse":
        representation_values = (x0_hat_cont - expected_cont).square()
    else:
        representation_values = _smooth_l1_values(x0_hat_cont, expected_cont)

    elements: dict[str, _ElementLoss] = {}
    unmasked_cont = valid[..., None] & ~hard_mask[..., :CONT_DIM]
    for name, block_slice in SEMANTIC_SLICES.items():
        elements[name] = _ElementLoss(
            name=name,
            group="representation",
            values=representation_values[..., block_slice],
            mask=unmasked_cont[..., block_slice],
            weight=(
                float(weights.representation_scale)
                * float(SEMANTIC_WEIGHTS[name])
                / SEMANTIC_WEIGHT_SUM
                * representation_multiplier
            ),
        )

    if unified_contact_flow:
        if contact_loss_space == "velocity_mse":
            contact_values = (
                (contact_logits - expected_contact) / velocity_denom
            ).square()
        elif contact_loss_space == "clean_x0_mse":
            contact_values = (contact_logits - expected_contact).square()
        else:
            contact_values = _smooth_l1_values(contact_logits, expected_contact)
        elements["contact_all"] = _ElementLoss(
            name="contact_all",
            group="contact",
            values=contact_values,
            mask=valid[..., None] & ~hard_mask[..., CONTACT_SLICE],
            weight=float(weights.contact) * contact_multiplier,
        )
    else:
        elements["contact_all"] = _ElementLoss(
            name="contact_all",
            group="contact",
            values=F.binary_cross_entropy_with_logits(
                contact_logits, expected_contact, reduction="none"
            ),
            mask=valid[..., None],
            weight=float(weights.contact),
        )
    elements["control_continuous"] = _ElementLoss(
        name="control_continuous",
        group="control_continuous",
        values=_smooth_l1_values(
            x0_hat_cont, hard_observed_norm[..., :CONT_DIM]
        ),
        mask=valid[..., None] & hard_mask[..., :CONT_DIM],
        weight=float(weights.control_continuous),
    )
    contact_control_values = (
        _smooth_l1_values(
            contact_logits,
            hard_observed_norm[..., CONTACT_SLICE],
        )
        if unified_contact_flow
        else F.binary_cross_entropy_with_logits(
            contact_logits,
            hard_observed_norm[..., CONTACT_SLICE],
            reduction="none",
        )
    )
    elements["control_contact"] = _ElementLoss(
        name="control_contact",
        group="control_contact",
        values=contact_control_values,
        mask=valid[..., None] & hard_mask[..., CONTACT_SLICE],
        weight=float(weights.control_contact),
    )

    x0_hat_norm = torch.cat(
        [
            x0_hat_cont,
            contact_logits if unified_contact_flow else torch.sigmoid(contact_logits),
        ],
        dim=-1,
    )
    with torch.autocast(device_type=x0_hat_cont.device.type, enabled=False):
        prediction_physical = normalizer.denormalize(x0_hat_norm.float())
        target_physical = x0_target_physical.to(
            device=x0_hat_cont.device, dtype=torch.float32
        )
        valid_pair = valid[:, 1:] & valid[:, :-1]
        fps = float(weights.fps)

        pred_root_velocity = (
            prediction_physical[:, 1:, ROOT_SLICE]
            - prediction_physical[:, :-1, ROOT_SLICE]
        ) * fps
        target_root_velocity = (
            target_physical[:, 1:, ROOT_SLICE]
            - target_physical[:, :-1, ROOT_SLICE]
        ) * fps
        elements["clean_root_velocity"] = _ElementLoss(
            name="clean_root_velocity",
            group="clean_root_velocity",
            values=_smooth_l1_values(pred_root_velocity, target_root_velocity),
            mask=valid_pair[..., None],
            weight=float(weights.clean_root_velocity),
        )

        pred_joints = reconstruct_global_joints_from_features(prediction_physical)
        target_joints = reconstruct_global_joints_from_features(target_physical)
        pred_joint_velocity = (pred_joints[:, 1:] - pred_joints[:, :-1]) * fps
        target_joint_velocity = (target_joints[:, 1:] - target_joints[:, :-1]) * fps
        elements["clean_joint_velocity"] = _ElementLoss(
            name="clean_joint_velocity",
            group="clean_joint_velocity",
            values=_smooth_l1_values(pred_joint_velocity, target_joint_velocity),
            mask=valid_pair[..., None, None],
            weight=float(weights.clean_joint_velocity),
        )

        contact_gt = target_physical[..., CONTACT_SLICE] > float(
            weights.contact_threshold
        )
        contact_pair = contact_gt[:, 1:] & contact_gt[:, :-1]
        contact_pair = contact_pair & valid_pair[..., None]
        pred_foot_velocity = pred_joint_velocity[:, :, list(CONTACT_JOINTS)]
        elements["foot_lock"] = _ElementLoss(
            name="foot_lock",
            group="foot_lock",
            values=_smooth_l1_values(
                pred_foot_velocity, torch.zeros_like(pred_foot_velocity)
            ),
            mask=contact_pair[..., None],
            weight=float(weights.foot_lock),
        )

        hard_mask_f = hard_mask.to(dtype=torch.float32)
        clamped_norm = (
            x0_hat_norm.float() * (1.0 - hard_mask_f)
            + hard_observed_norm.float() * hard_mask_f
        )
        clamped_physical = normalizer.denormalize(clamped_norm)
        joints_from_position = reconstruct_global_joints_from_features(clamped_physical)
        joints_from_fk = fk_positions_from_global_rot6d(clamped_physical)
        fk_residual = (joints_from_fk - joints_from_position) / float(weights.fk_scale_m)
        fk_weight, fk_warmup_factor = weights.fk_weight(global_step)
        elements["fk_consistency"] = _ElementLoss(
            name="fk_consistency",
            group="fk_consistency",
            values=_smooth_l1_values(fk_residual, torch.zeros_like(fk_residual)),
            mask=valid[..., None, None],
            weight=fk_weight,
        )
        fk_distance_cm_spec = _ElementLoss(
            name="fk_distance_cm",
            group="diagnostic",
            values=(joints_from_fk - joints_from_position).norm(dim=-1) * 100.0,
            mask=valid[..., None],
            weight=0.0,
        )

    terms = {name: _ratio_term(spec) for name, spec in elements.items()}
    total = sum((term.weighted for term in terms.values()), x0_hat_cont.sum() * 0.0)
    return HY273MultitaskLossBundle(
        total=total,
        terms=terms,
        fk_warmup_factor=fk_warmup_factor,
        fk_distance_cm=_ratio_term(fk_distance_cm_spec),
        _elements=elements,
        _fk_distance_element=fk_distance_cm_spec,
    )


def compute_hy273_unified_flow_loss(
    *,
    x0_hat_norm: torch.Tensor,
    z_imputed: torch.Tensor,
    x0_target_norm: torch.Tensor,
    x0_target_physical: torch.Tensor,
    hard_observed_norm: torch.Tensor,
    hard_mask: torch.Tensor,
    target_valid: torch.Tensor,
    timesteps: torch.Tensor,
    normalizer: HY273Normalizer,
    global_step: int,
    weights: HY273MultitaskLossWeights | None = None,
    representation_loss_space: str = "velocity_mse",
    contact_loss_space: str | None = None,
    representation_multiplier: float = 1.0,
    contact_multiplier: float = 1.0,
) -> HY273MultitaskLossBundle:
    """R13 loss where all 273 outputs mean normalized clean-state values."""

    if (
        x0_hat_norm.shape != x0_target_norm.shape
        or x0_hat_norm.shape[-1] != DIM_HY273
    ):
        raise ValueError("x0_hat_norm must match x0_target_norm [B,T,273]")
    if z_imputed.shape != x0_hat_norm.shape:
        raise ValueError("z_imputed must match x0_hat_norm [B,T,273]")
    if not normalizer.normalize_contacts:
        raise ValueError("Unified 273D loss requires contact-normalizing stats")
    if weights is None:
        weights = replace(
            HY273MultitaskLossWeights(),
            contact=R13_UNIFIED_CONTACT_WEIGHT,
            control_contact=R13_UNIFIED_CONTROL_CONTACT_WEIGHT,
        )
    return compute_hy273_multitask_loss(
        x0_hat_cont=x0_hat_norm[..., :CONT_DIM],
        contact_logits=x0_hat_norm[..., CONTACT_SLICE],
        z_cont_imputed=z_imputed[..., :CONT_DIM],
        x0_target_norm=x0_target_norm,
        x0_target_physical=x0_target_physical,
        hard_observed_norm=hard_observed_norm,
        hard_mask=hard_mask,
        target_valid=target_valid,
        timesteps=timesteps,
        normalizer=normalizer,
        global_step=global_step,
        weights=weights,
        unified_contact_flow=True,
        representation_loss_space=representation_loss_space,
        contact_loss_space=contact_loss_space,
        representation_multiplier=representation_multiplier,
        contact_multiplier=contact_multiplier,
    )
