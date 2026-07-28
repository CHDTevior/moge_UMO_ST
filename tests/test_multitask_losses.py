from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from models.raw_motion.hy273_multitask_losses import (
    HY273MultitaskLossWeights,
    compute_hy273_multitask_loss,
)
from models.raw_motion.hy273_normalizer import HY273Normalizer
from models.raw_motion.hy273_slices import CONTACT_SLICE, DIM_HY273
from models.raw_motion.flow_schedule import bce_logits_masked, smooth_l1_masked
from train_hy273_raw_flow import (
    compute_clean_semantic_losses,
    effective_fk_consistency_weight,
    fk_position_consistency_loss,
    representation_loss_pair,
    representation_mse_loss,
)


def _normalizer() -> HY273Normalizer:
    return HY273Normalizer(torch.zeros(DIM_HY273), torch.ones(DIM_HY273))


def _valid_motion(batch: int = 2, frames: int = 4) -> torch.Tensor:
    motion = torch.zeros(batch, frames, DIM_HY273)
    motion[..., 3] = 1.0
    rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    motion[..., 71:203] = rot.repeat(22)
    motion[..., CONTACT_SLICE] = 1.0
    return motion


def _bundle(mask: torch.Tensor | None = None):
    target = _valid_motion()
    mask = torch.zeros_like(target, dtype=torch.bool) if mask is None else mask
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    t = torch.tensor([0.25, 0.99])
    z = torch.randn(2, 4, 269)
    prediction = target[..., :269].clone().requires_grad_(True)
    logits = torch.zeros(2, 4, 4, requires_grad=True)
    return compute_hy273_multitask_loss(
        x0_hat_cont=prediction,
        contact_logits=logits,
        z_cont_imputed=z,
        x0_target_norm=target,
        x0_target_physical=target,
        hard_observed_norm=target,
        hard_mask=mask,
        target_valid=valid,
        timesteps=t,
        normalizer=_normalizer(),
        global_step=0,
    ), prediction, logits


def test_frozen_weights_and_fk_schedule() -> None:
    weights = HY273MultitaskLossWeights()
    assert np.isclose(weights.fk_weight(0)[0], 0.07 / 5000)
    assert np.isclose(weights.fk_weight(4999)[0], 0.07)
    assert np.isclose(weights.fk_weight(5000)[0], 0.07)


def test_x0_exact_has_zero_continuous_representation() -> None:
    bundle, prediction, logits = _bundle()
    for name in (
        "repr_root_xyz",
        "repr_heading",
        "repr_joint_position",
        "repr_global_rot6d",
        "repr_velocity",
    ):
        assert bundle.terms[name].raw.item() == 0.0
    assert bundle.terms["contact_all"].denominator.item() == 7 * 4
    bundle.total.backward()
    assert prediction.grad is not None
    assert logits.grad is not None


def test_control_entries_are_excluded_from_representation_and_audited() -> None:
    target = _valid_motion()
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[0, 0, 0:3] = True
    mask[1, 1, CONTACT_SLICE] = True
    bundle, _, _ = _bundle(mask)
    # Seven valid frames times three root channels, minus three hard entries.
    assert bundle.terms["repr_root_xyz"].denominator.item() == 7 * 3 - 3
    assert bundle.terms["control_continuous"].denominator.item() == 3
    assert bundle.terms["control_contact"].denominator.item() == 4


def test_capability_report_filters_numerator_and_denominator() -> None:
    bundle, _, _ = _bundle()
    first = bundle.terms_for_samples(torch.tensor([True, False]))
    second = bundle.terms_for_samples(torch.tensor([False, True]))
    assert first["contact_all"].denominator.item() == 3 * 4
    assert second["contact_all"].denominator.item() == 4 * 4
    assert torch.allclose(
        first["contact_all"].numerator + second["contact_all"].numerator,
        bundle.terms["contact_all"].numerator,
    )


def test_rank_local_loss_and_gradient_match_legacy_trainer() -> None:
    torch.manual_seed(11)
    target_physical = _valid_motion(batch=2, frames=5)
    target_physical[..., :269] += 0.01 * torch.randn_like(target_physical[..., :269])
    normalizer = _normalizer()
    target = normalizer.normalize(target_physical)
    observed = target.clone()
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[0, 1, :5] = True
    mask[1, 2, 269:273] = True
    valid = torch.tensor(
        [[True, True, True, True, False], [True, True, True, True, True]]
    )
    t = torch.tensor([0.31, 0.97])
    z = torch.randn(2, 5, 269)
    pred_old = torch.randn(2, 5, 269, requires_grad=True)
    logits_old = torch.randn(2, 5, 4, requires_grad=True)
    args = SimpleNamespace(
        representation_loss_mode="semantic_weighted",
        representation_loss_scale=0.09397019716051493,
        root_heading_loss_weight=1.0,
        velocity_loss_weight=1.0,
    )
    primary_pred, primary_target, _ = representation_loss_pair(
        z_cont_imp=z,
        t=t,
        x0_hat_cont=pred_old,
        x0_target_cont=target[..., :269],
        v_pred_cont=torch.zeros_like(pred_old),
        v_target_cont=torch.zeros_like(pred_old),
        prediction_type="x0",
        loss_space="velocity",
        velocity_t_eps=0.05,
    )
    repr_old, _, _ = representation_mse_loss(
        primary_pred,
        primary_target,
        valid[..., None] & ~mask[..., :269],
        args,
    )
    contact_old = bce_logits_masked(
        logits_old,
        target[..., 269:273],
        valid[..., None].expand_as(logits_old),
    )
    control_cont_old = smooth_l1_masked(
        pred_old, observed[..., :269], valid[..., None] & mask[..., :269]
    )
    control_contact_old = bce_logits_masked(
        logits_old,
        observed[..., 269:273],
        valid[..., None] & mask[..., 269:273],
    )
    x0_hat_old = torch.cat([pred_old, torch.sigmoid(logits_old)], dim=-1)
    physical_losses = compute_clean_semantic_losses(
        normalizer.denormalize(x0_hat_old.float()),
        target_physical,
        valid,
        fps=30.0,
        contact_threshold=0.5,
    )
    fk_old, _ = fk_position_consistency_loss(
        x0_hat_old,
        observed,
        mask,
        valid,
        normalizer,
        scale_m=0.05,
    )
    fk_weight, _ = effective_fk_consistency_weight(0.07, 5000, 123)
    total_old = (
        repr_old
        + 0.10 * contact_old
        + 0.25 * control_cont_old
        + 0.05 * control_contact_old
        + 0.01 * physical_losses["clean_root_vel"]
        + 0.01 * physical_losses["clean_joint_vel"]
        + 0.01 * physical_losses["foot_lock"]
        + fk_weight * fk_old
    )
    old_grad = torch.autograd.grad(total_old, (pred_old, logits_old))

    pred_new = pred_old.detach().clone().requires_grad_(True)
    logits_new = logits_old.detach().clone().requires_grad_(True)
    bundle = compute_hy273_multitask_loss(
        x0_hat_cont=pred_new,
        contact_logits=logits_new,
        z_cont_imputed=z,
        x0_target_norm=target,
        x0_target_physical=target_physical,
        hard_observed_norm=observed,
        hard_mask=mask,
        target_valid=valid,
        timesteps=t,
        normalizer=normalizer,
        global_step=123,
    )
    new_grad = torch.autograd.grad(bundle.total, (pred_new, logits_new))
    torch.testing.assert_close(bundle.total, total_old, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(new_grad[0], old_grad[0], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(new_grad[1], old_grad[1], rtol=1e-5, atol=1e-7)
