from __future__ import annotations

import torch
import pytest

from models.raw_motion.flow_schedule import build_unified_273_flow_state
from models.raw_motion.hy273_multitask_condition import CapabilityId, make_absent_condition
from models.raw_motion.hy273_multitask_losses import (
    HY273MultitaskLossWeights,
    R13_UNIFIED_CONTACT_WEIGHT,
    R13_UNIFIED_CONTROL_CONTACT_WEIGHT,
    compute_hy273_unified_flow_loss,
)
from models.raw_motion.hy273_normalizer import HY273Normalizer
from models.raw_motion.hy273_slices import CONTACT_SLICE, DIM_HY273
from sample_hy273_multitask import sample_hy273_multitask_ode


def _unified_normalizer() -> HY273Normalizer:
    mean = torch.zeros(DIM_HY273)
    std = torch.ones(DIM_HY273)
    mean[CONTACT_SLICE] = torch.tensor([0.60, 0.70, 0.65, 0.75])
    std[CONTACT_SLICE] = torch.tensor([0.49, 0.46, 0.48, 0.43])
    return HY273Normalizer(mean, std, normalize_contacts=True)


def _flow_only_weights() -> HY273MultitaskLossWeights:
    return HY273MultitaskLossWeights(
        representation_scale=0.0,
        contact=1.0,
        clean_root_velocity=0.0,
        clean_joint_velocity=0.0,
        foot_lock=0.0,
        fk_consistency=0.0,
        control_continuous=0.0,
        control_contact=0.0,
    )


def test_r13_contact_weight_matches_kimodo_semantic_ratio() -> None:
    representation_scale = HY273MultitaskLossWeights().representation_scale
    assert R13_UNIFIED_CONTACT_WEIGHT == representation_scale * 4.0 / 35.0
    control_scale = HY273MultitaskLossWeights().control_continuous
    assert R13_UNIFIED_CONTROL_CONTACT_WEIGHT == control_scale * 4.0 / 35.0


def test_unified_contact_normalization_round_trip_and_legacy_identity() -> None:
    motion = torch.randn(2, 5, DIM_HY273)
    motion[..., CONTACT_SLICE] = torch.tensor([0.0, 1.0, 1.0, 0.0])

    unified = _unified_normalizer()
    normalized = unified.normalize(motion)
    assert not torch.equal(normalized[..., CONTACT_SLICE], motion[..., CONTACT_SLICE])
    torch.testing.assert_close(unified.denormalize(normalized), motion)

    legacy = HY273Normalizer(unified.mean.reshape(-1), unified.std.reshape(-1))
    legacy_normalized = legacy.normalize(motion)
    torch.testing.assert_close(
        legacy_normalized[..., CONTACT_SLICE], motion[..., CONTACT_SLICE]
    )


def test_unified_flow_interpolates_and_overwrites_all_273_channels() -> None:
    x0 = torch.arange(2 * 3 * DIM_HY273, dtype=torch.float32).reshape(
        2, 3, DIM_HY273
    ) / 100.0
    noise = -torch.ones_like(x0)
    observed = torch.full_like(x0, 7.0)
    mask = torch.zeros_like(x0, dtype=torch.bool)
    mask[0, 1, 0] = True
    mask[0, 1, CONTACT_SLICE] = True
    mask[1, 2, 272] = True
    timesteps = torch.tensor([0.25, 0.75])

    state = build_unified_273_flow_state(
        x0, observed, mask, timesteps, noise=noise
    )
    expected = timesteps[:, None, None] * x0 + (
        1.0 - timesteps[:, None, None]
    ) * noise
    expected = torch.where(mask, observed, expected)

    torch.testing.assert_close(state["z_imp"], expected)
    torch.testing.assert_close(state["model_in"][..., :DIM_HY273], expected)
    torch.testing.assert_close(
        state["model_in"][..., DIM_HY273:], mask.to(torch.float32)
    )
    torch.testing.assert_close(state["v_target"], x0 - noise)


def test_unified_contact_flow_has_zero_exact_loss_and_direct_gradients() -> None:
    normalizer = _unified_normalizer()
    target_physical = torch.zeros(1, 4, DIM_HY273)
    target_physical[..., 3] = 1.0
    target_physical[..., CONTACT_SLICE] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    target_norm = normalizer.normalize(target_physical)
    z_imputed = torch.randn_like(target_norm)
    valid = torch.ones(1, 4, dtype=torch.bool)
    hard_mask = torch.zeros_like(target_norm, dtype=torch.bool)
    timesteps = torch.tensor([0.2])
    weights = _flow_only_weights()

    exact = compute_hy273_unified_flow_loss(
        x0_hat_norm=target_norm,
        z_imputed=z_imputed,
        x0_target_norm=target_norm,
        x0_target_physical=target_physical,
        hard_observed_norm=target_norm,
        hard_mask=hard_mask,
        target_valid=valid,
        timesteps=timesteps,
        normalizer=normalizer,
        global_step=0,
        weights=weights,
    )
    assert exact.terms["contact_all"].raw.item() == 0.0
    assert exact.total.item() == 0.0

    perturbation = torch.zeros_like(target_norm)
    perturbation[..., CONTACT_SLICE] = 0.25
    prediction = (target_norm + perturbation).detach().requires_grad_(True)
    perturbed = compute_hy273_unified_flow_loss(
        x0_hat_norm=prediction,
        z_imputed=z_imputed,
        x0_target_norm=target_norm,
        x0_target_physical=target_physical,
        hard_observed_norm=target_norm,
        hard_mask=hard_mask,
        target_valid=valid,
        timesteps=timesteps,
        normalizer=normalizer,
        global_step=0,
        weights=weights,
    )
    perturbed.total.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad[..., CONTACT_SLICE]) == 4 * 4
    assert torch.count_nonzero(prediction.grad[..., : CONTACT_SLICE.start]) == 0


def test_clean_x0_smooth_l1_is_not_reweighted_by_timestep() -> None:
    normalizer = _unified_normalizer()
    target_physical = torch.zeros(2, 3, DIM_HY273)
    target_physical[..., 3] = 1.0
    target_physical[..., CONTACT_SLICE] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    target_norm = normalizer.normalize(target_physical)
    prediction = target_norm + 0.25
    valid = torch.ones(2, 3, dtype=torch.bool)
    hard_mask = torch.zeros_like(target_norm, dtype=torch.bool)
    weights = HY273MultitaskLossWeights(
        representation_scale=1.0,
        contact=1.0,
        clean_root_velocity=0.0,
        clean_joint_velocity=0.0,
        foot_lock=0.0,
        fk_consistency=0.0,
        control_continuous=0.0,
        control_contact=0.0,
    )

    def compute(timesteps: torch.Tensor, loss_space: str):
        return compute_hy273_unified_flow_loss(
            x0_hat_norm=prediction,
            z_imputed=torch.zeros_like(prediction),
            x0_target_norm=target_norm,
            x0_target_physical=target_physical,
            hard_observed_norm=target_norm,
            hard_mask=hard_mask,
            target_valid=valid,
            timesteps=timesteps,
            normalizer=normalizer,
            global_step=0,
            weights=weights,
            representation_loss_space=loss_space,
        )

    clean_low = compute(torch.tensor([0.1, 0.1]), "clean_x0_smooth_l1")
    clean_high = compute(torch.tensor([0.9, 0.9]), "clean_x0_smooth_l1")
    torch.testing.assert_close(clean_low.total, clean_high.total)
    clean_mse_low = compute(torch.tensor([0.1, 0.1]), "clean_x0_mse")
    clean_mse_high = compute(torch.tensor([0.9, 0.9]), "clean_x0_mse")
    torch.testing.assert_close(clean_mse_low.total, clean_mse_high.total)
    velocity_low = compute(torch.tensor([0.1, 0.1]), "velocity_mse")
    velocity_high = compute(torch.tensor([0.9, 0.9]), "velocity_mse")
    assert velocity_high.total > velocity_low.total * 50.0


def test_continuous_loss_space_and_multiplier_do_not_change_contact_objective() -> None:
    normalizer = _unified_normalizer()
    target_physical = torch.zeros(1, 3, DIM_HY273)
    target_physical[..., 3] = 1.0
    target_physical[..., CONTACT_SLICE] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    target_norm = normalizer.normalize(target_physical)
    prediction = target_norm + 0.25
    kwargs = dict(
        x0_hat_norm=prediction,
        z_imputed=torch.zeros_like(prediction),
        x0_target_norm=target_norm,
        x0_target_physical=target_physical,
        hard_observed_norm=target_norm,
        hard_mask=torch.zeros_like(target_norm, dtype=torch.bool),
        target_valid=torch.ones(1, 3, dtype=torch.bool),
        timesteps=torch.tensor([0.8]),
        normalizer=normalizer,
        global_step=0,
        weights=HY273MultitaskLossWeights(
            representation_scale=1.0,
            contact=1.0,
            clean_root_velocity=0.0,
            clean_joint_velocity=0.0,
            foot_lock=0.0,
            fk_consistency=0.0,
            control_continuous=0.0,
            control_contact=0.0,
        ),
        representation_loss_space="clean_x0_mse",
        contact_loss_space="velocity_mse",
    )
    unit = compute_hy273_unified_flow_loss(**kwargs)
    triple = compute_hy273_unified_flow_loss(
        **kwargs, representation_multiplier=3.0
    )
    torch.testing.assert_close(
        triple.terms["contact_all"].weighted,
        unit.terms["contact_all"].weighted,
    )
    for name in (
        "repr_root_xyz",
        "repr_heading",
        "repr_joint_position",
        "repr_global_rot6d",
        "repr_velocity",
    ):
        torch.testing.assert_close(
            triple.terms[name].weighted,
            3.0 * unit.terms[name].weighted,
        )


class _UnifiedStateSpy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs: list[torch.Tensor] = []
        clean = torch.zeros(DIM_HY273)
        clean[CONTACT_SLICE] = torch.tensor([-2.0, 2.0, -4.0, 4.0])
        self.register_buffer("clean", clean)

    def forward(self, model_in: torch.Tensor, **_kwargs) -> torch.Tensor:
        self.inputs.append(model_in.detach().clone())
        return self.clean.to(model_in).view(1, 1, -1).expand(
            model_in.shape[0], model_in.shape[1], -1
        ) + self.anchor * 0.0


def test_unified_sampler_has_no_sigmoid_feedback_and_thresholds_only_on_decode() -> None:
    model = _UnifiedStateSpy()
    condition = make_absent_condition(
        batch_size=1,
        target_frames=3,
        target_lengths=torch.tensor([3]),
    )
    initial = torch.zeros(1, 3, DIM_HY273)
    output = sample_hy273_multitask_ode(
        model,
        HY273Normalizer(
            torch.zeros(DIM_HY273),
            torch.ones(DIM_HY273),
            normalize_contacts=True,
        ),
        condition,
        ["walk"],
        torch.zeros_like(initial),
        torch.zeros_like(initial, dtype=torch.bool),
        num_steps=2,
        text_cfg_scale=2.0,
        initial_unified_noise=initial,
    )

    assert len(model.inputs) == 2
    second_step_state = model.inputs[1][..., :DIM_HY273]
    expected_second_contacts = 0.5 * model.clean[CONTACT_SLICE]
    torch.testing.assert_close(
        second_step_state[..., CONTACT_SLICE],
        expected_second_contacts.view(1, 1, 4).expand_as(
            second_step_state[..., CONTACT_SLICE]
        ),
    )
    expected_binary = torch.tensor([0.0, 1.0, 0.0, 1.0])
    torch.testing.assert_close(
        output.raw_motion[..., CONTACT_SLICE],
        expected_binary.view(1, 1, 4).expand_as(output.raw_motion[..., CONTACT_SLICE]),
    )
    assert output.protocol["contact_init"] == "unified_273d_state"
    assert output.protocol["contact_feedback"] == "ode_273d"
    assert output.protocol["initial_noise_source"] == "provided_unified_273d"


def test_unified_sampler_rejects_legacy_split_noise() -> None:
    model = _UnifiedStateSpy()
    condition = make_absent_condition(
        batch_size=1,
        target_frames=3,
        target_lengths=torch.tensor([3]),
    )
    zeros = torch.zeros(1, 3, DIM_HY273)
    with pytest.raises(ValueError, match="legacy split noise tensors are forbidden"):
        sample_hy273_multitask_ode(
            model,
            _unified_normalizer(),
            condition,
            ["walk"],
            zeros,
            torch.zeros_like(zeros, dtype=torch.bool),
            num_steps=1,
            initial_continuous_noise=zeros[..., :269],
            initial_contact_noise=zeros[..., 269:273],
        )


def test_unified_control_overwrites_only_controlled_branches_and_clamps_physically() -> None:
    model = _UnifiedStateSpy()
    condition = make_absent_condition(
        batch_size=1,
        target_frames=3,
        target_lengths=torch.tensor([3]),
        capability=CapabilityId.KIMODO_CONTROL,
    )
    initial = -torch.ones(1, 3, DIM_HY273)
    observed = torch.zeros_like(initial)
    observed[0, 1, 0] = 42.0
    observed[0, 1, CONTACT_SLICE.start] = 1.0
    mask = torch.zeros_like(initial, dtype=torch.bool)
    mask[0, 1, 0] = True
    mask[0, 1, CONTACT_SLICE.start] = True
    normalizer = _unified_normalizer()

    output = sample_hy273_multitask_ode(
        model,
        normalizer,
        condition,
        ["walk"],
        observed,
        mask,
        num_steps=2,
        text_cfg_scale=2.0,
        control_cfg_scale=2.0,
        initial_unified_noise=initial,
    )

    assert output.branch_names == ("joint", "text", "control", "empty")
    second = model.inputs[1]
    state, seen_mask = second[..., :DIM_HY273], second[..., DIM_HY273:].bool()
    observed_norm = normalizer.normalize(observed)
    for branch_index in (0, 2):
        assert seen_mask[branch_index, 1, 0]
        assert seen_mask[branch_index, 1, CONTACT_SLICE.start]
        assert state[branch_index, 1, 0] == observed_norm[0, 1, 0]
        assert (
            state[branch_index, 1, CONTACT_SLICE.start]
            == observed_norm[0, 1, CONTACT_SLICE.start]
        )
    for branch_index in (1, 3):
        assert not seen_mask[branch_index].any()
        assert state[branch_index, 1, 0] != observed_norm[0, 1, 0]
        assert (
            state[branch_index, 1, CONTACT_SLICE.start]
            != observed_norm[0, 1, CONTACT_SLICE.start]
        )

    assert output.raw_motion[0, 1, 0] != observed[0, 1, 0]
    assert output.raw_motion[0, 1, CONTACT_SLICE.start] == 0.0
    assert torch.equal(output.exact_clamped_motion[mask], observed[mask])
