from __future__ import annotations

import torch

from models.raw_motion.flow_schedule import clean_x0_euler_step
from models.raw_motion.hy273_multitask_condition import CapabilityId
from models.raw_motion.hy273_normalizer import HY273Normalizer
from sample_hy273_multitask import (
    make_edit_condition,
    make_instruction_only_edit_diagnostic_condition,
    sample_hy273_multitask_ode,
)


class BranchOracle(torch.nn.Module):
    """Emit source + text + control constants so CFG algebra is observable."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.mask_seen: list[torch.Tensor] = []

    def forward(
        self,
        model_in,
        *,
        text,
        condition,
        **kwargs,
    ):
        del kwargs
        mask = model_in[..., 273:].bool()
        self.mask_seen.append(mask.detach().cpu())
        source = condition.source_present.any(dim=1).float()
        text_value = torch.tensor(
            [2.0 if value else 0.0 for value in text], device=model_in.device
        )
        control = mask.reshape(mask.shape[0], -1).any(dim=1).float() * 4.0
        value = source + text_value + control + self.anchor * 0.0
        return value[:, None, None].expand(-1, model_in.shape[1], 273).clone()


def _normalizer() -> HY273Normalizer:
    return HY273Normalizer(torch.zeros(273), torch.ones(273))


def _unified_normalizer() -> HY273Normalizer:
    return HY273Normalizer(
        torch.zeros(273), torch.ones(273), normalize_contacts=True
    )


def _source(frames: int = 4) -> torch.Tensor:
    value = torch.zeros(1, frames, 273)
    value[..., 3] = 1.0
    return value


def test_edit_condition_respects_padded_source_lengths() -> None:
    source = torch.zeros(2, 5, 273)
    source[..., 3] = 1.0
    condition = make_edit_condition(
        source,
        source_lengths=torch.tensor([3, 5]),
        target_lengths=torch.tensor([4, 4]),
    )
    assert condition.source_native_lengths.tolist() == [[3], [5]]
    assert condition.source_time_valid[0, 0].tolist() == [True, True, True, False, False]
    assert not bool(condition.source_value_mask[0, 0, 3:].any())
    assert bool(condition.source_value_mask[1, 0].all())


def test_edit_layered_cfg_separates_source_and_instruction_guidance() -> None:
    model = BranchOracle()
    condition = make_edit_condition(
        _source(),
        target_lengths=torch.tensor([4]),
        capability=CapabilityId.MOTION_EDIT,
    )
    output = sample_hy273_multitask_ode(
        model,
        _normalizer(),
        condition,
        ["raise the left hand"],
        torch.zeros(1, 4, 273),
        torch.zeros(1, 4, 273, dtype=torch.bool),
        num_steps=1,
        source_cfg_scale=2.0,
        edit_cfg_scale=2.0,
        contact_init="zeros",
    )
    # empty=0, source=1, source+text=3: 0 + 2*(1-0) + 2*(3-1) = 6.
    torch.testing.assert_close(output.raw_motion[..., :269], torch.full((1, 4, 269), 6.0))
    torch.testing.assert_close(
        output.raw_motion[..., 269:273],
        torch.full((1, 4, 4), torch.sigmoid(torch.tensor(3.0)).item()),
    )
    assert output.branch_names == ("empty", "source", "joint")
    assert output.protocol["source_cfg_scale"] == 2.0
    assert output.protocol["route"] == "edit"
    assert output.protocol["ode_state_persistent_clamp"] is False


def test_edit_defaults_to_joint_only_unit_edit_scale() -> None:
    condition = make_edit_condition(
        _source(),
        target_lengths=torch.tensor([4]),
        capability=CapabilityId.MOTION_EDIT,
    )
    inputs = (
        BranchOracle(),
        _normalizer(),
        condition,
        ["raise the left hand"],
        torch.zeros(1, 4, 273),
        torch.zeros(1, 4, 273, dtype=torch.bool),
    )
    output = sample_hy273_multitask_ode(
        *inputs,
        num_steps=1,
        contact_init="zeros",
    )
    # Unit source/edit scales reduce exactly to the trained joint branch (=3).
    torch.testing.assert_close(
        output.raw_motion[..., :269], torch.full((1, 4, 269), 3.0)
    )
    assert output.protocol["source_cfg_scale"] == 1.0
    assert output.protocol["edit_cfg_scale"] == 1.0



def test_edit_exact_source_baseline_uses_known_clean_source_x0() -> None:
    model = BranchOracle()
    condition = make_edit_condition(
        _source(),
        target_lengths=torch.tensor([4]),
        capability=CapabilityId.MOTION_EDIT,
    )
    source_anchor = torch.full((1, 4, 273), 4.0)
    source_anchor[..., 269:273] = 0.0
    output = sample_hy273_multitask_ode(
        model,
        _normalizer(),
        condition,
        ["raise the left hand"],
        torch.zeros(1, 4, 273),
        torch.zeros(1, 4, 273, dtype=torch.bool),
        num_steps=1,
        source_cfg_scale=1.0,
        edit_cfg_scale=2.0,
        contact_init="zeros",
        edit_source_baseline="exact",
        edit_source_anchor_physical=source_anchor,
    )
    # exact source=4, source+text model branch=3: 4 + 2*(3-4) = 2.
    torch.testing.assert_close(output.raw_motion[..., :269], torch.full((1, 4, 269), 2.0))
    torch.testing.assert_close(
        output.raw_motion[..., 269:273],
        torch.full((1, 4, 4), torch.sigmoid(torch.tensor(3.0)).item()),
    )
    assert output.protocol["edit_source_baseline"] == "exact"
    torch.testing.assert_close(
        output.final_branch_predictions["source"][..., :269],
        source_anchor[..., :269],
    )


def test_edit_exact_source_baseline_treats_empty_instruction_as_preserve() -> None:
    source_anchor = torch.full((1, 4, 273), 4.0)
    source_anchor[..., 269:273] = 1.0
    output = sample_hy273_multitask_ode(
        BranchOracle(),
        _normalizer(),
        make_edit_condition(_source(), target_lengths=torch.tensor([4])),
        [""],
        torch.zeros(1, 4, 273),
        torch.zeros(1, 4, 273, dtype=torch.bool),
        num_steps=1,
        source_cfg_scale=1.0,
        edit_cfg_scale=6.0,
        contact_init="zeros",
        edit_source_baseline="exact",
        edit_source_anchor_physical=source_anchor,
    )
    torch.testing.assert_close(output.raw_motion[..., :269], source_anchor[..., :269])
    torch.testing.assert_close(
        output.raw_motion[..., 269:273],
        torch.full((1, 4, 4), 1.0 - 1e-4),
        atol=1e-6,
        rtol=0.0,
    )


def test_edit_sampler_rejects_capability_mask_mismatch() -> None:
    condition = make_edit_condition(
        _source(),
        target_lengths=torch.tensor([4]),
        capability=CapabilityId.MOTION_EDIT_CONTROL,
    )
    try:
        sample_hy273_multitask_ode(
            BranchOracle(),
            _normalizer(),
            condition,
            ["edit"],
            torch.zeros(1, 4, 273),
            torch.zeros(1, 4, 273, dtype=torch.bool),
            num_steps=1,
        )
    except ValueError as exc:
        assert "Capability/control-mask mismatch" in str(exc)
    else:
        raise AssertionError("Expected fail-closed capability/mask validation")


def test_relative_instruction_only_route_is_explicitly_diagnostic() -> None:
    condition = make_instruction_only_edit_diagnostic_condition(
        target_lengths=torch.tensor([4])
    )
    inputs = (
        BranchOracle(),
        _normalizer(),
        condition,
        ["raise the left hand"],
        torch.zeros(1, 4, 273),
        torch.zeros(1, 4, 273, dtype=torch.bool),
    )
    try:
        sample_hy273_multitask_ode(*inputs, num_steps=1)
    except ValueError as exc:
        assert "EDIT samples require a source motion" in str(exc)
    else:
        raise AssertionError("Source-absent EDIT must fail without the diagnostic flag")

    output = sample_hy273_multitask_ode(
        *inputs,
        num_steps=1,
        edit_cfg_scale=1.0,
        contact_init="zeros",
        diagnostic_allow_source_absent_edit=True,
    )
    torch.testing.assert_close(
        output.raw_motion[..., :269], torch.full((1, 4, 269), 2.0)
    )
    assert output.protocol["diagnostic_allow_source_absent_edit"] is True
    assert output.branch_names == ("empty", "source", "joint")


def test_multitask_sampler_validates_fixed_initial_state_shapes() -> None:
    condition = make_edit_condition(_source(), target_lengths=torch.tensor([4]))
    try:
        sample_hy273_multitask_ode(
            BranchOracle(),
            _normalizer(),
            condition,
            ["edit"],
            torch.zeros(1, 4, 273),
            torch.zeros(1, 4, 273, dtype=torch.bool),
            num_steps=1,
            initial_continuous_noise=torch.zeros(1, 3, 269),
        )
    except ValueError as exc:
        assert "initial_continuous_noise" in str(exc)
    else:
        raise AssertionError("Invalid fixed initial state must fail closed")


def test_unified_sampler_uses_shared_clean_x0_euler_step() -> None:
    model = BranchOracle()
    condition = make_edit_condition(_source(), target_lengths=torch.tensor([4]))
    initial = torch.linspace(-1.0, 1.0, 4 * 273).reshape(1, 4, 273)
    output = sample_hy273_multitask_ode(
        model,
        _unified_normalizer(),
        condition,
        ["edit"],
        torch.zeros_like(initial),
        torch.zeros_like(initial, dtype=torch.bool),
        num_steps=1,
        initial_unified_noise=initial,
    )
    clean_x0 = torch.full_like(initial, 3.0)
    expected, _ = clean_x0_euler_step(
        initial,
        clean_x0,
        timestep=torch.tensor([0.0]),
        dt=torch.tensor(1.0),
    )
    torch.testing.assert_close(output.raw_motion[..., :269], expected[..., :269])
    torch.testing.assert_close(
        output.raw_motion[..., 269:273], torch.ones_like(expected[..., 269:273])
    )
