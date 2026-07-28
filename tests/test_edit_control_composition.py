from __future__ import annotations

import torch

from models.raw_motion.hy273_multitask_condition import CapabilityId
from models.raw_motion.hy273_normalizer import HY273Normalizer
from sample_hy273_multitask import make_edit_condition, sample_hy273_multitask_ode
from tests.test_edit_sampling import BranchOracle, _source


def test_edit_control_cfg_is_hierarchical_and_overwrite_is_input_only() -> None:
    model = BranchOracle()
    condition = make_edit_condition(
        _source(),
        target_lengths=torch.tensor([4]),
        capability=CapabilityId.MOTION_EDIT_CONTROL,
    )
    observed = torch.zeros(1, 4, 273)
    observed[..., 0] = 42.0
    observed[..., 269] = 1.0
    mask = torch.zeros_like(observed, dtype=torch.bool)
    mask[:, 2, 0] = True
    mask[:, 2, 269] = True
    output = sample_hy273_multitask_ode(
        model,
        HY273Normalizer(torch.zeros(273), torch.ones(273)),
        condition,
        ["move faster"],
        observed,
        mask,
        num_steps=1,
        edit_cfg_scale=2.0,
        control_cfg_scale=3.0,
        contact_init="zeros",
    )

    # source=1, edit=3, all=7: 1 + 2*(3-1) + 3*(7-3) = 17.
    torch.testing.assert_close(
        output.raw_motion[..., :269], torch.full((1, 4, 269), 17.0)
    )
    # Contact defaults to the most complete all-branch logits, not CFG=17 logits.
    expected_contact = torch.sigmoid(torch.tensor(7.0)).item()
    torch.testing.assert_close(
        output.raw_motion[..., 269:273],
        torch.full((1, 4, 4), expected_contact),
    )
    assert output.raw_motion[0, 2, 0].item() == 17.0
    assert output.exact_clamped_motion[0, 2, 0].item() == 42.0
    assert output.raw_motion[0, 2, 269].item() != 1.0
    assert output.exact_clamped_motion[0, 2, 269].item() == 1.0
    assert output.branch_names == ("empty", "source", "edit", "all")

    seen = model.mask_seen[-1]
    assert not seen[0].any()  # empty branch
    assert not seen[1].any()  # source branch
    assert not seen[2].any()  # edit branch
    assert seen[3].any()      # all branch only
