import torch

from models.raw_motion.hy273_normalizer import HY273Normalizer
from models.raw_motion.raw_flow_dit import HY273RawFlow
from argparse import Namespace

import pytest

from sample_hy273_raw import (
    ODESampleOutput,
    apply_checkpoint_path_override,
    resolve_sampling_cfg_scales,
    sample_ode,
)


def test_sample_ode_separates_raw_and_exact_clamped_outputs():
    model = HY273RawFlow(
        hidden_dim=64,
        num_heads=4,
        depth_double=1,
        depth_single=1,
        text_encoder="none",
        self_conditioning=True,
    )
    mean = torch.zeros(273)
    std = torch.ones(273)
    normalizer = HY273Normalizer(mean, std)
    obs = torch.zeros(2, 6, 273)
    mask = torch.zeros_like(obs, dtype=torch.bool)
    obs[:, 0, :5] = 3.0
    mask[:, 0, :5] = True
    lengths = torch.tensor([6, 6])
    out = sample_ode(
        model,
        normalizer,
        lengths,
        ["walk", "run"],
        obs,
        mask,
        c_dir=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        num_steps=2,
        self_conditioning=True,
        prediction_type="x0",
        return_details=True,
    )
    assert isinstance(out, ODESampleOutput)
    assert out.raw_motion.shape == (2, 6, 273)
    assert torch.allclose(out.exact_clamped_motion[:, 0, :5].cpu(), obs[:, 0, :5])
    assert not torch.allclose(out.raw_motion[:, 0, :5].cpu(), obs[:, 0, :5])


class _BranchSpy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs = []

    def forward(self, model_in, **_kwargs):
        self.inputs.append(model_in.detach().clone())
        return model_in.new_zeros((*model_in.shape[:-1], 273)) + self.anchor * 0


class _BranchValueSpy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.self_conditions = []

    def forward(self, model_in, x_self_cond=None, **_kwargs):
        self.self_conditions.append(
            None if x_self_cond is None else x_self_cond.detach().clone()
        )
        if model_in.shape[0] % 4:
            raise AssertionError("Expected four separated-CFG branches")
        batch = model_in.shape[0] // 4
        branch_values = model_in.new_tensor([10.0, 2.0, 4.0, 1.0]).repeat_interleave(batch)
        return branch_values[:, None, None].expand(-1, model_in.shape[1], 273) + self.anchor * 0


def test_separated_cfg_uses_branch_local_imputation_without_state_contamination():
    model = _BranchSpy()
    normalizer = HY273Normalizer(torch.zeros(273), torch.ones(273))
    obs = torch.zeros(1, 4, 273)
    mask = torch.zeros_like(obs, dtype=torch.bool)
    obs[:, 1, 0] = 99.0
    mask[:, 1, 0] = True
    sample_ode(
        model,
        normalizer,
        torch.tensor([4]),
        ["walk"],
        obs,
        mask,
        c_dir=torch.tensor([[1.0, 0.0]]),
        num_steps=2,
        cfg_scale=3.5,
        control_cfg_scale=2.0,
        prediction_type="x0",
    )
    assert len(model.inputs) == 2
    for branch_input in model.inputs:
        joint, text, control, empty = branch_input.chunk(4, dim=0)
        assert float(joint[0, 1, 0]) == 99.0
        assert float(control[0, 1, 0]) == 99.0
        assert float(text[0, 1, 0]) != 99.0
        assert float(empty[0, 1, 0]) != 99.0
        assert not text[..., 273:].bool().any()
        assert not empty[..., 273:].bool().any()


def test_sample_ode_uses_numerical_velocity_epsilon():
    model = HY273RawFlow(
        hidden_dim=64,
        num_heads=4,
        depth_double=1,
        depth_single=1,
        text_encoder="none",
    )
    normalizer = HY273Normalizer(torch.zeros(273), torch.ones(273))
    out = sample_ode(
        model,
        normalizer,
        torch.tensor([4]),
        ["walk"],
        torch.zeros(1, 4, 273),
        torch.zeros(1, 4, 273, dtype=torch.bool),
        c_dir=torch.tensor([[1.0, 0.0]]),
        num_steps=2,
        prediction_type="x0",
        velocity_t_eps=1e-4,
    )
    assert out.shape == (1, 4, 273)
    assert torch.isfinite(out).all()


def test_sample_ode_accepts_fixed_per_case_initial_state():
    model = HY273RawFlow(
        hidden_dim=64,
        num_heads=4,
        depth_double=1,
        depth_single=1,
        text_encoder="none",
    )
    model.eval()
    normalizer = HY273Normalizer(torch.zeros(273), torch.ones(273))
    common = dict(
        model=model,
        normalizer=normalizer,
        lengths=torch.tensor([4]),
        texts=["walk"],
        observed_un=torch.zeros(1, 4, 273),
        motion_mask=torch.zeros(1, 4, 273, dtype=torch.bool),
        c_dir=torch.tensor([[1.0, 0.0]]),
        num_steps=2,
        prediction_type="x0",
        initial_continuous_noise=torch.randn(1, 4, 269),
        initial_contact_noise=torch.rand(1, 4, 4),
    )
    torch.manual_seed(1)
    first = sample_ode(**common)
    torch.manual_seed(999)
    second = sample_ode(**common)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    with pytest.raises(ValueError, match="initial_continuous_noise"):
        sample_ode(
            **{**common, "initial_continuous_noise": torch.zeros(1, 3, 269)}
        )


def test_separated_cfg_oracle_algebra_and_contact_policy():
    model = _BranchValueSpy()
    normalizer = HY273Normalizer(torch.zeros(273), torch.ones(273))
    observed = torch.zeros(1, 3, 273)
    mask = torch.zeros_like(observed, dtype=torch.bool)
    mask[:, 0, 0] = True
    details = sample_ode(
        model,
        normalizer,
        torch.tensor([3]),
        ["walk"],
        observed,
        mask,
        c_dir=torch.tensor([[1.0, 0.0]]),
        num_steps=1,
        cfg_scale=3.5,
        control_cfg_scale=2.0,
        prediction_type="x0",
        return_details=True,
    )
    assert isinstance(details, ODESampleOutput)
    expected = 1.0 + 3.5 * (2.0 - 1.0) + 2.0 * (4.0 - 1.0)
    assert torch.allclose(details.final_clean_prediction[..., 1:269], torch.full_like(details.final_clean_prediction[..., 1:269], expected))
    assert torch.allclose(
        details.final_clean_prediction[..., 269:273],
        torch.full_like(details.final_clean_prediction[..., 269:273], torch.sigmoid(torch.tensor(10.0))),
    )


def test_separated_cfg_self_conditioning_stays_branch_local():
    model = _BranchValueSpy()
    normalizer = HY273Normalizer(torch.zeros(273), torch.ones(273))
    observed = torch.zeros(1, 3, 273)
    mask = torch.zeros_like(observed, dtype=torch.bool)
    mask[:, 0, 0] = True
    sample_ode(
        model,
        normalizer,
        torch.tensor([3]),
        ["walk"],
        observed,
        mask,
        c_dir=torch.tensor([[1.0, 0.0]]),
        num_steps=2,
        self_conditioning=True,
        cfg_scale=3.5,
        control_cfg_scale=2.0,
        prediction_type="x0",
    )
    assert model.self_conditions[0] is None
    joint, text, control, empty = model.self_conditions[1].chunk(4, dim=0)
    for value, expected in ((joint, 10.0), (text, 2.0), (control, 4.0), (empty, 1.0)):
        assert torch.allclose(value[..., 1:269], torch.full_like(value[..., 1:269], expected))


def test_kimodo_like_rejects_unpinned_sampling_path_override(tmp_path):
    args = Namespace(
        architecture="redenoise_kimodo_like",
        data_root=str(tmp_path / "pinned"),
    )
    with pytest.raises(RuntimeError, match="Cannot override pinned data_root"):
        apply_checkpoint_path_override(args, "data_root", str(tmp_path / "other"))
    apply_checkpoint_path_override(args, "data_root", str(tmp_path / "pinned"))


def test_kimodo_like_sampling_defaults_to_validated_cfg_scales():
    assert resolve_sampling_cfg_scales(
        Namespace(architecture="redenoise_kimodo_like"), None, None
    ) == (2.0, 2.0)
    assert resolve_sampling_cfg_scales(
        Namespace(architecture="one_stage"), None, None
    ) == (1.0, 1.0)
