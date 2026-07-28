import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from models.raw_motion.flow_schedule import build_flow_state, clean_from_velocity
from models.raw_motion.hy273_normalizer import HY273Normalizer, apply_yaw_rotation, transform_equivariance_error
from models.raw_motion.hy273_root_conditioning import KimodoRootConditioner
from models.raw_motion.kimodo_like_flow_dit import HY273RedenoiseKimodoLike
from models.raw_motion.hy273_slices import load_smplx22_neutral_joints, matrix_to_cont6d
from models.raw_motion.hytext_cache import hytext_key
from models.raw_motion.raw_flow_dit import HY273RawFlow
from train_hy273_raw_flow import (
    build_arg_parser,
    compute_clean_semantic_losses,
    effective_fk_consistency_weight,
    fk_position_consistency_loss,
    hash_model_state,
    load_yaml,
    make_train_state,
    merge_config,
    predict_clean_cont,
    prediction_velocity_cont,
    representation_loss_pair,
    representation_mse_loss,
    resolve_resume_cursor,
    save_checkpoint,
    sha256_file,
    trace_stream_seed,
    validate_normalizer_contract,
    validate_resume_contract,
)


def _model(self_conditioning=False):
    return HY273RawFlow(
        hidden_dim=64,
        num_heads=4,
        depth_double=1,
        depth_single=1,
        mlp_ratio=2.0,
        text_encoder="none",
        self_conditioning=self_conditioning,
    )


def _write_fake_hytext_cache(tmp_path):
    cache_dir = tmp_path / "hytext_cache"
    shard_dir = cache_dir / "shards" / "shard_00000"
    shard_dir.mkdir(parents=True)
    texts = ["", "walk", "run"]
    ctxt = np.random.randn(len(texts), 5, 16).astype(np.float16)
    vtxt = np.random.randn(len(texts), 1, 8).astype(np.float16)
    ctxt_len = np.array([1, 4, 3], dtype=np.int16)
    np.save(shard_dir / "ctxt.npy", ctxt)
    np.save(shard_dir / "vtxt.npy", vtxt)
    np.save(shard_dir / "ctxt_len.npy", ctxt_len)
    index = {
        hytext_key(text): {"shard": "shard_00000", "row": row, "text": text}
        for row, text in enumerate(texts)
    }
    (cache_dir / "index.json").write_text(json.dumps(index))
    (cache_dir / "manifest.json").write_text(json.dumps({"format": "hytext_memmap_v1"}))
    return cache_dir


def _write_root_stats(tmp_path):
    full = tmp_path / "stats" / "full"
    local = tmp_path / "stats" / "local_root"
    full.mkdir(parents=True)
    local.mkdir(parents=True)
    np.save(full / "Mean.npy", np.zeros(273, dtype=np.float32))
    np.save(full / "Std.npy", np.ones(273, dtype=np.float32))
    np.save(local / "Mean.npy", np.zeros(4, dtype=np.float32))
    np.save(local / "Std.npy", np.ones(4, dtype=np.float32))
    return full, local


def _kimodo_like_model(tmp_path):
    full, local = _write_root_stats(tmp_path)
    return HY273RedenoiseKimodoLike(
        hidden_dim=64,
        num_heads=4,
        root_depth_double=1,
        root_depth_single=1,
        body_depth_double=1,
        body_depth_single=1,
        text_encoder="none",
        max_text_tokens=4,
        motion_stats_dir=str(full),
        local_root_stats_dir=str(local),
        stats_variance_eps=0.0,
    )


def test_forward_without_self_conditioning_shape_finite():
    model = _model(False)
    x = torch.randn(2, 8, 546)
    out = model(x, t=torch.rand(2), text=["walk", "run"], length_mask=torch.ones(2, 8, dtype=torch.bool))
    assert out.shape == (2, 8, 273)
    assert torch.isfinite(out).all()


def test_forward_with_self_conditioning_shape_finite():
    model = _model(True)
    x = torch.randn(2, 8, 546)
    sc = torch.randn(2, 8, 273)
    out = model(
        x,
        t=torch.rand(2),
        text=["walk", "run"],
        length_mask=torch.ones(2, 8, dtype=torch.bool),
        x_self_cond=sc,
    )
    assert out.shape == (2, 8, 273)
    assert torch.isfinite(out).all()


def test_forward_with_cached_hytext_shape_finite(tmp_path):
    cache_dir = _write_fake_hytext_cache(tmp_path)
    model = HY273RawFlow(
        hidden_dim=64,
        num_heads=4,
        depth_double=1,
        depth_single=1,
        mlp_ratio=2.0,
        text_encoder="hy_cache",
        max_text_tokens=5,
        hytext_cache_dir=str(cache_dir),
        hytext_ctxt_dim=16,
        hytext_vtxt_dim=8,
    )
    x = torch.randn(2, 8, 546)
    out = model(x, t=torch.rand(2), text=["walk", "run"], length_mask=torch.ones(2, 8, dtype=torch.bool))
    out_drop = model(
        x,
        t=torch.rand(2),
        text=["walk", "run"],
        length_mask=torch.ones(2, 8, dtype=torch.bool),
        force_drop_text=True,
    )
    assert out.shape == (2, 8, 273)
    assert out_drop.shape == (2, 8, 273)
    assert torch.isfinite(out).all()
    assert torch.isfinite(out_drop).all()


def test_kimodo_like_forward_shape_and_detached_root_bridge(tmp_path):
    model = _kimodo_like_model(tmp_path)
    model.train()
    x = torch.randn(2, 8, 546)
    x[..., 273:] = 0
    valid = torch.ones(2, 8, dtype=torch.bool)
    out = model(x, t=torch.rand(2), text=["walk", "run"], length_mask=valid)
    assert out.shape == (2, 8, 273)
    assert torch.isfinite(out).all()
    out[..., 5:].square().mean().backward()
    root_grad = sum(
        float(param.grad.abs().sum())
        for param in model.root_backbone.parameters()
        if param.grad is not None
    )
    assert root_grad == 0.0
    assert any(param.grad is not None for param in model.body_backbone.parameters())


def test_kimodo_like_masked_root_bridge_uses_complete_root_prediction(tmp_path):
    model = _kimodo_like_model(tmp_path)
    model.train()
    x = torch.randn(1, 8, 546)
    x[..., 273:] = 0
    x[:, 3, 273:278] = 1
    x[:, 3, :5] = torch.tensor([80.0, 90.0, 100.0, 0.0, 1.0])
    valid = torch.ones(1, 8, dtype=torch.bool)
    details = model(
        x,
        t=torch.tensor([0.5]),
        text=["walk"],
        length_mask=valid,
        return_details=True,
    )
    assert torch.equal(details.root_for_body, details.root_prediction_raw)
    assert not torch.equal(details.root_for_body[:, 3], x[:, 3, :5])

    details.prediction[..., 5:].square().mean().backward()
    root_grad = sum(
        float(param.grad.abs().sum())
        for param in model.root_backbone.parameters()
        if param.grad is not None
    )
    assert root_grad == 0.0


def test_global_to_local_root_wrap_and_variable_length_boundaries(tmp_path):
    full, local = _write_root_stats(tmp_path)
    conditioner = KimodoRootConditioner(full, local, fps=30.0, variance_eps=0.0)
    root = torch.zeros(2, 3, 5)
    angles = torch.deg2rad(torch.tensor([[179.0, -179.0, -177.0], [45.0, 0.0, 0.0]]))
    root[..., 3] = torch.cos(angles)
    root[..., 4] = torch.sin(angles)
    root[0, :, 0] = torch.tensor([0.0, 0.1, 0.2])
    root[0, :, 2] = torch.tensor([0.0, -0.2, -0.4])
    root[..., 1] = 1.0
    out = conditioner(root, torch.tensor([3, 1]))
    expected_omega = torch.deg2rad(torch.tensor(2.0)) * 30.0
    assert torch.allclose(out[0, :2, 0], expected_omega.expand(2), atol=1e-5)
    assert torch.allclose(out[0, 2, :3], out[0, 1, :3])
    assert torch.equal(out[1, 0, :3], torch.zeros(3))
    assert torch.equal(out[..., 3], torch.ones(2, 3))


def test_flow_state_uses_imputed_clean_estimate_contract():
    x0 = torch.randn(2, 5, 273)
    x0[..., 269:273] = (torch.rand(2, 5, 4) > 0.5).float()
    obs = torch.zeros_like(x0)
    mask = torch.zeros_like(x0, dtype=torch.bool)
    mask[:, 0, :3] = True
    obs[mask] = x0[mask]
    t = torch.full((2,), 0.4)
    state = build_flow_state(x0, obs, mask, t)
    v = torch.randn(2, 5, 269)
    clean = clean_from_velocity(state["z_cont_imp"], t, v)
    assert clean.shape == (2, 5, 269)
    assert torch.allclose(state["z_imp"][:, 0, :3], obs[:, 0, :3])
    assert float(state["z_imp"][..., 269:273].min()) >= 0.0
    assert float(state["z_imp"][..., 269:273].max()) <= 1.0


def test_x0_prediction_contract_returns_clean_prediction():
    z = torch.randn(2, 5, 269)
    t = torch.rand(2)
    pred_x0 = torch.randn(2, 5, 269)
    clean = predict_clean_cont(z, t, pred_x0, "x0")
    assert torch.equal(clean, pred_x0)


def test_contacts_not_normalized():
    mean = torch.randn(273)
    std = torch.rand(273) + 0.1
    norm = HY273Normalizer(mean, std)
    x = torch.randn(2, 4, 273)
    x[..., 269:273] = (torch.rand(2, 4, 4) > 0.5).float()
    y = norm.normalize(x)
    z = norm.denormalize(y)
    assert torch.equal(y[..., 269:273], x[..., 269:273])
    assert torch.allclose(z, x, atol=1e-5)


def test_yaw_rotation_feature_global_positions_equivariant():
    x = torch.zeros(1, 6, 273)
    x[..., 3] = 1.0
    joints = torch.randn(1, 6, 22, 3)
    x[..., 5:71] = joints.reshape(1, 6, 66)
    angle = torch.tensor([0.5])
    err = transform_equivariance_error(x, angle)
    assert float(err) < 1e-5


def test_clean_velocity_losses_zero_for_matching_clean_motion():
    x = torch.randn(2, 6, 273)
    x[..., 3] = 1.0
    x[..., 4] = 0.0
    x[..., 269:273] = (torch.rand(2, 6, 4) > 0.5).float()
    valid = torch.ones(2, 6, dtype=torch.bool)
    losses = compute_clean_semantic_losses(x, x.clone(), valid, fps=30.0, contact_threshold=0.5)
    assert float(losses["clean_root_vel"]) == 0.0
    assert float(losses["clean_joint_vel"]) == 0.0
    assert torch.isfinite(losses["foot_lock"])


def test_semantic_weighted_mse_reduces_each_block_before_weighting():
    pred = torch.zeros(1, 2, 269)
    target = torch.zeros_like(pred)
    target[..., 0:3] = 1.0
    target[..., 3:5] = 2.0
    target[..., 5:71] = 3.0
    target[..., 71:203] = 4.0
    target[..., 203:269] = 5.0
    mask = torch.ones_like(pred, dtype=torch.bool)
    args = Namespace(
        representation_loss_mode="semantic_weighted",
        representation_loss_scale=1.0,
        root_heading_loss_weight=1.0,
        velocity_loss_weight=1.0,
    )
    total, raw, weighted = representation_mse_loss(pred, target, mask, args)
    expected = (10 * 1 + 2 * 4 + 10 * 9 + 10 * 16 + 3 * 25) / 35
    assert torch.allclose(total, torch.tensor(float(expected)))
    assert torch.allclose(raw["rot6d"], torch.tensor(16.0))
    assert torch.allclose(sum(weighted.values()), total)


def test_x0_prediction_can_use_capped_velocity_loss_space():
    z = torch.tensor([[[0.5]], [[-0.25]]])
    x0_target = torch.tensor([[[1.0]], [[0.75]]])
    x0_pred = torch.tensor([[[0.8]], [[1.25]]])
    t = torch.tensor([0.5, 0.99])
    unused = torch.zeros_like(z)

    pred, target, resolved = representation_loss_pair(
        z_cont_imp=z,
        t=t,
        x0_hat_cont=x0_pred,
        x0_target_cont=x0_target,
        v_pred_cont=unused,
        v_target_cont=unused,
        prediction_type="x0",
        loss_space="velocity",
        velocity_t_eps=0.05,
    )

    denom = torch.tensor([0.5, 0.05]).view(2, 1, 1)
    assert resolved == "velocity"
    assert torch.allclose(pred, (x0_pred - z) / denom)
    assert torch.allclose(target, (x0_target - z) / denom)
    assert torch.allclose((pred - target).square(), (x0_pred - x0_target).square() / denom.square())


@pytest.mark.parametrize("num_steps", [20, 32, 64, 100])
def test_x0_prediction_ode_velocity_reaches_clean_endpoint(num_steps):
    z = torch.tensor([[[0.0]]], dtype=torch.float64)
    x0 = torch.tensor([[[1.0]]], dtype=torch.float64)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.tensor([step / num_steps], dtype=torch.float64)
        velocity = prediction_velocity_cont(
            z,
            t,
            x0,
            x0,
            "x0",
            velocity_t_eps=1e-4,
        )
        z = z + dt * velocity
    assert torch.allclose(z, x0, atol=1e-12, rtol=0.0)


def test_auto_representation_loss_space_preserves_legacy_behavior():
    z = torch.randn(2, 3, 4)
    x0 = torch.randn_like(z)
    v = torch.randn_like(z)
    t = torch.tensor([0.25, 0.75])

    pred, target, resolved = representation_loss_pair(
        z_cont_imp=z,
        t=t,
        x0_hat_cont=x0,
        x0_target_cont=x0 + 1,
        v_pred_cont=v,
        v_target_cont=v + 1,
        prediction_type="velocity",
        loss_space="auto",
        velocity_t_eps=0.05,
    )

    assert resolved == "velocity"
    assert pred is v
    assert torch.allclose(target, v + 1)


def test_jit_timestep_config_overrides_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([])
    args = merge_config(
        args,
        {
            "train": {
                "time_schedule": "logit_normal",
                "denoiser_p_mean": -0.8,
                "denoiser_p_std": 0.8,
                "grad_clip": 0.75,
            }
        },
    )

    assert args.time_schedule == "logit_normal"
    assert args.denoiser_p_mean == pytest.approx(-0.8)
    assert args.denoiser_p_std == pytest.approx(0.8)
    assert args.grad_clip == pytest.approx(0.75)


def test_fk_position_consistency_zero_for_neutral_identity_pose_and_has_gradient():
    frames = 3
    x = torch.zeros(1, frames, 273, requires_grad=True)
    neutral = load_smplx22_neutral_joints()
    identity = torch.eye(3).reshape(1, 1, 1, 3, 3).expand(1, frames, 22, 3, 3)
    with torch.no_grad():
        x[..., 3] = 1.0
        x[..., 5:71] = neutral.reshape(1, 1, 66)
        x[..., 71:203] = matrix_to_cont6d(identity).reshape(1, frames, 132)
    normalizer = HY273Normalizer(torch.zeros(273), torch.ones(273))
    valid = torch.ones(1, frames, dtype=torch.bool)
    loss, mean_cm = fk_position_consistency_loss(
        x,
        torch.zeros_like(x),
        torch.zeros_like(x, dtype=torch.bool),
        valid,
        normalizer,
        scale_m=0.05,
    )
    assert float(loss) < 1e-10
    assert float(mean_cm) < 1e-5

    perturbed = x.detach().clone().requires_grad_(True)
    with torch.no_grad():
        perturbed[..., 5 + 20 * 3] += 0.10
    loss, _ = fk_position_consistency_loss(
        perturbed,
        torch.zeros_like(perturbed),
        torch.zeros_like(perturbed, dtype=torch.bool),
        valid,
        normalizer,
        scale_m=0.05,
    )
    loss.backward()
    assert float(loss) > 0.0
    assert perturbed.grad is not None
    assert torch.isfinite(perturbed.grad).all()
    assert float(perturbed.grad.abs().sum()) > 0.0


def test_trace_streams_are_reproducible_and_independent():
    first = trace_stream_seed(3407, rank=2, step=300123, micro_step=1, stream=4)
    assert first == trace_stream_seed(3407, rank=2, step=300123, micro_step=1, stream=4)
    assert first != trace_stream_seed(3407, rank=2, step=300123, micro_step=1, stream=5)
    assert first != trace_stream_seed(3407, rank=3, step=300123, micro_step=1, stream=4)


def test_resume_cursor_round_trip_and_legacy_fail_closed():
    state = make_train_state(
        next_epoch=1795,
        next_step_in_epoch=134,
        optimizer_steps_per_epoch=167,
        world_size=4,
        batch_size_per_rank=16,
        gradient_accumulation_steps=2,
    )
    args = Namespace(
        resume_epoch=-1,
        resume_step_in_epoch=-1,
        require_exact_resume_cursor=True,
    )
    assert resolve_resume_cursor(args, {"train_state": state}, 167, 128) == (1795, 134)

    legacy_args = Namespace(
        resume_epoch=1795,
        resume_step_in_epoch=134,
        require_exact_resume_cursor=True,
    )
    assert resolve_resume_cursor(legacy_args, {"epoch": 1795}, 167, 128) == (1795, 134)
    with pytest.raises(RuntimeError, match="no exact train_state cursor"):
        resolve_resume_cursor(args, {"epoch": 1795}, 167, 128)


def test_checkpoint_persists_authoritative_next_cursor(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    args = Namespace(name="cursor_test")
    state = make_train_state(
        next_epoch=8,
        next_step_in_epoch=0,
        optimizer_steps_per_epoch=5,
        world_size=1,
        batch_size_per_rank=2,
        gradient_accumulation_steps=1,
    )
    path = tmp_path / "latest.pt"
    normalizer = HY273Normalizer(torch.zeros(273), torch.ones(273))
    save_checkpoint(
        path,
        model,
        optimizer,
        args,
        epoch=7,
        step=35,
        train_state=state,
        normalizer_state=normalizer.state_dict(),
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 8
    assert checkpoint["step"] == 35
    assert checkpoint["train_state"] == state
    validate_normalizer_contract(normalizer, checkpoint["normalizer"], str(path))
    changed = HY273Normalizer(torch.ones(273), torch.ones(273))
    with pytest.raises(RuntimeError, match="normalizer tensor mismatch"):
        validate_normalizer_contract(changed, checkpoint["normalizer"], str(path))


def test_real_phase_transition_checkpoint_contract(tmp_path):
    parser = build_arg_parser()
    stage1 = parser.parse_args(["--config", "configs/redenoise_kimodo_like_stage1.yaml"])
    stage1 = merge_config(stage1, load_yaml(stage1.config), explicit_cli={"config"})
    stage2 = parser.parse_args(["--config", "configs/redenoise_kimodo_like_stage2_control.yaml"])
    stage2 = merge_config(stage2, load_yaml(stage2.config), explicit_cli={"config"})
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage1.lr)
    normalizer = HY273Normalizer(torch.zeros(273), torch.ones(273))
    cursor = make_train_state(
        next_epoch=1,
        next_step_in_epoch=0,
        optimizer_steps_per_epoch=1,
        world_size=8,
        batch_size_per_rank=16,
        gradient_accumulation_steps=1,
    )
    path = tmp_path / "stage1.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        stage1,
        epoch=0,
        step=200_000,
        train_state=cursor,
        normalizer_state=normalizer.state_dict(),
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_resume_contract(
        stage2,
        checkpoint["args"],
        str(path),
        allowed_mismatches=(
            "training_phase",
            "control_modes",
            "control_cont_loss_weight",
            "control_contact_loss_weight",
        ),
    )
    validate_normalizer_contract(normalizer, checkpoint["normalizer"], str(path))

    complete = parser.parse_args(
        ["--config", "configs/redenoise_kimodo_complete_stage2_control.yaml"]
    )
    complete = merge_config(
        complete, load_yaml(complete.config), explicit_cli={"config"}
    )
    validate_resume_contract(
        complete,
        checkpoint["args"],
        str(path),
        allowed_mismatches=(
            "training_phase",
            "control_modes",
            "control_cont_loss_weight",
            "control_contact_loss_weight",
            "control_dense_min_fraction",
            "control_include_endpoint_rotations",
            "endpoint_preset",
            "endpoint_subset_mode",
            "endpoint_root_ref_mode",
        ),
    )


def test_sha256_file_hashes_checkpoint_parent_bytes(tmp_path):
    path = tmp_path / "parent.pt"
    path.write_bytes(b"parent-checkpoint")
    assert sha256_file(path) == "259299da489be859340845f381c27a9fec3bfb828593801e448cb10ecc8d123b"


def test_fk_consistency_warmup_and_model_hash_are_deterministic():
    weight, factor = effective_fk_consistency_weight(0.07, 5000, optimizer_step=0)
    assert weight == pytest.approx(0.07 / 5000)
    assert factor == pytest.approx(1.0 / 5000)
    weight, factor = effective_fk_consistency_weight(0.07, 5000, optimizer_step=4999)
    assert weight == pytest.approx(0.07)
    assert factor == pytest.approx(1.0)

    torch.manual_seed(123)
    first = torch.nn.Linear(4, 3)
    torch.manual_seed(123)
    second = torch.nn.Linear(4, 3)
    assert hash_model_state(first) == hash_model_state(second)
    with torch.no_grad():
        second.weight[0, 0].add_(1.0)
    assert hash_model_state(first) != hash_model_state(second)
