from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from data.hy273_interaction_dataset import apply_shared_interaction_gauge
from data.hy273_unified_actor_batcher import PiecewiseTaskScheduler
from data.hy273_multitask_scheduler import (
    KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
    edit_pattern_from_draw,
    hml_capability_from_draw,
)
from models.raw_motion.hy273_actor_exchange import BidirectionalActorExchange
from models.raw_motion.flow_schedule import sample_timesteps_with_low_t_mixture
from models.raw_motion.hy273_interaction_losses import (
    HY273InteractionLossBundle,
    HY273InteractionLossWeights,
    compute_hy273_interaction_loss,
)
from models.raw_motion.hy273_multitask_condition import (
    ABSOLUTE_TEXT_PROFILE,
    CapabilityId,
    INTERACTION_TEXT_PROFILE,
    TaskId,
    TrainStream,
    make_absent_condition,
)
from models.raw_motion.hy273_normalizer import HY273Normalizer, apply_yaw_rotation
from models.raw_motion.hy273_unified_edit_losses import (
    UnifiedEditLossWeights,
    compute_unified_edit_loss,
)
from models.raw_motion.hy273_slices import (
    DIM_HY273,
    GLOBAL_ROT_SLICE,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    LOCAL_ROOT_DIM,
    ROOT_SLICE,
    fk_positions_from_global_rot6d,
    reconstruct_global_joints_from_features,
)
from models.raw_motion.hy273_unified_actor_flow import HY273UnifiedActorFlow
from models.raw_motion.hytext_cache import (
    LLM2VEC_CACHE_FORMAT,
    hytext_profile_key,
)
from models.raw_motion.kimodo_context_flow_dit import HY273KimodoContextFlow
from tools.build_hy273_unified_actor_stats import (
    YAW_TARGETS,
    _interaction_versions,
)
from sample_hy273_multitask import sample_hy273_interaction_ode
from train_hy273_unified_actor import (
    FULLTEXT_STAGE_A_CONTRACT,
    FULLTEXT_STAGE_B_CONTRACT,
    FULLTEXT_STAGE_B_CONTINUE_CONTRACT,
    FULLTEXT_REACTION_V2_STAGE_B_CONTRACT,
    MetricWindow,
    _masked_gradient_rms,
    globalize_ratio_terms,
    load_config,
    make_interaction_weights,
    validate_config,
    validate_fulltext_phase_contract,
    validate_resume_config,
    validate_resume_run_name,
)


def test_masked_gradient_rms_accepts_reaction_and_two_actor_shapes() -> None:
    reaction_gradient = torch.ones(2, 4, DIM_HY273)
    reaction_valid = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    assert float(_masked_gradient_rms(reaction_gradient, reaction_valid)) == 1.0

    interaction_gradient = torch.ones(2, 2, 4, DIM_HY273)
    interaction_valid = reaction_valid[:, None].expand(2, 2, 4)
    assert float(_masked_gradient_rms(interaction_gradient, interaction_valid)) == 1.0

    with pytest.raises(ValueError, match="non-feature axis"):
        _masked_gradient_rms(
            reaction_gradient,
            reaction_valid[:, None],
        )


ENCODER_ID = "unified-actor-test-encoder"
PROMPT_VERSION = "unified-actor-test-prompt"


def _build_cache(root: Path) -> None:
    shard = root / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    rows = (
        (ABSOLUTE_TEXT_PROFILE, "walk"),
        (ABSOLUTE_TEXT_PROFILE, ""),
        (INTERACTION_TEXT_PROFILE, "two people shake hands"),
        (INTERACTION_TEXT_PROFILE, ""),
    )
    embeddings = np.arange(len(rows) * 6, dtype=np.float32).reshape(
        len(rows), 1, 6
    )
    np.save(shard / "ctxt.npy", embeddings.astype(np.float16))
    np.save(
        shard / "vtxt.npy",
        np.zeros((len(rows), 1, 1), dtype=np.float16),
    )
    np.save(shard / "ctxt_len.npy", np.ones(len(rows), dtype=np.int16))
    index = {}
    for row_index, (profile, text) in enumerate(rows):
        key = hytext_profile_key(
            text,
            profile,
            encoder_identity=ENCODER_ID,
            prompt_template_version=PROMPT_VERSION,
        )
        index[key] = {
            "shard": "shard_00000",
            "row": row_index,
            "text": text,
            "profile": profile,
        }
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": LLM2VEC_CACHE_FORMAT,
                "encoder_identity": ENCODER_ID,
                "prompt_template_version": PROMPT_VERSION,
                "ctxt_dim": 6,
                "vtxt_dim": 1,
                "max_length_llm": 1,
                "encoding_batch_size": 1,
                "storage_dtype": "fp16",
            }
        ),
        encoding="utf-8",
    )


def _build_stats(root: Path) -> tuple[Path, Path]:
    full = root / "full"
    local = root / "local_root"
    full.mkdir(parents=True)
    local.mkdir(parents=True)
    np.save(full / "Mean.npy", np.zeros(DIM_HY273, dtype=np.float32))
    np.save(full / "Std.npy", np.ones(DIM_HY273, dtype=np.float32))
    np.save(local / "Mean.npy", np.zeros(LOCAL_ROOT_DIM, dtype=np.float32))
    np.save(local / "Std.npy", np.ones(LOCAL_ROOT_DIM, dtype=np.float32))
    return full, local


def _tiny_model(tmp_path: Path) -> HY273UnifiedActorFlow:
    cache = tmp_path / "cache"
    _build_cache(cache)
    full, local = _build_stats(tmp_path / "stats")
    return HY273UnifiedActorFlow(
        hidden_dim=16,
        num_heads=4,
        root_depth_double=1,
        root_depth_single=1,
        body_depth_double=1,
        body_depth_single=1,
        mlp_ratio=1.0,
        dropout=0.0,
        max_text_tokens=1,
        text_encoder="llm2vec_cache",
        hytext_cache_dir=str(cache),
        hytext_ctxt_dim=6,
        hytext_vtxt_dim=1,
        motion_stats_dir=str(full),
        local_root_stats_dir=str(local),
        max_frames=8,
        normalize_contacts=True,
        text_global_conditioning="llm2vec_tokens_only",
        actor_exchange_dim=8,
        actor_exchange_heads=2,
    )


def _tiny_base_model(tmp_path: Path) -> HY273KimodoContextFlow:
    cache = tmp_path / "cache"
    _build_cache(cache)
    full, local = _build_stats(tmp_path / "stats")
    return HY273KimodoContextFlow(
        hidden_dim=16,
        num_heads=4,
        root_depth_double=1,
        root_depth_single=1,
        body_depth_double=1,
        body_depth_single=1,
        mlp_ratio=1.0,
        dropout=0.0,
        max_text_tokens=1,
        text_encoder="llm2vec_cache",
        hytext_cache_dir=str(cache),
        hytext_ctxt_dim=6,
        hytext_vtxt_dim=1,
        motion_stats_dir=str(full),
        local_root_stats_dir=str(local),
        max_frames=8,
        normalize_contacts=True,
        text_global_conditioning="llm2vec_tokens_only",
    )


def test_actor_exchange_is_exact_noop_for_one_actor_and_keeps_grad_graph() -> None:
    module = BidirectionalActorExchange(12, exchange_dim=8, num_heads=2)
    value = torch.randn(3, 5, 12, requires_grad=True)
    valid = torch.ones(3, 5, dtype=torch.bool)
    output = module(value, valid, scene_batch_size=3, actor_count=1)
    torch.testing.assert_close(output, value, rtol=0.0, atol=0.0)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_actor_exchange_is_permutation_equivariant() -> None:
    torch.manual_seed(7)
    module = BidirectionalActorExchange(12, exchange_dim=8, num_heads=2).eval()
    torch.nn.init.normal_(module.up.weight, std=0.1)
    torch.nn.init.normal_(module.up.bias, std=0.1)
    value = torch.randn(4, 6, 12)
    valid = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
            [True, True, True, False, False, False],
        ]
    )
    baseline = module(value, valid, scene_batch_size=2, actor_count=2)
    swapped_value = value.reshape(2, 2, 6, 12).flip(1).reshape_as(value)
    swapped_valid = valid.reshape(2, 2, 6).flip(1).reshape_as(valid)
    swapped = module(
        swapped_value,
        swapped_valid,
        scene_batch_size=2,
        actor_count=2,
    )
    expected = baseline.reshape(2, 2, 6, 12).flip(1).reshape_as(baseline)
    torch.testing.assert_close(swapped, expected, rtol=1e-5, atol=1e-6)


def test_shared_gauge_then_actor_swap_preserves_pair_geometry() -> None:
    motion = torch.zeros(2, 5, DIM_HY273)
    motion[0, :, ROOT_SLICE] = torch.tensor([1.0, 0.9, -2.0])
    motion[1, :, ROOT_SLICE] = torch.tensor([-0.5, 1.0, 0.75])
    motion[:, :, HEADING_SLICE] = torch.tensor([1.0, 0.0])
    joints = torch.linspace(-0.4, 0.4, 22 * 3).reshape(22, 3)
    motion[0, :, JOINT_POS_SLICE] = joints.reshape(-1)
    motion[1, :, JOINT_POS_SLICE] = (joints + 0.2).reshape(-1)
    before = reconstruct_global_joints_from_features(motion)
    transformed, _ = apply_shared_interaction_gauge(motion, phi=1.3)
    after = reconstruct_global_joints_from_features(transformed)
    before_distance = torch.cdist(before[0], before[1])
    after_distance = torch.cdist(after[0], after[1])
    torch.testing.assert_close(after_distance, before_distance, rtol=1e-5, atol=1e-5)
    swapped = transformed.flip(0)
    torch.testing.assert_close(swapped[0], transformed[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(swapped[1], transformed[0], rtol=0.0, atol=0.0)


def test_interaction_stats_apply_crop_before_shared_gauge(tmp_path: Path) -> None:
    frames = 320
    pair = torch.zeros(2, frames, DIM_HY273)
    pair[:, :, HEADING_SLICE] = torch.tensor([1.0, 0.0])
    pair[0, :, ROOT_SLICE.start] = torch.arange(frames) * 0.01
    pair[1, :, ROOT_SLICE.start] = 1.0 + torch.arange(frames) * 0.01
    np.save(tmp_path / "p1.npy", pair[0].numpy().astype(np.float32))
    np.save(tmp_path / "p2.npy", pair[1].numpy().astype(np.float32))
    versions = _interaction_versions(
        tmp_path,
        {"frames": frames, "clip_id": "test", "person1": "p1.npy", "person2": "p2.npy"},
        max_frames=300,
        crop_samples=8,
    )
    assert len(versions) == 8 * len(YAW_TARGETS)
    assert {tuple(version.shape) for version in versions} == {
        (2, 300, DIM_HY273)
    }
    for version in versions:
        assert float(version[0, 0, ROOT_SLICE.start]) == 0.0
        assert float(version[0, 0, ROOT_SLICE.start + 2]) == 0.0


def test_interaction_relation_loss_gt_zero_and_flow_time_gate() -> None:
    target = torch.zeros(2, 2, 4, DIM_HY273)
    target[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    valid = torch.ones(2, 2, 4, dtype=torch.bool)
    exact = compute_hy273_interaction_loss(
        prediction_physical=target.clone(),
        target_physical=target,
        actor_valid=valid,
        timesteps=torch.tensor([0.5, 0.7]),
    )
    assert float(exact.total) == 0.0
    assert all(float(term.numerator) == 0.0 for term in exact.terms.values())

    prediction = target.clone()
    prediction[:, 1, :, ROOT_SLICE.start] = 1.0
    gated = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=valid,
        timesteps=torch.tensor([0.1, 0.1]),
        weights=HY273InteractionLossWeights(min_flow_t=0.2),
    )
    assert float(gated.total) == 0.0
    assert float(gated.active_scene_fraction) == 0.0

    active = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=valid,
        timesteps=torch.tensor([0.5, 0.5]),
        weights=HY273InteractionLossWeights(min_flow_t=0.2),
    )
    assert float(active.total) > 0.0
    assert float(active.active_scene_fraction) == 1.0

    source_absent = valid.clone()
    source_absent[0, 0] = False
    partly_active = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=source_absent,
        timesteps=torch.tensor([0.5, 0.5]),
        weights=HY273InteractionLossWeights(min_flow_t=0.2),
    )
    assert float(partly_active.active_scene_fraction) == 0.5


def _reaction_v2_test_weights(**overrides: object) -> HY273InteractionLossWeights:
    values: dict[str, object] = {
        "relative_root": 0.0,
        "relative_heading": 0.0,
        "joint_distance": 1.0,
        "close_joint_vector": 0.0,
        "relative_root_radius": 1.0,
        "relative_root_bearing": 1.0,
        "partner_facing": 1.0,
        "soft_proximity": 1.0,
        "false_close": 1.0,
        "distance_include_predicted_near": True,
        "coarse_min_flow_t": 0.0,
        "fine_min_flow_t": 0.55,
    }
    values.update(overrides)
    return HY273InteractionLossWeights(**values)


def _reaction_v4_layout_weights(
    **overrides: object,
) -> HY273InteractionLossWeights:
    values: dict[str, object] = {
        "relative_root": 1.0,
        "relative_heading": 1.0,
        "joint_distance": 0.0,
        "close_joint_vector": 0.0,
        "relative_root_radius": 0.0,
        "relative_root_bearing": 0.0,
        "partner_facing": 0.0,
        "soft_proximity": 0.0,
        "false_close": 0.0,
        "root_scale_m": 0.25,
        "heading_beta": 0.10,
        "coarse_min_flow_t": 0.0,
        "fine_min_flow_t": 0.20,
    }
    values.update(overrides)
    return HY273InteractionLossWeights(**values)


def _reaction_v5_event_weights(
    **overrides: object,
) -> HY273InteractionLossWeights:
    values: dict[str, object] = {
        "relative_root": 0.0,
        "relative_heading": 0.0,
        "joint_distance": 0.0,
        "close_joint_vector": 0.0,
        "relative_root_radius": 0.0,
        "relative_root_bearing": 0.0,
        "partner_facing": 0.0,
        "soft_proximity": 0.0,
        "false_close": 0.0,
        "scene_proximity": 0.0,
        "precontact_false_close": 0.0,
        "first_contact_cdf": 0.0,
        "layout_contact_threshold_m": 0.20,
        "proximity_temperature_m": 0.03,
        "precontact_directional_strength": 0.25,
        "coarse_min_flow_t": 0.0,
        "fine_min_flow_t": 0.20,
    }
    values.update(overrides)
    return HY273InteractionLossWeights(**values)


def _identity_interaction_motion(frames: int = 4) -> torch.Tensor:
    motion = torch.zeros(1, 2, frames, DIM_HY273)
    motion[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    motion[..., GLOBAL_ROT_SLICE] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ).repeat(22)
    return motion


def _reaction_v5_1_contact_weights(
    **overrides: object,
) -> HY273InteractionLossWeights:
    values: dict[str, object] = {
        "fk_contact_map_positive": 1.0,
        "fk_contact_map_negative": 1.0,
        "fk_contact_vector": 1.0,
        "fk_contact_transition": 1.0,
        "fk_contact_threshold_m": 0.15,
        "fk_contact_temperature_m": 0.02,
        "fk_contact_vector_scale_m": 0.05,
        "fk_contact_transition_beta": 0.10,
    }
    values.update(overrides)
    return replace(_reaction_v5_event_weights(), **values)


def test_reaction_v5_1_full_contact_is_zero_for_exact_prediction() -> None:
    target = _identity_interaction_motion(frames=4)
    target[:, 1, 3, ROOT_SLICE.start] = 3.0
    bundle = compute_hy273_interaction_loss(
        prediction_physical=target.clone(),
        target_physical=target,
        actor_valid=torch.ones(1, 2, 4, dtype=torch.bool),
        timesteps=torch.tensor([0.8]),
        weights=_reaction_v5_1_contact_weights(),
    )
    for name in (
        "interaction_fk_contact_map_positive",
        "interaction_fk_contact_map_negative",
        "interaction_fk_contact_vector",
        "interaction_fk_contact_transition",
    ):
        assert float(bundle.terms[name].raw) == 0.0


def test_reaction_v5_1_fk_contact_error_reaches_root_and_rotation_channels() -> None:
    target = _identity_interaction_motion(frames=3)
    prediction = target.clone()
    prediction[:, 1] = apply_yaw_rotation(
        prediction[:, 1], torch.tensor(0.6)
    )
    prediction[:, 1, :, ROOT_SLICE.start] += 0.08
    prediction.requires_grad_()
    bundle = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=torch.ones(1, 2, 3, dtype=torch.bool),
        timesteps=torch.tensor([0.8]),
        weights=_reaction_v5_1_contact_weights(),
    )
    full_contact = sum(
        bundle.terms[name].weighted
        for name in (
            "interaction_fk_contact_map_positive",
            "interaction_fk_contact_map_negative",
            "interaction_fk_contact_vector",
            "interaction_fk_contact_transition",
        )
    )
    assert float(full_contact) > 0.0
    full_contact.backward()
    assert prediction.grad is not None
    assert float(prediction.grad[:, 1, :, ROOT_SLICE].norm()) > 0.0
    assert float(prediction.grad[:, 1, :, GLOBAL_ROT_SLICE].norm()) > 0.0


def test_reaction_v5_1_negative_contact_separates_exact_root_overlap() -> None:
    target = _identity_interaction_motion(frames=1)
    target[:, 1, :, ROOT_SLICE.start] = 3.0
    prediction = _identity_interaction_motion(frames=1).requires_grad_()
    weights = _reaction_v5_1_contact_weights(
        fk_contact_map_positive=0.0,
        fk_contact_vector=0.0,
        fk_contact_transition=0.0,
    )
    bundle = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=torch.ones(1, 2, 1, dtype=torch.bool),
        timesteps=torch.tensor([0.8]),
        weights=weights,
    )
    term = bundle.terms["interaction_fk_contact_map_negative"]
    assert float(term.raw) > 0.0
    term.weighted.backward()
    assert prediction.grad is not None
    reactor_root_x_gradient = prediction.grad[0, 1, 0, ROOT_SLICE.start]
    assert torch.isfinite(reactor_root_x_gradient)
    assert float(reactor_root_x_gradient) < 0.0
    assert float(prediction.grad[0, 1, 0, JOINT_POS_SLICE].abs().max()) < 1e-6


def test_reaction_v5_1_negative_contact_uses_fk_vertical_root_carrier() -> None:
    target = _identity_interaction_motion(frames=1)
    pelvis_y = JOINT_POS_SLICE.start + 1
    target[:, 1, :, pelvis_y] = 3.0
    prediction = _identity_interaction_motion(frames=1).requires_grad_()
    weights = _reaction_v5_1_contact_weights(
        fk_contact_map_positive=0.0,
        fk_contact_vector=0.0,
        fk_contact_transition=0.0,
    )
    bundle = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=torch.ones(1, 2, 1, dtype=torch.bool),
        timesteps=torch.tensor([0.8]),
        weights=weights,
    )
    bundle.terms["interaction_fk_contact_map_negative"].weighted.backward()
    assert prediction.grad is not None
    assert float(prediction.grad[0, 1, 0, pelvis_y]) < 0.0
    assert abs(float(prediction.grad[0, 1, 0, ROOT_SLICE.start + 1])) < 1e-8


def test_reaction_v5_1_contact_map_normalizes_positive_and_negative_pairs_separately() -> None:
    target = _identity_interaction_motion(frames=2)
    target[:, 1, 1, ROOT_SLICE.start] = 3.0
    prediction = target.clone()
    prediction[:, 1, 0, ROOT_SLICE.start] += 0.05
    prediction[:, 1, 1, ROOT_SLICE.start] -= 0.05
    bundle = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=torch.ones(1, 2, 2, dtype=torch.bool),
        timesteps=torch.tensor([0.8]),
        weights=_reaction_v5_1_contact_weights(
            fk_contact_vector=0.0,
            fk_contact_transition=0.0,
        ),
    )
    target_fk = fk_positions_from_global_rot6d(target.reshape(2, 2, DIM_HY273))
    target_distance = torch.cdist(target_fk[0], target_fk[1])
    positive = int((target_distance < 0.15).sum().item())
    total = 2 * 22 * 22
    positive_term = bundle.terms["interaction_fk_contact_map_positive"]
    negative_term = bundle.terms["interaction_fk_contact_map_negative"]
    assert int(positive_term.denominator.item()) == positive
    assert int(negative_term.denominator.item()) == total - positive
    assert float(positive_term.raw) > 0.0
    assert float(negative_term.raw) > 0.0


def test_reaction_v5_1_transition_penalizes_shifted_onset_and_release() -> None:
    target = _identity_interaction_motion(frames=4)
    prediction = target.clone()
    target[:, 1, :, ROOT_SLICE.start] = torch.tensor([3.0, 0.0, 0.0, 3.0])
    prediction[:, 1, :, ROOT_SLICE.start] = torch.tensor([3.0, 3.0, 0.0, 0.0])
    weights = _reaction_v5_1_contact_weights(
        fk_contact_map_positive=0.0,
        fk_contact_map_negative=0.0,
        fk_contact_vector=0.0,
    )
    exact = compute_hy273_interaction_loss(
        prediction_physical=target.clone(),
        target_physical=target,
        actor_valid=torch.ones(1, 2, 4, dtype=torch.bool),
        timesteps=torch.tensor([0.8]),
        weights=weights,
    )
    shifted = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=torch.ones(1, 2, 4, dtype=torch.bool),
        timesteps=torch.tensor([0.8]),
        weights=weights,
    )
    term = "interaction_fk_contact_transition"
    assert float(exact.terms[term].raw) == 0.0
    assert float(shifted.terms[term].raw) > 0.0


def test_reaction_v5_1_full_contact_respects_fine_gate_and_padding() -> None:
    target = _identity_interaction_motion(frames=2)
    prediction = target.clone()
    prediction[:, 1, :, ROOT_SLICE.start] = 0.4
    weights = _reaction_v5_1_contact_weights()
    gated = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=torch.ones(1, 2, 2, dtype=torch.bool),
        timesteps=torch.tensor([0.1]),
        weights=weights,
    )
    assert float(gated.total) == 0.0

    prediction[:, 1, 0] = target[:, 1, 0]
    valid = torch.tensor([[[True, False], [True, False]]])
    padded = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=valid,
        timesteps=torch.tensor([0.8]),
        weights=weights,
    )
    assert float(padded.total) == 0.0


def test_reaction_v5_2_full_contact_is_active_at_flow_epsilon() -> None:
    target = _identity_interaction_motion(frames=4)
    prediction = target.clone()
    target[:, 1, :, ROOT_SLICE.start] = torch.tensor([3.0, 0.0, 0.0, 3.0])
    prediction[:, 1, :, ROOT_SLICE.start] = torch.tensor([3.0, 3.0, 0.0, 0.0])
    weights = _reaction_v5_1_contact_weights(
        joint_distance=1.0,
        close_joint_vector=1.0,
        min_flow_t=0.0,
        coarse_min_flow_t=0.0,
        fine_min_flow_t=0.0,
    )
    bundle = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=torch.ones(1, 2, 4, dtype=torch.bool),
        timesteps=torch.tensor([1.0e-4]),
        weights=weights,
    )
    assert float(bundle.fine_active_scene_fraction) == 1.0
    for name in (
        "interaction_joint_distance",
        "interaction_close_joint_vector",
        "interaction_fk_contact_map_positive",
        "interaction_fk_contact_map_negative",
        "interaction_fk_contact_vector",
        "interaction_fk_contact_transition",
    ):
        assert bundle.terms[name].denominator.item() > 0
        assert float(bundle.terms[name].weighted) > 0.0


def test_reaction_v5_2_active_fraction_excludes_source_absent_scenes() -> None:
    target = _identity_interaction_motion(frames=3).repeat(2, 1, 1, 1)
    actor_valid = torch.ones(2, 2, 3, dtype=torch.bool)
    actor_valid[1, 0] = False
    bundle = compute_hy273_interaction_loss(
        prediction_physical=target.clone(),
        target_physical=target,
        actor_valid=actor_valid,
        timesteps=torch.full((2,), 1.0e-4),
        weights=_reaction_v5_1_contact_weights(
            min_flow_t=0.0,
            coarse_min_flow_t=0.0,
            fine_min_flow_t=0.0,
        ),
    )
    assert float(bundle.fine_active_scene_fraction) == 0.5
    numerator, denominator = bundle.diagnostic_ratios[
        "fine_active_scene_fraction"
    ]
    assert float(numerator) == 1.0
    assert float(denominator) == 2.0


def test_reaction_min_flow_t_is_noop_with_explicit_coarse_and_fine_gates() -> None:
    target = _identity_interaction_motion(frames=4)
    prediction = target.clone()
    prediction[:, 1, :, ROOT_SLICE.start] = torch.tensor([0.0, 0.4, 0.0, -0.4])
    baseline_weights = _reaction_v5_1_contact_weights(
        min_flow_t=0.2,
        coarse_min_flow_t=0.0,
        fine_min_flow_t=0.2,
    )
    candidate_weights = replace(baseline_weights, min_flow_t=0.0)
    bundles = []
    gradients = []
    for weights in (baseline_weights, candidate_weights):
        local_prediction = prediction.clone().requires_grad_(True)
        bundle = compute_hy273_interaction_loss(
            prediction_physical=local_prediction,
            target_physical=target,
            actor_valid=torch.ones(1, 2, 4, dtype=torch.bool),
            timesteps=torch.tensor([0.1]),
            weights=weights,
        )
        bundles.append(bundle)
        gradients.append(torch.autograd.grad(bundle.total, local_prediction)[0])
    torch.testing.assert_close(bundles[0].total, bundles[1].total, rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    for name in bundles[0].terms:
        torch.testing.assert_close(
            bundles[0].terms[name].weighted,
            bundles[1].terms[name].weighted,
            rtol=0,
            atol=0,
        )


def test_reaction_v4_layout_is_signed_3d_and_shared_yaw_invariant() -> None:
    target = torch.zeros(1, 2, 3, DIM_HY273)
    prediction = target.clone()
    target[:, 0, :, HEADING_SLICE] = torch.tensor([1.0, 0.0])
    prediction[:, 0, :, HEADING_SLICE] = torch.tensor([1.0, 0.0])
    target[:, 1, :, HEADING_SLICE] = torch.tensor([-1.0, 0.0])
    prediction[:, 1, :, HEADING_SLICE] = torch.tensor([0.0, 1.0])
    target[:, 1, :, ROOT_SLICE] = torch.tensor([1.0, 0.9, 0.25])
    prediction[:, 1, :, ROOT_SLICE] = torch.tensor([-0.6, 1.1, 0.40])
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    weights = _reaction_v4_layout_weights()

    before = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=valid,
        timesteps=torch.tensor([0.05]),
        weights=weights,
    )
    assert float(before.terms["interaction_relative_root"].raw) > 0.0
    assert float(before.terms["interaction_relative_heading"].raw) > 0.0

    angle = torch.tensor(1.1)
    after = compute_hy273_interaction_loss(
        prediction_physical=apply_yaw_rotation(prediction, angle),
        target_physical=apply_yaw_rotation(target, angle),
        actor_valid=valid,
        timesteps=torch.tensor([0.05]),
        weights=weights,
    )
    for name in (
        "interaction_relative_root",
        "interaction_relative_heading",
    ):
        torch.testing.assert_close(
            before.terms[name].raw,
            after.terms[name].raw,
            rtol=1e-5,
            atol=1e-6,
        )

    vertical_only = target.clone()
    vertical_only[:, 1, :, ROOT_SLICE.start + 1] += 0.2
    vertical = compute_hy273_interaction_loss(
        prediction_physical=vertical_only,
        target_physical=target,
        actor_valid=valid,
        timesteps=torch.tensor([0.05]),
        weights=_reaction_v4_layout_weights(relative_heading=0.0),
    )
    assert float(vertical.terms["interaction_relative_root"].raw) > 0.0


def _reaction_v4_phase_bundle(
    *,
    error_frame: int | None,
    target_root_x: list[float],
) -> HY273InteractionLossBundle:
    frames = len(target_root_x)
    target = torch.zeros(1, 2, frames, DIM_HY273)
    target[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    target[:, 1, :, ROOT_SLICE.start] = torch.tensor(target_root_x)
    prediction = target.clone()
    if error_frame is not None:
        prediction[:, 1, error_frame, ROOT_SLICE.start] += 0.10
    return compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=torch.ones(1, 2, frames, dtype=torch.bool),
        timesteps=torch.tensor([0.05]),
        weights=_reaction_v4_layout_weights(
            relative_heading=0.0,
            layout_initial_frames=1,
            layout_initial_multiplier=3.0,
            layout_precontact_multiplier=2.0,
            layout_contact_threshold_m=0.20,
        ),
    )


def test_reaction_v4_layout_phase_weights_initial_precontact_and_postcontact() -> None:
    # GT first becomes close at frame 2: phase weights are [3, 2, 1, 1].
    initial = _reaction_v4_phase_bundle(
        error_frame=0, target_root_x=[1.0, 1.0, 0.0, 0.0]
    )
    precontact = _reaction_v4_phase_bundle(
        error_frame=1, target_root_x=[1.0, 1.0, 0.0, 0.0]
    )
    postcontact = _reaction_v4_phase_bundle(
        error_frame=2, target_root_x=[1.0, 1.0, 0.0, 0.0]
    )
    initial_raw = initial.terms["interaction_relative_root"].raw
    precontact_raw = precontact.terms["interaction_relative_root"].raw
    postcontact_raw = postcontact.terms["interaction_relative_root"].raw
    torch.testing.assert_close(initial_raw, 3.0 * postcontact_raw)
    torch.testing.assert_close(precontact_raw, 2.0 * postcontact_raw)
    phase_numerator, phase_denominator = initial.diagnostic_ratios[
        "layout_phase_weight_mean"
    ]
    assert float(phase_numerator / phase_denominator) == pytest.approx(1.75)


def test_reaction_v4_layout_phase_handles_no_contact_and_frame0_contact() -> None:
    no_contact = _reaction_v4_phase_bundle(
        error_frame=None, target_root_x=[1.0, 1.0, 1.0, 1.0]
    )
    no_contact_num, no_contact_den = no_contact.diagnostic_ratios[
        "layout_phase_weight_mean"
    ]
    assert float(no_contact_num / no_contact_den) == pytest.approx(2.25)

    frame0_contact = _reaction_v4_phase_bundle(
        error_frame=None, target_root_x=[0.0, 0.0, 0.0, 0.0]
    )
    frame0_num, frame0_den = frame0_contact.diagnostic_ratios[
        "layout_phase_weight_mean"
    ]
    assert float(frame0_num / frame0_den) == pytest.approx(1.5)


def _reaction_v5_phase_bundle(
    *,
    error_frame: int,
    term: str,
) -> HY273InteractionLossBundle:
    target = torch.zeros(1, 2, 4, DIM_HY273)
    prediction = target.clone()
    target[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    prediction[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    target[:, 1, :, ROOT_SLICE.start] = torch.tensor([1.0, 1.0, 0.15, 0.15])
    prediction[:, 1, :, ROOT_SLICE.start] = target[:, 1, :, ROOT_SLICE.start]
    weights: dict[str, object] = {
        "layout_initial_frames": 1,
        "layout_initial_multiplier": 4.0,
        "layout_precontact_multiplier": 3.0,
    }
    if term == "bearing":
        prediction[:, 1, error_frame, ROOT_SLICE.start] = 0.0
        prediction[:, 1, error_frame, ROOT_SLICE.start + 2] = target[
            :, 1, error_frame, ROOT_SLICE.start
        ]
        weights["relative_root_bearing"] = 1.0
    elif term == "facing":
        prediction[:, 1, error_frame, HEADING_SLICE] = torch.tensor([0.0, 1.0])
        weights["partner_facing"] = 1.0
    elif term == "radius":
        prediction[:, 1, error_frame, ROOT_SLICE.start] += 0.10
        weights["relative_root_radius"] = 1.0
    else:
        raise AssertionError(term)
    return compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=torch.ones(1, 2, 4, dtype=torch.bool),
        timesteps=torch.tensor([0.05]),
        weights=_reaction_v5_event_weights(**weights),
    )


@pytest.mark.parametrize(
    ("term", "term_name"),
    [
        ("radius", "interaction_relative_root_radius"),
        ("bearing", "interaction_relative_root_bearing"),
        ("facing", "interaction_partner_facing"),
    ],
)
def test_reaction_v5_phase_weights_true_layout_terms(
    term: str,
    term_name: str,
) -> None:
    initial = _reaction_v5_phase_bundle(error_frame=0, term=term)
    precontact = _reaction_v5_phase_bundle(error_frame=1, term=term)
    postcontact = _reaction_v5_phase_bundle(error_frame=2, term=term)
    torch.testing.assert_close(
        initial.terms[term_name].raw,
        4.0 * postcontact.terms[term_name].raw,
    )
    torch.testing.assert_close(
        precontact.terms[term_name].raw,
        3.0 * postcontact.terms[term_name].raw,
    )


def _reaction_v5_event_bundle(
    *,
    target_root_x: list[float],
    prediction_root_x: list[float],
    timestep: float = 0.05,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, HY273InteractionLossBundle]:
    if len(target_root_x) != len(prediction_root_x):
        raise AssertionError("target and prediction lengths differ")
    frames = len(target_root_x)
    target = torch.zeros(1, 2, frames, DIM_HY273)
    prediction = target.clone()
    target[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    prediction[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    target[:, 1, :, ROOT_SLICE.start] = torch.tensor(target_root_x)
    prediction[:, 1, :, ROOT_SLICE.start] = torch.tensor(prediction_root_x)
    prediction.requires_grad_(requires_grad)
    bundle = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=torch.ones(1, 2, frames, dtype=torch.bool),
        timesteps=torch.tensor([timestep]),
        weights=_reaction_v5_event_weights(
            scene_proximity=1.0,
            precontact_false_close=1.0,
            first_contact_cdf=1.0,
        ),
    )
    return prediction, bundle


def test_reaction_v5_penalizes_benchmark_range_precontact_false_close() -> None:
    _, correct = _reaction_v5_event_bundle(
        target_root_x=[1.0], prediction_root_x=[1.0]
    )
    _, false_close_15cm = _reaction_v5_event_bundle(
        target_root_x=[1.0], prediction_root_x=[0.15]
    )
    _, safe_25cm = _reaction_v5_event_bundle(
        target_root_x=[1.0], prediction_root_x=[0.25]
    )
    term = "interaction_precontact_false_close"
    assert float(correct.terms[term].raw) == 0.0
    assert float(false_close_15cm.terms[term].raw) > 0.0
    assert float(safe_25cm.terms[term].raw) == 0.0
    assert float(false_close_15cm.terms["interaction_false_close"].raw) == 0.0


def test_reaction_v5_event_state_and_first_contact_timing_are_low_t_active() -> None:
    _, exact = _reaction_v5_event_bundle(
        target_root_x=[1.0, 1.0, 0.0, 0.0],
        prediction_root_x=[1.0, 1.0, 0.0, 0.0],
        timestep=0.05,
    )
    _, early = _reaction_v5_event_bundle(
        target_root_x=[1.0, 1.0, 0.0, 0.0],
        prediction_root_x=[1.0, 0.0, 0.0, 0.0],
        timestep=0.05,
    )
    assert float(
        early.terms["interaction_scene_proximity_negative"].raw
    ) > float(exact.terms["interaction_scene_proximity_negative"].raw)
    assert float(early.terms["interaction_first_contact_cdf"].raw) > float(
        exact.terms["interaction_first_contact_cdf"].raw
    )
    assert float(early.terms["interaction_joint_distance"].raw) == 0.0


def test_reaction_v5_scene_kl_is_zero_at_target_and_locally_well_directed() -> None:
    exact_prediction, exact = _reaction_v5_event_bundle(
        target_root_x=[0.21],
        prediction_root_x=[0.21],
        requires_grad=True,
    )
    term_name = "interaction_scene_proximity_negative"
    assert float(exact.terms[term_name].raw) == 0.0
    exact.terms[term_name].weighted.backward()
    # The logits-form KL is evaluated in float64, but its mathematically zero
    # derivative can retain roundoff at the 1e-15 scale after autograd.
    assert float(exact_prediction.grad.abs().max()) < 1e-12

    farther_prediction, farther = _reaction_v5_event_bundle(
        target_root_x=[0.21],
        prediction_root_x=[0.211],
        requires_grad=True,
    )
    farther.terms[term_name].weighted.backward()
    gradient = farther_prediction.grad[0, 1, 0, ROOT_SLICE.start]
    assert float(farther.terms[term_name].raw) > 0.0
    assert float(gradient) > 0.0


def test_reaction_v5_first_contact_gradient_ignores_later_contact_duration() -> None:
    target_root_x = [1.0, 1.0, 1.0, 1.0, 0.10, 0.10]
    prediction_once, once = _reaction_v5_event_bundle(
        target_root_x=target_root_x,
        prediction_root_x=[1.0, 0.10, 1.0, 1.0, 1.0, 1.0],
        requires_grad=True,
    )
    prediction_persistent, persistent = _reaction_v5_event_bundle(
        target_root_x=target_root_x,
        prediction_root_x=[1.0, 0.10, 0.10, 0.10, 1.0, 1.0],
        requires_grad=True,
    )
    term_name = "interaction_first_contact_cdf"
    torch.testing.assert_close(
        once.terms[term_name].raw,
        persistent.terms[term_name].raw,
    )
    once.terms[term_name].weighted.backward()
    persistent.terms[term_name].weighted.backward()
    once_root_gradient = prediction_once.grad[0, 1, :, ROOT_SLICE.start]
    persistent_root_gradient = prediction_persistent.grad[
        0, 1, :, ROOT_SLICE.start
    ]
    torch.testing.assert_close(
        once_root_gradient.sum(),
        persistent_root_gradient.sum(),
        rtol=1e-5,
        atol=1e-6,
    )
    assert float(once_root_gradient.abs().sum()) > 0.0


def test_reaction_v5_exact_overlap_has_directional_separation_gradient() -> None:
    prediction, bundle = _reaction_v5_event_bundle(
        target_root_x=[1.0],
        prediction_root_x=[0.0],
        requires_grad=True,
    )
    bundle.terms["interaction_precontact_false_close"].weighted.backward()
    root_gradient = prediction.grad[0, 1, 0, ROOT_SLICE.start]
    assert torch.isfinite(root_gradient)
    assert float(root_gradient.abs()) > 0.0
    assert float(prediction.grad[0, 1, 0, JOINT_POS_SLICE].abs().max()) == 0.0


def test_reaction_low_t_mixture_is_deterministic_and_material() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260805)
    timesteps, selected = sample_timesteps_with_low_t_mixture(
        20_000,
        torch.device("cpu"),
        schedule="logit_normal",
        p_mean=-0.8,
        p_std=0.8,
        low_t_fraction=0.30,
        low_t_max=0.15,
        generator=generator,
    )
    assert float(selected.float().mean()) == pytest.approx(0.30, abs=0.015)
    assert bool((timesteps[selected] <= 0.15).all())
    assert float((timesteps <= 0.03125).float().mean()) > 0.05

    replay_generator = torch.Generator(device="cpu").manual_seed(20260805)
    replay_timesteps, replay_selected = sample_timesteps_with_low_t_mixture(
        20_000,
        torch.device("cpu"),
        schedule="logit_normal",
        p_mean=-0.8,
        p_std=0.8,
        low_t_fraction=0.30,
        low_t_max=0.15,
        generator=replay_generator,
    )
    assert torch.equal(timesteps, replay_timesteps)
    assert torch.equal(selected, replay_selected)


def _reaction_p_only_bundle(
    *,
    source_x: float = 0.0,
    target_x: float = 0.10,
    prediction_x: float = 0.20,
    timestep: float = 0.8,
) -> tuple[torch.Tensor, HY273InteractionLossBundle]:
    target = torch.zeros(1, 2, 1, DIM_HY273)
    prediction = target.clone()
    target[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    prediction[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    target[:, 0, :, ROOT_SLICE.start] = source_x
    prediction[:, 0, :, ROOT_SLICE.start] = source_x
    target[:, 1, :, ROOT_SLICE.start] = target_x
    prediction[:, 1, :, ROOT_SLICE.start] = prediction_x
    prediction.requires_grad_()
    weights = _reaction_v2_test_weights(
        joint_distance=0.0,
        close_joint_vector=0.01,
        relative_root_radius=0.0,
        relative_root_bearing=0.0,
        partner_facing=0.0,
        soft_proximity=0.0,
        false_close=0.0,
    )
    bundle = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=torch.ones(1, 2, 1, dtype=torch.bool),
        timesteps=torch.tensor([timestep]),
        weights=weights,
    )
    return prediction, bundle


def test_reaction_p_only_is_gt_close_reconstruction_with_fine_gate() -> None:
    prediction, active = _reaction_p_only_bundle()
    term = active.terms["interaction_close_joint_vector"]
    assert float(term.raw) > 0.0
    torch.testing.assert_close(term.weighted, term.raw * 0.01)
    assert float(active.close_mask_fraction) == 1.0
    active.total.backward()
    assert prediction.grad is not None
    reactor_gradient = prediction.grad[0, 1, 0, ROOT_SLICE.start]
    assert float(reactor_gradient) > 0.0

    _, shifted_source = _reaction_p_only_bundle(source_x=0.05)
    torch.testing.assert_close(
        term.raw,
        shifted_source.terms["interaction_close_joint_vector"].raw,
    )

    _, gated = _reaction_p_only_bundle(timestep=0.2)
    assert float(gated.total) == 0.0
    assert float(gated.close_mask_fraction) == 0.0

    _, gt_far = _reaction_p_only_bundle(target_x=0.25, prediction_x=0.35)
    assert float(gt_far.total) == 0.0
    assert float(gt_far.close_mask_fraction) == 0.0


def test_reaction_p_only_has_constant_tail_gradient_and_quadratic_yaw_invariance() -> None:
    tail_gradients = []
    for prediction_x in (0.20, 0.30):
        prediction, bundle = _reaction_p_only_bundle(prediction_x=prediction_x)
        bundle.total.backward()
        assert prediction.grad is not None
        tail_gradients.append(
            prediction.grad[0, 1, 0, ROOT_SLICE.start].detach().clone()
        )
    torch.testing.assert_close(tail_gradients[0], tail_gradients[1])

    target = torch.zeros(1, 2, 1, DIM_HY273)
    prediction = target.clone()
    target[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    prediction[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    target[:, 1, :, ROOT_SLICE.start] = 0.10
    prediction[:, 1, :, ROOT_SLICE.start] = 0.12
    weights = _reaction_v2_test_weights(
        joint_distance=0.0,
        close_joint_vector=0.01,
        relative_root_radius=0.0,
        relative_root_bearing=0.0,
        partner_facing=0.0,
        soft_proximity=0.0,
        false_close=0.0,
    )
    kwargs = {
        "actor_valid": torch.ones(1, 2, 1, dtype=torch.bool),
        "timesteps": torch.tensor([0.8]),
        "weights": weights,
    }
    before = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        **kwargs,
    )
    angle = torch.tensor(1.1)
    after = compute_hy273_interaction_loss(
        prediction_physical=apply_yaw_rotation(prediction, angle),
        target_physical=apply_yaw_rotation(target, angle),
        **kwargs,
    )
    torch.testing.assert_close(
        before.terms["interaction_close_joint_vector"].raw,
        after.terms["interaction_close_joint_vector"].raw,
        rtol=1e-5,
        atol=1e-6,
    )


def test_reaction_v2_coarse_bearing_is_active_before_fine_geometry() -> None:
    target = torch.zeros(1, 2, 4, DIM_HY273)
    prediction = target.clone()
    target[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    prediction[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    target[:, 1, :, ROOT_SLICE.start] = 1.0
    prediction[:, 1, :, ROOT_SLICE.start + 2] = 1.0
    valid = torch.ones(1, 2, 4, dtype=torch.bool)

    bundle = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=valid,
        timesteps=torch.tensor([0.30]),
        weights=_reaction_v2_test_weights(),
    )
    assert float(bundle.coarse_active_scene_fraction) == 1.0
    assert float(bundle.fine_active_scene_fraction) == 0.0
    assert float(bundle.terms["interaction_relative_root_radius"].raw) == 0.0
    assert float(bundle.terms["interaction_relative_root_bearing"].raw) > 0.0
    assert float(bundle.terms["interaction_joint_distance"].raw) == 0.0


def test_reaction_v2_descriptors_are_invariant_to_shared_world_yaw() -> None:
    target = torch.zeros(1, 2, 3, DIM_HY273)
    prediction = target.clone()
    target[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    prediction[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    target[:, 1, :, ROOT_SLICE.start] = 1.0
    prediction[:, 1, :, ROOT_SLICE.start] = 0.7
    prediction[:, 1, :, ROOT_SLICE.start + 2] = 0.4
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    weights = _reaction_v2_test_weights()

    before = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=valid,
        timesteps=torch.tensor([0.8]),
        weights=weights,
    )
    angle = torch.tensor(1.1)
    after = compute_hy273_interaction_loss(
        prediction_physical=apply_yaw_rotation(prediction, angle),
        target_physical=apply_yaw_rotation(target, angle),
        actor_valid=valid,
        timesteps=torch.tensor([0.8]),
        weights=weights,
    )
    for name in (
        "interaction_relative_root_radius",
        "interaction_relative_root_bearing",
        "interaction_partner_facing",
        "interaction_joint_distance",
    ):
        torch.testing.assert_close(
            before.terms[name].raw,
            after.terms[name].raw,
            rtol=1e-5,
            atol=1e-6,
        )


def test_reaction_v2_false_close_does_not_penalize_true_contact() -> None:
    target_far = torch.zeros(1, 2, 2, DIM_HY273)
    prediction_false_close = target_far.clone()
    target_far[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    prediction_false_close[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    target_far[:, 1, :, ROOT_SLICE.start] = 1.0
    prediction_false_close.requires_grad_()
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    weights = _reaction_v2_test_weights(
        relative_root_radius=0.0,
        relative_root_bearing=0.0,
        partner_facing=0.0,
        joint_distance=0.0,
        soft_proximity=0.0,
    )

    false_close = compute_hy273_interaction_loss(
        prediction_physical=prediction_false_close,
        target_physical=target_far,
        actor_valid=valid,
        timesteps=torch.tensor([0.8]),
        weights=weights,
    )
    assert float(false_close.terms["interaction_false_close"].raw) > 0.0
    false_close.total.backward()
    assert prediction_false_close.grad is not None
    assert float(prediction_false_close.grad.norm()) > 0.0
    assert (
        float(prediction_false_close.grad[0, 1, 0, ROOT_SLICE.start]) < 0.0
    )

    target_contact = prediction_false_close.detach().clone()
    true_contact = compute_hy273_interaction_loss(
        prediction_physical=target_contact,
        target_physical=target_contact,
        actor_valid=valid,
        timesteps=torch.tensor([0.8]),
        weights=weights,
    )
    assert float(true_contact.terms["interaction_false_close"].raw) == 0.0


def test_reaction_adaptive_distance_matches_inverse_gt_weighted_huber() -> None:
    target = torch.zeros(1, 2, 1, DIM_HY273)
    prediction = target.clone()
    target[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    prediction[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    target[:, 1, :, ROOT_SLICE.start] = 1.5
    # Keep a nonzero separation: cdist has zero directional gradient at exact
    # overlap, which is handled by the separate directional false-close term.
    prediction[:, 1, :, ROOT_SLICE.start] = 0.2
    prediction.requires_grad_()
    weights = _reaction_v2_test_weights(
        relative_root_radius=0.0,
        relative_root_bearing=0.0,
        partner_facing=0.0,
        joint_distance=1.0,
        joint_distance_mode="adaptive_gt_inverse",
        adaptive_distance_eps_m=0.10,
        adaptive_distance_beta_m=0.05,
        close_joint_vector=0.0,
        soft_proximity=0.0,
        false_close=0.0,
        fine_min_flow_t=0.20,
    )
    bundle = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=torch.ones(1, 2, 1, dtype=torch.bool),
        timesteps=torch.tensor([0.30]),
        weights=weights,
    )
    with torch.no_grad():
        pred_joints = reconstruct_global_joints_from_features(prediction.float())
        target_joints = reconstruct_global_joints_from_features(target.float())
        pred_distance = torch.cdist(pred_joints[:, 0, 0], pred_joints[:, 1, 0])
        target_distance = torch.cdist(
            target_joints[:, 0, 0], target_joints[:, 1, 0]
        )
        adaptive_weight = 1.0 / (target_distance + 0.10)
        values = F.smooth_l1_loss(
            pred_distance,
            target_distance,
            reduction="none",
            beta=0.05,
        )
        expected = (values * adaptive_weight).sum() / adaptive_weight.sum()
    torch.testing.assert_close(
        bundle.terms["interaction_joint_distance"].raw, expected
    )
    assert float(bundle.distance_mask_fraction) == 1.0
    bundle.total.backward()
    assert prediction.grad is not None
    assert float(prediction.grad.norm()) > 0.0


def test_reaction_close_vector_tail_is_invariant_to_shared_world_yaw() -> None:
    target = torch.zeros(1, 2, 1, DIM_HY273)
    prediction = target.clone()
    target[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    prediction[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    target[:, 1, :, ROOT_SLICE.start] = 0.05
    prediction[:, 1, :, ROOT_SLICE.start] = 0.20
    weights = _reaction_v2_test_weights(
        relative_root_radius=0.0,
        relative_root_bearing=0.0,
        partner_facing=0.0,
        joint_distance=0.0,
        close_joint_vector=1.0,
        soft_proximity=0.0,
        false_close=0.0,
    )
    kwargs = {
        "actor_valid": torch.ones(1, 2, 1, dtype=torch.bool),
        "timesteps": torch.tensor([0.8]),
        "weights": weights,
    }
    before = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        **kwargs,
    )
    angle = torch.tensor(0.75)
    after = compute_hy273_interaction_loss(
        prediction_physical=apply_yaw_rotation(prediction, angle),
        target_physical=apply_yaw_rotation(target, angle),
        **kwargs,
    )
    torch.testing.assert_close(
        before.terms["interaction_close_joint_vector"].raw,
        after.terms["interaction_close_joint_vector"].raw,
        rtol=1e-5,
        atol=1e-6,
    )


def test_reaction_v2_diagnostic_ratios_keep_global_counts() -> None:
    target = torch.zeros(2, 2, 2, DIM_HY273)
    prediction = target.clone()
    target[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    prediction[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    valid = torch.ones(2, 2, 2, dtype=torch.bool)
    bundle = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=valid,
        timesteps=torch.tensor([0.8, 0.2]),
        weights=_reaction_v2_test_weights(),
    )
    numerator, denominator = bundle.diagnostic_ratios[
        "fine_active_scene_fraction"
    ]
    assert float(numerator) == 1.0
    assert float(denominator) == 2.0

    window = MetricWindow(torch.device("cpu"))
    window.add_ratio("reaction/fine_active_scene_fraction", 0.0, 1.0)
    window.add_ratio("reaction/fine_active_scene_fraction", 1.0, 1.0)
    result = window.flush(elapsed=1.0, world_size=1)
    assert result["reaction/fine_active_scene_fraction/raw"] == 0.5


def test_piecewise_task_schedule_exact_mix_and_resume() -> None:
    segments = [
        {"start": 0, "end": 100000, "t2m": 100, "edit": 0, "interaction": 0},
        {
            "start": 100000,
            "end": 200000,
            "t2m": 30,
            "edit": 35,
            "interaction": 35,
        },
    ]
    scheduler = PiecewiseTaskScheduler(segments)
    assert all(
        scheduler.choose(step) == TrainStream.HML_MIXED
        for step in range(100000)
    )
    counts = {stream: 0 for stream in TrainStream}
    for step in range(100000, 100100):
        counts[scheduler.choose(step)] += 1
    assert counts == {
        TrainStream.HML_MIXED: 30,
        TrainStream.MOTION_EDIT: 35,
        TrainStream.INTERACTION: 35,
    }
    state = scheduler.state_dict()
    resumed = PiecewiseTaskScheduler(segments)
    resumed.load_state_dict(state)
    expected = [scheduler.choose(step) for step in range(100100, 100200)]
    actual = [resumed.choose(step) for step in range(100100, 100200)]
    assert actual == expected


def test_piecewise_task_schedule_allows_only_exact_boundary_extension() -> None:
    base_segments = [
        {"start": 0, "end": 100000, "t2m": 100, "edit": 0, "reaction": 0},
        {
            "start": 100000,
            "end": 200000,
            "t2m": 30,
            "edit": 35,
            "reaction": 35,
        },
    ]
    extended_segments = [
        base_segments[0],
        {**base_segments[1], "end": 250000},
    ]
    base = PiecewiseTaskScheduler(base_segments)
    for step in range(200000):
        base.choose(step)
    state = base.state_dict()

    extended = PiecewiseTaskScheduler(extended_segments)
    with pytest.raises(ValueError, match="segments differ"):
        extended.load_state_dict(state)
    extended.load_state_dict(state, allow_segment_extension_at_step=200000)

    continuation_counts = {stream: 0 for stream in TrainStream}
    for step in range(200000, 250000):
        continuation_counts[extended.choose(step)] += 1
    assert continuation_counts == {
        TrainStream.HML_MIXED: 15000,
        TrainStream.MOTION_EDIT: 17500,
        TrainStream.INTERACTION: 17500,
    }
    assert extended.state.next_step == 250000
    assert extended.state.realized_hml == 145000
    assert extended.state.realized_edit == 52500
    assert extended.state.realized_interaction == 52500

    extended_state = extended.state_dict()
    extended_to_350k = PiecewiseTaskScheduler(
        [extended_segments[0], {**extended_segments[1], "end": 350000}]
    )
    extended_to_350k.load_state_dict(
        extended_state,
        allow_segment_extension_at_step=250000,
    )
    next_100 = [extended_to_350k.choose(step) for step in range(250000, 250100)]
    assert next_100.count(TrainStream.HML_MIXED) == 30
    assert next_100.count(TrainStream.MOTION_EDIT) == 35
    assert next_100.count(TrainStream.REACTION) == 35

    changed_mix = PiecewiseTaskScheduler(
        [
            base_segments[0],
            {
                "start": 100000,
                "end": 250000,
                "t2m": 40,
                "edit": 30,
                "reaction": 30,
            },
        ]
    )
    with pytest.raises(ValueError, match="segments differ"):
        changed_mix.load_state_dict(
            state,
            allow_segment_extension_at_step=200000,
        )


def test_fulltext_zero_to_200k_has_no_control_capability() -> None:
    draws = (0, (1 << 63), (1 << 64) - 1)
    for step in (0, 99_999, 100_000, 199_999):
        for draw in draws:
            assert (
                hml_capability_from_draw(
                    step,
                    draw,
                    KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
                )
                == CapabilityId.T2M
            )
            assert not edit_pattern_from_draw(
                draw,
                KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
            ).uses_control


def test_resume_config_requires_the_same_resolved_scientific_config() -> None:
    config = {
        "flow": {"timestep_mean": -0.8},
        "loss": {"contact": 0.01},
        "schedule": [{"start": 0, "end": 100000}],
    }
    validate_resume_config(config, config)
    changed = {
        **config,
        "loss": {"contact": 0.02},
    }
    with pytest.raises(ValueError, match="config changed"):
        validate_resume_config(config, changed)


def test_resume_config_allows_only_the_declared_same_mix_extension() -> None:
    checkpoint_config = {
        "model": {"hidden_dim": 1024},
        "loss": {"contact": 0.01},
        "schedule": {
            "segments": [
                {"start": 0, "end": 100000, "t2m": 100, "edit": 0, "reaction": 0},
                {
                    "start": 100000,
                    "end": 200000,
                    "t2m": 30,
                    "edit": 35,
                    "reaction": 35,
                },
            ]
        },
        "training": {"max_global_step": 200000, "stage_b_base_lr": 5e-5},
    }
    continued_config = {
        **checkpoint_config,
        "schedule": {
            "segments": [
                checkpoint_config["schedule"]["segments"][0],
                {
                    **checkpoint_config["schedule"]["segments"][1],
                    "end": 250000,
                },
            ]
        },
        "training": {
            **checkpoint_config["training"],
            "max_global_step": 250000,
        },
    }
    validate_resume_config(
        checkpoint_config,
        continued_config,
        allow_same_mix_extension_at_step=200000,
    )

    continued_to_350k = {
        **continued_config,
        "schedule": {
            "segments": [
                continued_config["schedule"]["segments"][0],
                {
                    **continued_config["schedule"]["segments"][1],
                    "end": 350000,
                },
            ]
        },
        "training": {
            **continued_config["training"],
            "max_global_step": 350000,
        },
    }
    validate_resume_config(
        continued_config,
        continued_to_350k,
        allow_same_mix_extension_at_step=250000,
    )

    changed_loss = {
        **continued_config,
        "loss": {"contact": 0.02},
    }
    with pytest.raises(ValueError, match="config changed"):
        validate_resume_config(
            checkpoint_config,
            changed_loss,
            allow_same_mix_extension_at_step=200000,
        )

    changed_mix = {
        **continued_config,
        "schedule": {
            "segments": [
                continued_config["schedule"]["segments"][0],
                {
                    **continued_config["schedule"]["segments"][1],
                    "t2m": 40,
                    "edit": 30,
                    "reaction": 30,
                },
            ]
        },
    }
    with pytest.raises(ValueError, match="config changed"):
        validate_resume_config(
            checkpoint_config,
            changed_mix,
            allow_same_mix_extension_at_step=200000,
        )


def test_reaction_v2_resume_transition_changes_only_reaction_loss() -> None:
    checkpoint_config = {
        "data": {"paired_task": "reaction"},
        "model": {"hidden_dim": 1024},
        "reaction_loss": {"joint_distance": 0.01, "min_flow_t": 0.20},
    }
    current_config = {
        **checkpoint_config,
        "reaction_loss": {
            "joint_distance": 0.01,
            "coarse_min_flow_t": 0.0,
            "fine_min_flow_t": 0.55,
        },
    }
    validate_resume_config(
        checkpoint_config,
        current_config,
        allow_reaction_v2_transition_at_step=100_000,
    )
    changed_model = {**current_config, "model": {"hidden_dim": 2048}}
    with pytest.raises(ValueError, match="config changed"):
        validate_resume_config(
            checkpoint_config,
            changed_model,
            allow_reaction_v2_transition_at_step=100_000,
        )


def test_reaction_v4_real_configs_change_only_reaction_loss() -> None:
    repository = Path(__file__).resolve().parents[1]
    parent_config, _ = load_config(
        repository / "configs/hy273_unified_fulltext_reaction_v1.yaml"
    )
    v4_config, _ = load_config(
        repository / "configs/hy273_unified_fulltext_reaction_v4_layout.yaml"
    )
    validate_resume_config(
        parent_config,
        v4_config,
        allow_reaction_v2_transition_at_step=100_000,
    )

    changed_output_dir = {
        **v4_config,
        "training": {
            **v4_config["training"],
            "output_dir": "/tmp/reaction-v4-different-output",
        },
    }
    with pytest.raises(ValueError, match="config changed"):
        validate_resume_config(
            parent_config,
            changed_output_dir,
            allow_reaction_v2_transition_at_step=100_000,
        )


def test_reaction_v5_real_config_changes_only_reaction_loss() -> None:
    repository = Path(__file__).resolve().parents[1]
    parent_config, _ = load_config(
        repository / "configs/hy273_unified_fulltext_reaction_v1.yaml"
    )
    v5_config, _ = load_config(
        repository / "configs/hy273_unified_fulltext_reaction_v5_event_layout.yaml"
    )
    validate_config(v5_config)
    validate_resume_config(
        parent_config,
        v5_config,
        allow_reaction_v2_transition_at_step=100_000,
    )
    weights = make_interaction_weights(v5_config)
    assert weights.relative_root == 0.0
    assert weights.relative_heading == 0.0
    assert weights.relative_root_bearing == pytest.approx(0.05)
    assert weights.partner_facing == pytest.approx(0.04)
    assert weights.scene_proximity == pytest.approx(0.008)
    assert weights.precontact_false_close == pytest.approx(0.008)
    assert weights.first_contact_cdf == pytest.approx(0.003)
    assert v5_config["reaction_loss"]["low_t_fraction"] == pytest.approx(0.30)
    assert v5_config["reaction_loss"]["low_t_max"] == pytest.approx(0.15)


def test_reaction_v5_1_real_config_only_adds_full_contact_loss() -> None:
    repository = Path(__file__).resolve().parents[1]
    v5_config, _ = load_config(
        repository / "configs/hy273_unified_fulltext_reaction_v5_event_layout.yaml"
    )
    v5_1_config, _ = load_config(
        repository
        / "configs/hy273_unified_fulltext_reaction_v5_1_full_contact.yaml"
    )
    validate_config(v5_1_config)
    assert {
        name: section for name, section in v5_config.items() if name != "reaction_loss"
    } == {
        name: section for name, section in v5_1_config.items() if name != "reaction_loss"
    }
    added = set(v5_1_config["reaction_loss"]) - set(v5_config["reaction_loss"])
    assert added == {
        "fk_contact_map_positive",
        "fk_contact_map_negative",
        "fk_contact_vector",
        "fk_contact_transition",
        "fk_contact_threshold_m",
        "fk_contact_temperature_m",
        "fk_contact_vector_scale_m",
        "fk_contact_transition_beta",
    }
    assert all(
        v5_1_config["reaction_loss"][name] == value
        for name, value in v5_config["reaction_loss"].items()
    )
    weights = make_interaction_weights(v5_1_config)
    assert weights.fk_contact_map_positive == pytest.approx(0.001)
    assert weights.fk_contact_map_negative == pytest.approx(0.005)
    assert weights.fk_contact_vector == pytest.approx(0.002)
    assert weights.fk_contact_transition == pytest.approx(0.003)


def test_reaction_v5_2_real_config_only_opens_all_timestep_gates() -> None:
    repository = Path(__file__).resolve().parents[1]
    v5_1_config, _ = load_config(
        repository
        / "configs/hy273_unified_fulltext_reaction_v5_1_full_contact.yaml"
    )
    v5_2_config, _ = load_config(
        repository
        / "configs/hy273_unified_fulltext_reaction_v5_2_all_t_fine.yaml"
    )
    validate_config(v5_2_config)
    expected = {
        **v5_1_config,
        "reaction_loss": {
            **v5_1_config["reaction_loss"],
            "min_flow_t": 0.0,
            "fine_min_flow_t": 0.0,
        },
    }
    assert v5_2_config == expected
    weights = make_interaction_weights(v5_2_config)
    assert weights.min_flow_t == 0.0
    assert weights.coarse_gate == 0.0
    assert weights.fine_gate == 0.0


def test_reaction_v5_2_continue300k_only_extends_the_frozen_mix() -> None:
    repository = Path(__file__).resolve().parents[1]
    v5_2_config, _ = load_config(
        repository
        / "configs/hy273_unified_fulltext_reaction_v5_2_all_t_fine.yaml"
    )
    extended_config, _ = load_config(
        repository
        / "configs/hy273_unified_fulltext_reaction_v5_2_all_t_fine_continue300k.yaml"
    )
    validate_config(extended_config)
    expected = {
        **v5_2_config,
        "schedule": {
            "segments": [
                {"start": 0, "end": 100_000, "t2m": 100, "edit": 0, "reaction": 0},
                {
                    "start": 100_000,
                    "end": 300_000,
                    "t2m": 30,
                    "edit": 35,
                    "reaction": 35,
                },
            ]
        },
        "training": {
            **v5_2_config["training"],
            "max_global_step": 300_000,
        },
    }
    assert extended_config == expected
    validate_resume_config(
        v5_2_config,
        extended_config,
        allow_same_mix_extension_at_step=200_000,
    )


def test_reaction_v5_2_launchers_preserve_training_and_evaluation_protocols() -> None:
    repository = Path(__file__).resolve().parents[1]
    continuation = (
        repository
        / "scripts/launch/train_hy273_unified_reaction_v5_2_continue300k_ddp8.sh"
    ).read_text()
    continue_250k = (
        repository
        / "scripts/launch/train_hy273_unified_reaction_stage_b_continue250k_ddp8.sh"
    ).read_text()
    continue_50k = (
        repository
        / "scripts/launch/train_hy273_unified_reaction_stage_b_continue50k_ddp8.sh"
    ).read_text()
    evaluation_200k = (
        repository
        / "scripts/launch/eval_hy273_unified_reaction_v5_2_200k_final.sh"
    ).read_text()
    evaluation_300k = (
        repository
        / "scripts/launch/eval_hy273_unified_reaction_v5_2_300k_final.sh"
    ).read_text()

    assert (
        'CHECKPOINT="${CURRENT_CHECKPOINT}" STOP_STEP=250000' in continuation
    )
    assert (
        "bash scripts/launch/"
        "train_hy273_unified_reaction_stage_b_continue250k_ddp8.sh"
        in continuation
    )
    assert '[[ "${STOP_STEP}" == "250000" ]]' in continue_250k
    assert (
        'CHECKPOINT="${CURRENT_CHECKPOINT}" STOP_STEP=300000' in continuation
    )
    assert (
        "bash scripts/launch/"
        "train_hy273_unified_reaction_stage_b_continue50k_ddp8.sh"
        in continuation
    )
    assert (
        '[[ "${STOP_STEP}" == "300000" || "${STOP_STEP}" == "350000" ]]'
        in continue_50k
    )

    for evaluation in (evaluation_200k, evaluation_300k):
        assert "EDIT_CFG=3.0" in evaluation
        assert "EVAL_PHASE=all" in evaluation
    assert "--training_contract reaction_v5_2_all_t_fine" in evaluation_200k
    assert "--expected_checkpoint_step 200000" in evaluation_200k
    assert "--training_contract same_run_dose_extension" in evaluation_300k
    assert "--baseline_checkpoint_step 200000" in evaluation_300k
    assert "--candidate_checkpoint_step 300000" in evaluation_300k


def test_reaction_v5_config_disables_full_contact_without_changing_old_total() -> None:
    repository = Path(__file__).resolve().parents[1]
    v5_config, _ = load_config(
        repository / "configs/hy273_unified_fulltext_reaction_v5_event_layout.yaml"
    )
    weights = make_interaction_weights(v5_config)
    target = _identity_interaction_motion(frames=3)
    prediction = target.clone()
    prediction[:, 1, :, ROOT_SLICE.start] = 0.4
    bundle = compute_hy273_interaction_loss(
        prediction_physical=prediction,
        target_physical=target,
        actor_valid=torch.ones(1, 2, 3, dtype=torch.bool),
        timesteps=torch.tensor([0.8]),
        weights=weights,
    )
    new_names = {
        "interaction_fk_contact_map_positive",
        "interaction_fk_contact_map_negative",
        "interaction_fk_contact_vector",
        "interaction_fk_contact_transition",
    }
    assert all(bundle.terms[name].weight == 0.0 for name in new_names)
    old_total = sum(
        (term.weighted for name, term in bundle.terms.items() if name not in new_names),
        prediction.sum() * 0.0,
    )
    torch.testing.assert_close(bundle.total, old_total, rtol=0.0, atol=0.0)


def test_reaction_v4_same_mix_extension_to_250k_changes_only_horizon() -> None:
    repository = Path(__file__).resolve().parents[1]
    base_config, _ = load_config(
        repository / "configs/hy273_unified_fulltext_reaction_v4_layout.yaml"
    )
    extended_config, _ = load_config(
        repository
        / "configs/hy273_unified_fulltext_reaction_v4_layout_continue250k.yaml"
    )
    validate_config(extended_config)
    validate_resume_config(
        base_config,
        extended_config,
        allow_same_mix_extension_at_step=200_000,
    )


def test_resume_requires_the_same_scientific_run() -> None:
    validate_resume_run_name("fulltext_scratch_001", "fulltext_scratch_001")
    with pytest.raises(ValueError, match="different run"):
        validate_resume_run_name("old_global_only", "fulltext_scratch_001")


def test_fulltext_phase_contract_is_scratch_then_exact_same_run_resume() -> None:
    validate_fulltext_phase_contract(
        FULLTEXT_STAGE_A_CONTRACT,
        has_resume=False,
        run_dir_exists=False,
        declared_stop_step=100_000,
        global_step=0,
    )
    validate_fulltext_phase_contract(
        FULLTEXT_STAGE_A_CONTRACT,
        has_resume=True,
        run_dir_exists=True,
        declared_stop_step=100_000,
        global_step=50_000,
    )
    validate_fulltext_phase_contract(
        FULLTEXT_STAGE_B_CONTRACT,
        has_resume=True,
        run_dir_exists=True,
        declared_stop_step=200_000,
        global_step=100_000,
    )
    validate_fulltext_phase_contract(
        FULLTEXT_STAGE_B_CONTRACT,
        has_resume=True,
        run_dir_exists=True,
        declared_stop_step=200_000,
        global_step=150_000,
    )
    validate_fulltext_phase_contract(
        FULLTEXT_REACTION_V2_STAGE_B_CONTRACT,
        has_resume=True,
        run_dir_exists=False,
        declared_stop_step=150_000,
        global_step=100_000,
    )
    validate_fulltext_phase_contract(
        FULLTEXT_STAGE_B_CONTINUE_CONTRACT,
        has_resume=True,
        run_dir_exists=True,
        declared_stop_step=250_000,
        global_step=200_000,
    )
    validate_fulltext_phase_contract(
        FULLTEXT_STAGE_B_CONTINUE_CONTRACT,
        has_resume=True,
        run_dir_exists=True,
        declared_stop_step=250_000,
        global_step=240_000,
    )
    validate_fulltext_phase_contract(
        FULLTEXT_STAGE_B_CONTINUE_CONTRACT,
        has_resume=True,
        run_dir_exists=True,
        declared_stop_step=300_000,
        global_step=250_000,
    )
    validate_fulltext_phase_contract(
        FULLTEXT_STAGE_B_CONTINUE_CONTRACT,
        has_resume=True,
        run_dir_exists=True,
        declared_stop_step=350_000,
        global_step=300_000,
    )
    with pytest.raises(ValueError, match=r"\[300000,350000\)"):
        validate_fulltext_phase_contract(
            FULLTEXT_STAGE_B_CONTINUE_CONTRACT,
            has_resume=True,
            run_dir_exists=True,
            declared_stop_step=350_000,
            global_step=250_000,
        )
    with pytest.raises(FileExistsError, match="fresh run"):
        validate_fulltext_phase_contract(
            FULLTEXT_STAGE_A_CONTRACT,
            has_resume=False,
            run_dir_exists=True,
            declared_stop_step=100_000,
        )
    with pytest.raises(ValueError, match="stop_step=100000"):
        validate_fulltext_phase_contract(
            FULLTEXT_STAGE_A_CONTRACT,
            has_resume=False,
            run_dir_exists=False,
            declared_stop_step=200_000,
        )
    with pytest.raises(ValueError, match=r"in \(0,100000\)"):
        validate_fulltext_phase_contract(
            FULLTEXT_STAGE_A_CONTRACT,
            has_resume=True,
            run_dir_exists=True,
            declared_stop_step=100_000,
            global_step=100_000,
        )
    with pytest.raises(ValueError, match=r"\[100000,200000\)"):
        validate_fulltext_phase_contract(
            FULLTEXT_STAGE_B_CONTRACT,
            has_resume=True,
            run_dir_exists=True,
            declared_stop_step=200_000,
            global_step=200_000,
        )
    with pytest.raises(ValueError, match=r"\[200000,250000\)"):
        validate_fulltext_phase_contract(
            FULLTEXT_STAGE_B_CONTINUE_CONTRACT,
            has_resume=True,
            run_dir_exists=True,
            declared_stop_step=250_000,
            global_step=250_000,
        )


def test_continue350k_config_changes_only_the_declared_training_horizon() -> None:
    config_250k, _ = load_config(
        "configs/hy273_unified_fulltext_reaction_v1_continue250k.yaml"
    )
    config_350k, _ = load_config(
        "configs/hy273_unified_fulltext_reaction_v1_continue350k.yaml"
    )
    validate_config(config_250k)
    validate_config(config_350k)

    normalized_250k = {
        **config_250k,
        "schedule": {
            **config_250k["schedule"],
            "segments": [
                config_250k["schedule"]["segments"][0],
                {
                    **config_250k["schedule"]["segments"][1],
                    "end": 350000,
                },
            ],
        },
        "training": {
            **config_250k["training"],
            "max_global_step": 350000,
        },
    }
    assert normalized_250k == config_350k


def test_unified_stage_a_path_matches_kencoder_base_at_initialization(
    tmp_path: Path,
) -> None:
    torch.manual_seed(23)
    base = _tiny_base_model(tmp_path / "base")
    torch.manual_seed(23)
    unified = _tiny_model(tmp_path / "unified")
    unified_state = unified.state_dict()
    for name, value in base.state_dict().items():
        if name == "source_context.task_embed.weight":
            torch.testing.assert_close(
                unified_state[name][: value.shape[0]],
                value,
                rtol=0.0,
                atol=0.0,
            )
            continue
        torch.testing.assert_close(
            unified_state[name],
            value,
            rtol=0.0,
            atol=0.0,
        )
    assert base.source_context.task_embed.num_embeddings == 3
    assert unified.source_context.task_embed.num_embeddings == 4

    frames = 4
    condition = make_absent_condition(
        batch_size=2,
        target_frames=frames,
        target_lengths=torch.tensor([frames, frames]),
    )
    model_input = torch.randn(2, frames, DIM_HY273 * 2)
    kwargs = {
        "t": torch.tensor([0.25, 0.75]),
        "c_dir": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "text": ["walk", "walk"],
        "length_mask": torch.ones(2, frames, dtype=torch.bool),
        "condition": condition,
    }
    base_output = base(model_input, **kwargs)
    unified_output = unified(
        model_input,
        **kwargs,
        task_id=condition.task_id,
    )
    torch.testing.assert_close(
        unified_output,
        base_output,
        rtol=0.0,
        atol=0.0,
    )


def test_unified_model_single_and_double_actor_forward_backward(
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    model = _tiny_model(tmp_path)
    frames = 4
    single_condition = make_absent_condition(
        batch_size=2,
        target_frames=frames,
        target_lengths=torch.tensor([frames, frames]),
    )
    single = model(
        torch.randn(2, frames, DIM_HY273 * 2),
        torch.tensor([0.2, 0.7]),
        c_dir=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        text=["walk", "walk"],
        length_mask=torch.ones(2, frames, dtype=torch.bool),
        condition=single_condition,
        task_id=single_condition.task_id,
    )
    assert single.shape == (2, frames, DIM_HY273)
    single.square().mean().backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    model.zero_grad(set_to_none=True)
    double_input = torch.randn(2, 2, frames, DIM_HY273 * 2)
    double_valid = torch.ones(2, 2, frames, dtype=torch.bool)
    interaction_task = torch.full((2,), int(TaskId.INTERACTION), dtype=torch.long)
    double = model(
        double_input,
        torch.tensor([0.3, 0.6]),
        c_dir=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        text=["two people shake hands"] * 2,
        length_mask=double_valid,
        task_id=interaction_task,
        text_profiles=(INTERACTION_TEXT_PROFILE,) * 2,
    )
    assert double.shape == (2, 2, frames, DIM_HY273)
    double.square().mean().backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_unified_model_rejects_unimplemented_ease_conditioning(
    tmp_path: Path,
) -> None:
    model = _tiny_model(tmp_path)
    condition = make_absent_condition(
        batch_size=1,
        target_frames=4,
        target_lengths=torch.tensor([4]),
    )
    condition = replace(
        condition,
        ease_physical=torch.ones(1, 6),
        ease_present=torch.ones(1, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="does not implement Ease"):
        model(
            torch.randn(1, 4, DIM_HY273 * 2),
            torch.tensor([0.5]),
            text=["walk"],
            length_mask=torch.ones(1, 4, dtype=torch.bool),
            condition=condition,
            task_id=condition.task_id,
        )


def test_full_unified_model_actor_swap_equivariance(tmp_path: Path) -> None:
    torch.manual_seed(19)
    model = _tiny_model(tmp_path).eval()
    for module in model.modules():
        if isinstance(module, BidirectionalActorExchange):
            torch.nn.init.normal_(module.up.weight, std=0.03)
            torch.nn.init.normal_(module.up.bias, std=0.03)
    frames = 4
    model_input = torch.randn(2, 2, frames, DIM_HY273 * 2)
    valid = torch.ones(2, 2, frames, dtype=torch.bool)
    task = torch.full((2,), int(TaskId.INTERACTION), dtype=torch.long)
    kwargs = {
        "t": torch.tensor([0.4, 0.8]),
        "c_dir": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "text": ["two people shake hands"] * 2,
        "length_mask": valid,
        "task_id": task,
        "text_profiles": (INTERACTION_TEXT_PROFILE,) * 2,
    }
    baseline = model(model_input, **kwargs)
    swapped = model(
        model_input.flip(1),
        **{**kwargs, "length_mask": valid.flip(1)},
    )
    torch.testing.assert_close(
        swapped,
        baseline.flip(1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_interaction_ode_sampling_is_actor_swap_equivariant(
    tmp_path: Path,
) -> None:
    torch.manual_seed(29)
    model = _tiny_model(tmp_path).eval()
    for module in model.modules():
        if isinstance(module, BidirectionalActorExchange):
            torch.nn.init.normal_(module.up.weight, std=0.03)
            torch.nn.init.normal_(module.up.bias, std=0.03)
    normalizer = HY273Normalizer(
        torch.zeros(DIM_HY273),
        torch.ones(DIM_HY273),
        normalize_contacts=True,
    )
    initial_noise = torch.randn(1, 2, 4, DIM_HY273)
    kwargs = {
        "texts": ["two people shake hands"],
        "target_lengths": torch.tensor([4]),
        "num_steps": 2,
        "cfg_scale": 3.5,
        "c_dir": torch.tensor([[1.0, 0.0]]),
    }
    baseline = sample_hy273_interaction_ode(
        model,
        normalizer,
        initial_noise=initial_noise,
        **kwargs,
    )
    swapped = sample_hy273_interaction_ode(
        model,
        normalizer,
        initial_noise=initial_noise.flip(1),
        **kwargs,
    )
    assert baseline.raw_motion.shape == (1, 2, 4, DIM_HY273)
    assert bool(torch.isfinite(baseline.raw_motion).all())
    assert baseline.branch_names == ("empty", "text")
    torch.testing.assert_close(
        swapped.raw_motion,
        baseline.raw_motion.flip(1),
        rtol=1e-5,
        atol=1e-5,
    )


def _edit_ratio_inputs(rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = (2, 5)[rank]
    basis = torch.zeros(1, frames, 269)
    scale = torch.arange(1, frames + 1, dtype=torch.float32).view(1, frames, 1)
    basis[..., :3] = scale * float(rank + 1)
    basis[..., 5:8] = 0.25 * scale.flip(1)
    valid = torch.ones(1, frames, dtype=torch.bool)
    hard_mask = torch.zeros(1, frames, DIM_HY273, dtype=torch.bool)
    if rank == 1:
        hard_mask[:, -1, :3] = True
    return basis, valid, hard_mask


def _edit_global_ratio_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_file: str,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        basis, valid, hard_mask = _edit_ratio_inputs(rank)
        parameter = torch.tensor(0.7, requires_grad=True)
        target = torch.zeros(1, basis.shape[1], DIM_HY273)
        weights = UnifiedEditLossWeights(
            target_x0_scale=0.05,
            hard_x0_scale=0.02,
            instruction_rank_scale=0.0,
        )
        bundle = compute_unified_edit_loss(
            correct_x0_hat_cont=parameter * basis,
            shuffled_x0_hat_cont=None,
            x0_target_norm=target,
            target_valid=valid,
            hard_mask=hard_mask,
            weights=weights,
            auxiliary_reduction="global_element_ratio",
        )
        globalize_ratio_terms(bundle.terms, world_size=world_size)
        bundle.refresh_ratio_terms(weights)
        bundle.total.backward()
        ddp_gradient = parameter.grad.detach().clone()
        dist.all_reduce(ddp_gradient, op=dist.ReduceOp.SUM)
        ddp_gradient /= float(world_size)

        if rank == 0:
            bases = []
            valid_rows = []
            hard_rows = []
            max_frames = 5
            for source_rank in range(world_size):
                local_basis, local_valid, local_hard = _edit_ratio_inputs(source_rank)
                pad = max_frames - local_basis.shape[1]
                bases.append(torch.nn.functional.pad(local_basis, (0, 0, 0, pad)))
                valid_rows.append(
                    torch.nn.functional.pad(local_valid, (0, pad), value=False)
                )
                hard_rows.append(
                    torch.nn.functional.pad(local_hard, (0, 0, 0, pad), value=False)
                )
            reference_parameter = torch.tensor(0.7, requires_grad=True)
            reference = compute_unified_edit_loss(
                correct_x0_hat_cont=reference_parameter * torch.cat(bases),
                shuffled_x0_hat_cont=None,
                x0_target_norm=torch.zeros(world_size, max_frames, DIM_HY273),
                target_valid=torch.cat(valid_rows),
                hard_mask=torch.cat(hard_rows),
                weights=weights,
                auxiliary_reduction="global_element_ratio",
            )
            reference.refresh_ratio_terms(weights)
            reference.total.backward()
            torch.save(
                {
                    "ddp_gradient": ddp_gradient,
                    "reference_gradient": reference_parameter.grad.detach(),
                },
                result_file,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed is unavailable",
)
def test_edit_aux_global_ratio_matches_two_rank_gloo(tmp_path: Path) -> None:
    init_file = tmp_path / "edit_ratio_init"
    result_file = tmp_path / "edit_ratio_result.pt"
    mp.spawn(
        _edit_global_ratio_worker,
        args=(2, str(init_file), str(result_file)),
        nprocs=2,
        join=True,
    )
    result = torch.load(result_file, map_location="cpu", weights_only=True)
    torch.testing.assert_close(
        result["ddp_gradient"],
        result["reference_gradient"],
        rtol=1e-6,
        atol=1e-7,
    )
