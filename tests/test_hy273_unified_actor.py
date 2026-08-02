from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data.hy273_interaction_dataset import apply_shared_interaction_gauge
from data.hy273_unified_actor_batcher import PiecewiseTaskScheduler
from data.hy273_multitask_scheduler import (
    KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
    edit_pattern_from_draw,
    hml_capability_from_draw,
)
from models.raw_motion.hy273_actor_exchange import BidirectionalActorExchange
from models.raw_motion.hy273_interaction_losses import (
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
from models.raw_motion.hy273_normalizer import HY273Normalizer
from models.raw_motion.hy273_unified_edit_losses import (
    UnifiedEditLossWeights,
    compute_unified_edit_loss,
)
from models.raw_motion.hy273_slices import (
    DIM_HY273,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    LOCAL_ROOT_DIM,
    ROOT_SLICE,
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
    _masked_gradient_rms,
    globalize_ratio_terms,
    load_config,
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
