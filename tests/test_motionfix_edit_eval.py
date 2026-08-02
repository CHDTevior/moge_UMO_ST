from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.raw_motion.hy273_motionfix_metrics import (
    PHYSICAL_TIMEWARP_PROTOCOL,
    _slerp_local_rotations,
    evaluate_motionfix_internal_case,
    physical_timewarp_hy201_to_k273,
)
import models.raw_motion.hy273_kimodo_benchmark as kimodo_benchmark
from models.raw_motion.hy273_kimodo_benchmark import SMPLX22_METRIC_JOINTS_PATH
from models.raw_motion.hy273_normalizer import HY273Normalizer
from models.raw_motion.hy273_slices import CONTACT_SLICE, GLOBAL_ROT_SLICE
from tools.eval_hy273_motionfix_edit import (
    ALL_SYSTEMS,
    CONTROL_SAFETY_METRICS,
    DEFAULT_COUNTERFACTUAL_MANIFEST,
    DEFAULT_COUNTERFACTUAL_MANIFEST_SHA256,
    EDIT_CONTROL_SUBTYPES,
    EQUAL_PROTOCOL,
    FULL_PROTOCOL,
    VAL_PROTOCOL,
    SYSTEM_EDIT_CONTROL,
    SYSTEM_INSTRUCTION_DROP,
    SYSTEM_INSTRUCTION_ONLY,
    SYSTEM_INSTRUCTION_SHUFFLE,
    SYSTEM_MODEL,
    SYSTEM_STANDALONE_CONTROL,
    SYSTEM_SOURCE_COPY,
    SYSTEM_SOURCE_SHUFFLE,
    EditCase,
    _aggregate_control_metrics,
    _bootstrap_mean_ci,
    _checkpoint_step,
    _checkpoint_preflight_identity,
    _compile_edit_control,
    _control_identity,
    _control_metrics_with_safety,
    _control_metric_direction,
    _expected_case_rows,
    _expected_sampling_protocol,
    _generate_model_case,
    _replay_record_metrics,
    _run_contract,
    _to_gauge,
    _physical_exact_clamp,
    _validate_exact_overwrite_contract,
    _paired_counterfactual_summary,
    _paired_edit_control_summary,
    _paired_data_root,
    _protocol_manifest_from_preflight,
    _load_train_seen_index,
    _seen_strata,
    _sampling_protocol_identity,
    _load_json_object_strict,
    _record_scientific_identity,
    _required_render_views,
    _resolve_model_inputs,
    _validate_scientific_identity,
    _validate_protocol_run_contract,
    _validate_aggregate_training_code_identity,
    _validate_checkpoint_code_identity,
    _validate_shard_ownership,
    _write_motion_output,
    build_parser,
    build_plan,
    load_counterfactual_rows,
    load_motionfix_rows,
    protocol_rows,
    canonical_sha,
    evaluation_code_identity,
)
from train_hy273_multitask import current_code_identity
import tools.render_hy273_motionfix_review as motionfix_renderer
from tools.render_hy273_motionfix_review import (
    ARTIFACT_INDEX_CONTRACT_FORMAT,
    ARTIFACT_INDEX_FORMAT,
    DECODER_PROTOCOL,
    RENDER_FORMAT,
    SELECTION_FORMAT,
    SELECTION_POLICY,
    TITLE_METRIC_PROTOCOL,
    _expected_render_parameters,
    _immutable_artifact_contract,
    _register_render_manifest,
    _required_render_views as renderer_required_render_views,
    _selected_prediction_metrics,
    _stratified_render_cases,
    _validate_artifact_index,
    _validate_renderer_identity,
    _view_cases,
    renderer_dependency_identity,
)


class _ZeroEditModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, model_in: torch.Tensor, **_kwargs) -> torch.Tensor:
        return torch.zeros_like(model_in[..., :273]) + self.anchor * 0.0


def test_motionfix_r13_expected_protocol_matches_actual_sampler() -> None:
    args = SimpleNamespace(
        num_steps=1,
        generate_text_cfg_scale=2.0,
        source_cfg_scale=1.0,
        edit_cfg_scale=1.0,
        control_cfg_scale=2.0,
        contact_init="random",
        contact_feedback="blend",
        generate_cfg_apply_contacts=True,
        cfg_apply_contacts=False,
        edit_source_baseline="learned",
    )
    case = EditCase(0, "pair", SYSTEM_MODEL, 17)
    source = torch.zeros(3, 273)
    observed = torch.zeros_like(source)
    mask = torch.zeros_like(source, dtype=torch.bool)
    normalizer = HY273Normalizer(
        torch.zeros(273), torch.ones(273), normalize_contacts=True
    )
    _, _, actual = _generate_model_case(
        model=_ZeroEditModel(),
        normalizer=normalizer,
        source=source,
        source_anchor=None,
        text="move faster",
        target_frames=3,
        phi=0.0,
        case=case,
        observed=observed,
        mask=mask,
        args=args,
    )
    expected = _expected_sampling_protocol(
        case, args, {}, unified_273_flow=True
    )
    assert _sampling_protocol_identity(actual) == expected
    assert actual["evaluator_contact_protocol_id"] == "hy273_unified_273_clean_flow_v1"


@pytest.mark.parametrize(
    ("contact_protocol", "unified"),
    [
        ("split_contact_logits_v1", False),
        ("unified_273_clean_flow_v1", True),
    ],
)
def test_motionfix_checkpoint_preflight_identity_closes_contact_protocol(
    tmp_path: Path,
    contact_protocol: str,
    unified: bool,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    checkpoint = {
        "format": "hy273_multitask_checkpoint_v2",
        "next_global_step": 123,
        "config": {"flow": {"contact_protocol": contact_protocol}},
    }

    identity = _checkpoint_preflight_identity(
        checkpoint_path,
        checkpoint,
        checkpoint_sha256="a" * 64,
    )

    assert identity["unified_273_flow"] is unified
    assert identity["step"] == 123
    assert identity["path"] == str(checkpoint_path)


MANIFEST = Path(
    "/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/"
    "hy273_multitask_v1/test.jsonl"
)
VAL_MANIFEST = MANIFEST.with_name("val.jsonl")


def test_checkpoint_step_prefers_next_global_step() -> None:
    assert _checkpoint_step({"next_global_step": 17, "step": 16}) == 17
    assert _checkpoint_step({"step": 16}) == 16


def test_unified_actor_asset_root_follows_paired_task(tmp_path: Path) -> None:
    interaction = tmp_path / "interaction"
    reaction = tmp_path / "reaction"
    assert _paired_data_root(
        {"data": {"paired_task": "interaction", "interaction_root": str(interaction)}}
    ) == ("interaction", interaction.resolve())
    assert _paired_data_root(
        {"data": {"paired_task": "reaction", "reaction_root": str(reaction)}}
    ) == ("reaction", reaction.resolve())


def test_seen_strata_tracks_exact_payloads_independently(tmp_path: Path) -> None:
    train_row = {
        "schema_version": "hy273_multitask_manifest_v1",
        "split": "train",
        "dataset": "motionfix_k273",
        "source_motion": {
            "base_motion_id": "train_source",
            "k273_asset": {"sha256": "source_payload"},
        },
        "target_motion": {
            "base_motion_id": "train_target",
            "k273_asset": {"sha256": "target_payload"},
        },
        "pair": {"official_pair_id": "train_pair"},
        "texts": [{"value": "raise the arm"}],
    }
    train_manifest = tmp_path / "train.jsonl"
    train_manifest.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
    index = _load_train_seen_index(train_manifest)

    eval_row = copy.deepcopy(train_row)
    eval_row["source_motion"]["base_motion_id"] = "different_source_base"
    eval_row["target_motion"]["base_motion_id"] = "different_target_base"
    eval_row["target_motion"]["k273_asset"]["sha256"] = "unseen_target_payload"
    eval_row["pair"]["official_pair_id"] = "test_pair"
    strata = _seen_strata(eval_row, index)

    assert strata["source_payload_seen"] is True
    assert strata["target_payload_seen"] is False
    assert strata["source_target_payload_seen_category"] == "source_only"
    assert strata["source_base_seen"] is False
    assert strata["target_base_seen"] is False


def test_validation_protocol_loads_only_the_frozen_val_split() -> None:
    rows = load_motionfix_rows(VAL_MANIFEST, VAL_PROTOCOL)
    assert len(rows) == 330
    assert len({row["pair"]["official_pair_id"] for row in rows}) == 330
    assert {row["split"] for row in rows} == {"val"}


def test_so3_slerp_midpoint_is_rotation_not_matrix_lerp() -> None:
    rotations = np.zeros((2, 1, 3, 3), dtype=np.float64)
    rotations[0, 0] = np.eye(3)
    rotations[1, 0] = np.diag([-1.0, 1.0, -1.0])
    result = _slerp_local_rotations(rotations, 3)
    np.testing.assert_allclose(result[0, 0], rotations[0, 0], atol=1e-7)
    np.testing.assert_allclose(result[-1, 0], rotations[-1, 0], atol=1e-7)
    np.testing.assert_allclose(result[1, 0].T @ result[1, 0], np.eye(3), atol=1e-7)
    assert np.linalg.det(result[1, 0]) > 0.999999


def _unequal_motionfix_row() -> dict:
    with MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") == "motionfix_k273" and row["pair"]["length_relation"] != "equal":
                return row
    raise AssertionError("Test manifest has no unequal MotionFix pair")


def test_real_hy201_physical_timewarp_reextracts_k273() -> None:
    row = _unequal_motionfix_row()
    source_hy = np.load(row["source_motion"]["hy201_asset"]["path"])
    target_frames = int(row["target_motion"]["k273_asset"]["frames"])
    output, info = physical_timewarp_hy201_to_k273(source_hy, target_frames)
    assert output.shape == (target_frames, 273)
    assert info["protocol"] == PHYSICAL_TIMEWARP_PROTOCOL
    assert info["source_frames"] != info["target_frames"]
    assert torch.isfinite(output).all()
    contacts = output[:, CONTACT_SLICE]
    assert bool(((contacts == 0.0) | (contacts == 1.0)).all())

    source_k273 = torch.from_numpy(
        np.load(row["source_motion"]["k273_asset"]["path"])
    ).float()
    torch.testing.assert_close(
        output[[0, -1], GLOBAL_ROT_SLICE],
        source_k273[[0, -1], GLOBAL_ROT_SLICE],
        atol=2e-5,
        rtol=2e-5,
    )


def test_internal_metrics_identity_has_zero_target_error() -> None:
    row = _unequal_motionfix_row()
    target = torch.from_numpy(np.load(row["target_motion"]["k273_asset"]["path"])).float()
    metrics = evaluate_motionfix_internal_case(target, target, target)
    assert metrics["global_joint_target_error_m"] == 0.0
    assert metrics["root_target_error_m"] == 0.0
    assert metrics["global_rotation_target_error_deg"] < 0.05
    assert metrics["contact_target_accuracy"] == 1.0
    assert metrics["changed_joint_entries"] == 0
    assert metrics["unchanged_region_source_error_m"] == 0.0
    assert metrics["global_rotation_source_error_deg"] < 0.05


def test_internal_changed_regions_use_hysteresis_and_temporal_dilation() -> None:
    rows = load_motionfix_rows(MANIFEST)
    target = torch.from_numpy(
        np.load(rows[0]["target_motion"]["k273_asset"]["path"])
    ).float()
    source = target.clone()
    source[3, 0] += 0.03
    source[10, 0] += 0.015

    metrics = evaluate_motionfix_internal_case(target, target, source)
    joint_count = 22
    assert metrics["changed_joint_entries"] == 5 * joint_count
    assert metrics["ambiguous_joint_entries"] == joint_count
    assert metrics["unchanged_joint_entries"] == (target.shape[0] - 6) * joint_count
    assert metrics["changed_region_target_error_m"] == 0.0
    assert metrics["unchanged_region_source_error_m"] == 0.0


def test_motionfix_protocols_are_disjointly_named_and_exactly_counted() -> None:
    rows = load_motionfix_rows(MANIFEST)
    counterfactual = load_counterfactual_rows(DEFAULT_COUNTERFACTUAL_MANIFEST, rows)
    equal = protocol_rows(rows, EQUAL_PROTOCOL)
    full = protocol_rows(rows, FULL_PROTOCOL)
    assert len(equal) == 952
    assert len(full) == 1013
    assert all(row["pair"]["length_relation"] == "equal" for row in equal)
    assert sum(row["pair"]["length_relation"] != "equal" for row in full) == 61

    plan = build_plan(
        equal,
        systems=ALL_SYSTEMS,
        seed=3407,
        counterfactual_rows=counterfactual,
    )
    assert len(plan) == 952 * len(ALL_SYSTEMS)
    pair_to_systems: dict[str, set[str]] = {}
    for case in plan:
        pair_to_systems.setdefault(case.pair_id, set()).add(case.system)
    assert len(pair_to_systems) == 952
    assert all(systems == set(ALL_SYSTEMS) for systems in pair_to_systems.values())

    full_plan = build_plan(
        full,
        systems=ALL_SYSTEMS,
        seed=3407,
        counterfactual_rows=counterfactual,
    )
    assert len(full_plan) == 1013 * len(ALL_SYSTEMS) - 1
    failed_systems = {
        case.system for case in full_plan if case.pair_id == "002794"
    }
    assert failed_systems == set(ALL_SYSTEMS) - {SYSTEM_SOURCE_SHUFFLE}

    controlled = [case for case in full_plan if case.system == SYSTEM_EDIT_CONTROL]
    assert len(controlled) == 1013
    assert {case.control_subtype for case in controlled} == set(EDIT_CONTROL_SUBTYPES)
    standalone = [
        case for case in full_plan if case.system == SYSTEM_STANDALONE_CONTROL
    ]
    assert len(standalone) == 1013
    assert [case.control_subtype for case in standalone] == [
        case.control_subtype for case in controlled
    ]
    seeds_by_pair: dict[str, set[int]] = {}
    for case in full_plan:
        seeds_by_pair.setdefault(case.pair_id, set()).add(case.sample_seed)
    assert all(len(seeds) == 1 for seeds in seeds_by_pair.values())


def test_counterfactual_inputs_resolve_to_pinned_source_and_instruction() -> None:
    rows = load_motionfix_rows(MANIFEST)
    counterfactual = load_counterfactual_rows(DEFAULT_COUNTERFACTUAL_MANIFEST, rows)
    row = rows[0]
    pair_id = str(row["pair"]["official_pair_id"])
    cf = counterfactual[pair_id]

    source_case = EditCase(0, pair_id, SYSTEM_SOURCE_SHUFFLE, 17)
    source_ref, source_text, source_provenance = _resolve_model_inputs(
        row, source_case, cf
    )
    assert source_ref["path"] == cf["source_shuffle"]["donor_source_path"]
    assert source_text == row["texts"][0]["value"]
    assert source_provenance["source_shuffled"] is True
    assert source_provenance["source_sha256"] == cf["source_shuffle"][
        "donor_source_sha256"
    ]

    instruction_case = EditCase(0, pair_id, SYSTEM_INSTRUCTION_SHUFFLE, 17)
    instruction_ref, instruction_text, instruction_provenance = _resolve_model_inputs(
        row, instruction_case, cf
    )
    assert instruction_ref == row["source_motion"]["k273_asset"]
    assert instruction_text == cf["instruction_shuffle"]["donor_instruction"]
    assert instruction_provenance["instruction_shuffled"] is True

    drop_case = EditCase(0, pair_id, SYSTEM_INSTRUCTION_DROP, 17)
    drop_source, dropped_text, drop_provenance = _resolve_model_inputs(
        row, drop_case, cf
    )
    assert drop_source == row["source_motion"]["k273_asset"]
    assert dropped_text == ""
    assert drop_provenance["instruction_dropped"] is True

    for absent_system in (
        SYSTEM_SOURCE_COPY,
        SYSTEM_INSTRUCTION_ONLY,
        SYSTEM_STANDALONE_CONTROL,
    ):
        absent_source, _, absent_provenance = _resolve_model_inputs(
            row, EditCase(0, pair_id, absent_system, 17), cf
        )
        assert absent_source is None
        assert absent_provenance["source_condition_present"] is False


def test_bootstrap_and_control_aggregation_are_deterministic_and_keep_zeroes() -> None:
    first = _bootstrap_mean_ci([1.0, 2.0, 3.0], seed=7, samples=200)
    second = _bootstrap_mean_ci([1.0, 2.0, 3.0], seed=7, samples=200)
    assert first == second
    assert first["count"] == 3
    assert first["mean"] == 2.0

    control_metrics = {
        pass_name: {"zero_error": 0.0, "nonzero_error": 2.0}
        for pass_name in (
            "generated_raw",
            "diagnostic_exact_clamp",
            "ground_truth",
        )
    }
    rows = _aggregate_control_metrics(
        [
            {
                "system": SYSTEM_EDIT_CONTROL,
                "control": {"subtype": EDIT_CONTROL_SUBTYPES[0]},
                "control_metrics": control_metrics,
            }
        ]
    )
    assert len(rows) == 2
    assert rows[0]["passes"]["generated_raw"]["metrics"]["zero_error"] == 0.0


def test_parser_pins_counterfactual_sha_and_bootstrap_default() -> None:
    args = build_parser().parse_args(
        ["--output_dir", "/tmp/out", "--protocol", EQUAL_PROTOCOL]
    )
    assert args.counterfactual_manifest_sha256 == DEFAULT_COUNTERFACTUAL_MANIFEST_SHA256
    assert args.bootstrap_samples == 10_000
    assert SYSTEM_MODEL in args.systems


def test_evaluation_identity_covers_transitive_local_inference_dependencies() -> None:
    identity = evaluation_code_identity()
    paths = {row["path"] for row in identity["files"]}
    assert {
        "models/__init__.py",
        "models/raw_motion/kimodo_like_flow_dit.py",
        "models/raw_motion/hy273_root_conditioning.py",
        "models/raw_motion/raw_flow_dit.py",
        "models/raw_motion/text_condition.py",
    } <= paths
    external = identity["external_dependencies"]
    hy201_paths = {
        row["path"] for row in external["hy201_to_kimodo273"]["files"]
    }
    kimodo_paths = {row["path"] for row in external["kimodo_runtime"]["files"]}
    assert "hy201_to_kimodo273/__init__.py" in hy201_paths
    assert "kimodo/model/__init__.py" in kimodo_paths


def test_checkpoint_code_identity_must_match_current_runtime() -> None:
    current = current_code_identity()
    assert _validate_checkpoint_code_identity({"code_identity": current}) == current
    changed = copy.deepcopy(current)
    changed["files"][next(iter(changed["files"]))] = "changed"
    with pytest.raises(RuntimeError, match="Checkpoint training-code identity differs"):
        _validate_checkpoint_code_identity({"code_identity": changed})

    preflight = {"checkpoint_training_code_identity": current}
    assert _validate_aggregate_training_code_identity(preflight) == current
    preflight["checkpoint_training_code_identity"] = changed
    with pytest.raises(RuntimeError, match="before aggregation"):
        _validate_aggregate_training_code_identity(preflight)


def test_control_safety_metrics_are_complete_and_finite() -> None:
    internal = {
        name: float(index + 1)
        for index, name in enumerate(CONTROL_SAFETY_METRICS)
    }
    merged = _control_metrics_with_safety({}, internal)
    assert set(merged) == set(CONTROL_SAFETY_METRICS)
    assert _control_metric_direction("prediction_jerk_mps3") == "lower"


def test_preflight_and_runtime_control_safety_schema_match_all_subtypes(
    tmp_path: Path,
) -> None:
    row = next(
        value
        for value in load_motionfix_rows(MANIFEST)
        if value["pair"]["length_relation"] == "equal"
    )
    pair_id = str(row["pair"]["official_pair_id"])
    plan = [
        EditCase(0, pair_id, SYSTEM_EDIT_CONTROL, 17, subtype)
        for subtype in EDIT_CONTROL_SUBTYPES
    ]
    args = build_parser().parse_args(
        [
            "--output_dir",
            str(tmp_path),
            "--protocol",
            FULL_PROTOCOL,
            "--num_shards",
            "1",
        ]
    )
    empty_train_index = {
        "base_motion_ids": set(),
        "motion_payload_sha256s": set(),
        "source_target_base_pairs": set(),
        "source_target_payload_pairs": set(),
        "instruction_target_payload_pairs": set(),
        "instruction_target_base_pairs": set(),
        "official_pair_ids": set(),
    }
    expected_rows = _expected_case_rows(
        [row],
        plan,
        args=args,
        counterfactual_rows=None,
        train_seen_index=empty_train_index,
        output_dir=tmp_path,
    )
    target_native = torch.from_numpy(
        np.load(row["target_motion"]["k273_asset"]["path"])
    ).float()
    target, _ = _to_gauge(
        target_native,
        expected_rows[0]["scientific_identity"]["gauge"]["output_gauge_phi"],
    )
    internal = evaluate_motionfix_internal_case(target, target, target)
    for case, expected in zip(plan, expected_rows, strict=True):
        constraint, evaluator = _compile_edit_control(
            target, case, max_sparse_keyframes=args.max_sparse_keyframes
        )
        runtime = _control_metrics_with_safety(
            evaluator(target, target, constraint), internal
        )
        frozen = expected["scientific_identity"]["control_metric_schema"]
        assert frozen is not None
        for pass_name in (
            "generated_raw",
            "diagnostic_exact_clamp",
            "ground_truth",
        ):
            assert set(frozen[pass_name]) == set(runtime)
            assert set(CONTROL_SAFETY_METRICS) <= set(frozen[pass_name])


def test_required_render_views_cover_raw_exact_editing_and_contact() -> None:
    views = _required_render_views()
    assert views == renderer_required_render_views()
    assert [view["view_id"] for view in views] == [
        "raw_editing6",
        "raw_contact12",
        "exact_editing6",
        "exact_contact12",
    ]
    assert all(view["output_format"] == "gif" for view in views)


def _artifact_ref(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_artifact_bundle(tmp_path: Path) -> tuple[Path, Path, dict]:
    rows = []
    for index, system in enumerate(motionfix_renderer.EDITING_VISUAL_SYSTEMS):
        rows.append(
            {
                "case_key": f"editing-{index}",
                "pair_id": f"e{index:02d}",
                "system": system,
                "control_subtype": None,
                "length_relation": "equal",
                "assets": {"target": f"target-{index}"},
                "aligned_reference_source": {"source": f"source-{index}"},
                "motion_output": {"output": f"output-{index}"},
            }
        )
    for system_index, system in enumerate(motionfix_renderer.CONTROL_SYSTEMS):
        for subtype_index, subtype in enumerate(
            motionfix_renderer.CONTACT_CONTROL_SUBTYPES
        ):
            suffix = f"{system_index}-{subtype_index}"
            rows.append(
                {
                    "case_key": f"contact-{suffix}",
                    "pair_id": f"c{suffix}",
                    "system": system,
                    "control_subtype": subtype,
                    "length_relation": "equal",
                    "assets": {"target": f"target-{suffix}"},
                    "aligned_reference_source": {"source": f"source-{suffix}"},
                    "motion_output": {"output": f"output-{suffix}"},
                }
            )

    all_cases_path = tmp_path / "visual_review_manifest.jsonl"
    all_cases_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    selected = motionfix_renderer._replay_frozen_selection(rows)
    renderer_identity = renderer_dependency_identity()
    selection = {
        "format": SELECTION_FORMAT,
        "selection_policy": SELECTION_POLICY,
        "all_cases_manifest": {
            "path": str(all_cases_path.resolve()),
            "sha256": _artifact_ref(all_cases_path)["sha256"],
            "count": len(rows),
        },
        "renderer": renderer_identity,
        "case_count": len(selected),
        "selected_case_keys_sha256": canonical_sha(
            [row["case_key"] for row in selected]
        ),
        "cases": selected,
    }
    selection_path = tmp_path / "visual_review_selection.json"
    _write_json(selection_path, selection)

    preflight_path = tmp_path / "preflight_manifest.json"
    preflight = {
        "code": {"sha256": "evaluation-code"},
        "checkpoint": {"sha256": "checkpoint"},
    }
    _write_json(preflight_path, preflight)
    protocol_path = tmp_path / "protocol_manifest.json"
    protocol = {
        "preflight_manifest": str(preflight_path.resolve()),
        "preflight_sha256": _artifact_ref(preflight_path)["sha256"],
    }
    _write_json(protocol_path, protocol)
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, {"protocol": protocol})
    expected_path = tmp_path / "expected_cases.jsonl"
    expected_path.write_text("{}\n", encoding="utf-8")
    checkpoint_stamp = tmp_path / "checkpoint_content_verification.json"
    _write_json(checkpoint_stamp, {"checkpoint": "checkpoint"})

    core = {
        "preflight_manifest": _artifact_ref(preflight_path),
        "protocol_manifest": _artifact_ref(protocol_path),
        "expected_case_manifest": _artifact_ref(expected_path),
        "checkpoint_content_verification": _artifact_ref(checkpoint_stamp),
        "summary": _artifact_ref(summary_path),
        "visual_review_manifest": _artifact_ref(all_cases_path),
        "visual_review_selection": _artifact_ref(selection_path),
        "shards": [],
    }
    required_views = _required_render_views()
    artifact_index = {
        "format": ARTIFACT_INDEX_FORMAT,
        "status": "awaiting_required_renders",
        "evaluation_code_identity_sha256": "evaluation-code",
        "checkpoint_sha256": "checkpoint",
        "protocol_manifest_sha256": core["protocol_manifest"]["sha256"],
        "summary_sha256": core["summary"]["sha256"],
        "core_artifacts": core,
        "required_render_views": required_views,
        "render_manifests": {},
        "render_manifests_sha256": canonical_sha({}),
    }
    immutable = _immutable_artifact_contract(artifact_index, required_views)
    assert immutable["format"] == ARTIFACT_INDEX_CONTRACT_FORMAT
    artifact_index.update(
        {key: value for key, value in immutable.items() if key != "format"}
    )
    artifact_index["immutable_contract"] = immutable
    artifact_index["immutable_contract_sha256"] = canonical_sha(immutable)
    index_path = tmp_path / "artifact_index.json"
    _write_json(index_path, artifact_index)
    return index_path, selection_path, selection


def _write_required_render_manifest(
    tmp_path: Path,
    selection_path: Path,
    selection: dict,
    view: dict,
) -> tuple[Path, dict]:
    output_dir = tmp_path / f"render_{view['view_id']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _view_cases(selection, view)
    parameters = _expected_render_parameters(selection, view)
    videos = []
    for case in cases:
        video_path = output_dir / f"{case['case_key']}.{view['output_format']}"
        video_path.write_bytes(f"gif:{view['view_id']}:{case['case_key']}".encode())
        videos.append(
            {
                "case_key": case["case_key"],
                **_artifact_ref(video_path),
                "title_metric_protocol": TITLE_METRIC_PROTOCOL,
                "title_metrics": {
                    "prediction_jerk_mps3": 1.0,
                    "foot_skate_ratio": 0.1,
                    "foot_contact_consistency": 0.9,
                },
                "inputs": {
                    "assets": case["assets"],
                    "aligned_reference_source": case["aligned_reference_source"],
                    "motion_output": case["motion_output"],
                },
            }
        )
    manifest = {
        "format": RENDER_FORMAT,
        "artifact_view_id": view["view_id"],
        "selection": {
            "path": str(selection_path.resolve()),
            "sha256": _artifact_ref(selection_path)["sha256"],
            "format": SELECTION_FORMAT,
        },
        "renderer": selection["renderer"],
        "parameters": parameters,
        "videos": videos,
    }
    manifest_path = output_dir / "render_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def test_artifact_index_becomes_validated_only_after_all_required_views(
    tmp_path: Path,
) -> None:
    index_path, selection_path, selection = _synthetic_artifact_bundle(tmp_path)
    views = _required_render_views()
    for index, view in enumerate(views):
        manifest_path, manifest = _write_required_render_manifest(
            tmp_path, selection_path, selection, view
        )
        updated = _register_render_manifest(
            artifact_index_path=index_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        expected_status = (
            "validated" if index == len(views) - 1 else "awaiting_required_renders"
        )
        assert updated["status"] == expected_status
        _validate_artifact_index(index_path, updated)


def test_artifact_index_rejects_reduced_required_view_set(tmp_path: Path) -> None:
    index_path, _, _ = _synthetic_artifact_bundle(tmp_path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["required_render_views"] = payload["required_render_views"][:1]
    with pytest.raises(RuntimeError, match="required render views changed"):
        _validate_artifact_index(index_path, payload)


def test_render_registration_rejects_incomplete_and_divergent_manifest(
    tmp_path: Path,
) -> None:
    index_path, selection_path, selection = _synthetic_artifact_bundle(tmp_path)
    view = _required_render_views()[0]
    manifest_path, manifest = _write_required_render_manifest(
        tmp_path, selection_path, selection, view
    )
    incomplete = {
        "format": RENDER_FORMAT,
        "artifact_view_id": view["view_id"],
        "videos": manifest["videos"],
    }
    _write_json(manifest_path, incomplete)
    with pytest.raises(RuntimeError, match="manifest schema changed"):
        _register_render_manifest(
            artifact_index_path=index_path,
            manifest_path=manifest_path,
            manifest=incomplete,
        )

    _write_json(manifest_path, manifest)
    changed_in_memory = copy.deepcopy(manifest)
    changed_in_memory["parameters"]["selected_case_keys"].reverse()
    with pytest.raises(RuntimeError, match="[Ii]n-memory and persisted"):
        _register_render_manifest(
            artifact_index_path=index_path,
            manifest_path=manifest_path,
            manifest=changed_in_memory,
        )


def test_run_contract_rejects_protocol_and_aggregate_argument_mutation() -> None:
    args = build_parser().parse_args(
        [
            "--output_dir",
            "/tmp/out",
            "--protocol",
            FULL_PROTOCOL,
            "--num_shards",
            "2",
            "--num_steps",
            "1",
        ]
    )
    contract = _run_contract(
        args,
        checkpoint_sha256="checkpoint",
        plan_sha256="plan",
        counterfactual_identity={"sha256": "counterfactual"},
    )
    preflight = {
        "run_contract": contract,
        "run_contract_sha256": canonical_sha(contract),
        "plan": {"pair_count": 19, "case_count": 151, "case_keys_sha256": "plan"},
        "checkpoint": {"sha256": "checkpoint"},
        "train_manifest": {"path": "/data/train.jsonl", "sha256": "train"},
        "counterfactual_source_fail_closed_pairs": ["002794"],
        "expected_case_manifest": {
            "path": "/output/expected.jsonl",
            "sha256": "expected",
            "count": 151,
        },
    }
    preflight_path = Path("/output/preflight.json").resolve()
    preflight_sha = "a" * 64
    protocol = _protocol_manifest_from_preflight(
        preflight,
        preflight_path=preflight_path,
        preflight_sha256=preflight_sha,
    )
    validation_kwargs = {
        "preflight_path": preflight_path,
        "preflight_sha256": preflight_sha,
    }
    _validate_protocol_run_contract(preflight, protocol, args, **validation_kwargs)

    changed_args = copy.deepcopy(args)
    changed_args.num_steps = 32
    with pytest.raises(RuntimeError, match="aggregate arguments"):
        _validate_protocol_run_contract(
            preflight, protocol, changed_args, **validation_kwargs
        )

    changed_protocol = copy.deepcopy(protocol)
    changed_protocol["checkpoint_sha256"] = "forged"
    with pytest.raises(RuntimeError, match="frozen run contract"):
        _validate_protocol_run_contract(
            preflight, changed_protocol, args, **validation_kwargs
        )

    for field, value in (
        ("train_manifest", {"path": "/forged", "sha256": "0" * 64}),
        ("counterfactual_source_fail_closed_pairs", ["002794", "forged"]),
        ("unknown_top_level", "forged"),
    ):
        changed_protocol = copy.deepcopy(protocol)
        changed_protocol[field] = value
        with pytest.raises(RuntimeError, match="protocol envelope"):
            _validate_protocol_run_contract(
                preflight, changed_protocol, args, **validation_kwargs
            )

    aliased_fields = []
    for field, value in protocol.items():
        alias = None
        if type(value) is bool:
            alias = int(value)
        elif type(value) is int:
            alias = float(value)
        elif type(value) is float and value.is_integer():
            alias = int(value)
        if alias is None:
            continue
        aliased_fields.append(field)
        changed_protocol = copy.deepcopy(protocol)
        changed_protocol[field] = alias
        with pytest.raises(RuntimeError):
            _validate_protocol_run_contract(
                preflight, changed_protocol, args, **validation_kwargs
            )
    assert aliased_fields


def test_strict_json_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"num_steps": 1, "num_steps": 1}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Duplicate JSON object key: num_steps"):
        _load_json_object_strict(path)


def test_renderer_identity_seals_transitive_metric_dependencies() -> None:
    identity = renderer_dependency_identity()
    paths = {row["path"] for row in identity["files"]}
    assert {
        "tools/render_hy273_motionfix_review.py",
        "models/__init__.py",
        "models/raw_motion/__init__.py",
        "models/raw_motion/hy273_motionfix_metrics.py",
        "models/raw_motion/hy273_kimodo_benchmark.py",
        "models/raw_motion/hy273_slices.py",
        "models/raw_motion/hy273_normalizer.py",
        "models/raw_motion/evidence_hash.py",
        "external_repos/kimodo/kimodo/assets/skeletons/smplx22/joints.p",
    } <= paths
    assert _validate_renderer_identity(identity) == identity
    assert identity["protocols"]["skeleton_resolution"] == (
        "project_root_explicit_path_v1"
    )
    assert identity["protocols"]["skeleton_asset"] == (
        "external_repos/kimodo/kimodo/assets/skeletons/smplx22/joints.p"
    )

    forged = copy.deepcopy(identity)
    forged["files"][1]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="renderer dependencies changed"):
        _validate_renderer_identity(forged)


def test_renderer_and_internal_metrics_resolve_sealed_skeleton_outside_repo_cwd(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[Path] = []
    original = kimodo_benchmark.load_smplx22_neutral_joints

    def recording_loader(*, path, device=None, dtype=torch.float32):
        calls.append(Path(path).expanduser().resolve())
        return original(path=path, device=device, dtype=dtype)

    monkeypatch.setattr(
        kimodo_benchmark,
        "load_smplx22_neutral_joints",
        recording_loader,
    )
    monkeypatch.chdir(tmp_path)
    row = _unequal_motionfix_row()
    target = np.load(row["target_motion"]["k273_asset"]["path"])[0:4]

    motionfix_renderer._joints(target)
    _selected_prediction_metrics(target, target, target)

    assert calls
    assert set(calls) == {SMPLX22_METRIC_JOINTS_PATH}
    assert SMPLX22_METRIC_JOINTS_PATH.is_absolute()


def test_renderer_title_metrics_follow_selected_prediction(monkeypatch) -> None:
    def fake_metrics(prediction, target, reference_source):
        del target, reference_source
        marker = float(prediction[0, 0])
        return {
            "prediction_jerk_mps3": marker,
            "foot_skate_ratio": marker + 1.0,
            "foot_contact_consistency": marker + 2.0,
        }

    monkeypatch.setattr(
        motionfix_renderer, "evaluate_motionfix_internal_case", fake_metrics
    )
    raw = np.zeros((2, 273), dtype=np.float32)
    exact = np.full((2, 273), 7.0, dtype=np.float32)
    target = np.zeros_like(raw)
    source = np.zeros_like(raw)
    assert _selected_prediction_metrics(raw, target, source)[
        "prediction_jerk_mps3"
    ] == 0.0
    exact_metrics = _selected_prediction_metrics(exact, target, source)
    assert exact_metrics == {
        "prediction_jerk_mps3": 7.0,
        "foot_skate_ratio": 8.0,
        "foot_contact_consistency": 9.0,
    }


def test_motion_output_writer_rejects_stale_tensor(tmp_path: Path) -> None:
    case = EditCase(0, "x", SYSTEM_MODEL, 17)
    raw = torch.zeros(3, 273)
    exact = raw.clone()
    observed = torch.zeros_like(raw)
    mask = torch.zeros_like(raw, dtype=torch.bool)
    identity = _write_motion_output(
        tmp_path,
        case,
        shard_id=0,
        prediction=raw,
        exact_prediction=exact,
        observed=observed,
        mask=mask,
        overwrite=False,
    )
    assert set(identity["arrays"]) == {"raw", "exact"}
    with pytest.raises(RuntimeError, match="Stale MotionFix output tensor raw"):
        _write_motion_output(
            tmp_path,
            case,
            shard_id=0,
            prediction=raw + 1.0,
            exact_prediction=exact + 1.0,
            observed=observed,
            mask=mask,
            overwrite=False,
        )


def test_exact_overwrite_contract_rejects_uncontrolled_exact_mutation() -> None:
    raw = torch.zeros(3, 273)
    exact = raw.clone()
    exact[1, 17] = 1.0
    with pytest.raises(RuntimeError, match="uncontrolled exact output differs from raw"):
        _validate_exact_overwrite_contract(
            raw,
            exact,
            observed=None,
            mask=None,
            label="negative uncontrolled case",
        )


def test_exact_overwrite_contract_rejects_controlled_unmasked_mutation() -> None:
    raw = torch.zeros(3, 273)
    observed = torch.ones_like(raw)
    mask = torch.zeros_like(raw, dtype=torch.bool)
    mask[1, 17] = True
    exact = torch.where(mask, observed, raw)
    exact[2, 18] = 1.0
    with pytest.raises(RuntimeError, match="uncontrolled_mismatches=1"):
        _validate_exact_overwrite_contract(
            raw,
            exact,
            observed=observed,
            mask=mask,
            label="negative controlled case",
        )


def test_motion_output_writer_rejects_invalid_exact_overwrite(tmp_path: Path) -> None:
    case = EditCase(0, "x", SYSTEM_EDIT_CONTROL, 17, EDIT_CONTROL_SUBTYPES[0])
    raw = torch.zeros(3, 273)
    observed = torch.ones_like(raw)
    mask = torch.zeros_like(raw, dtype=torch.bool)
    mask[1, 17] = True
    exact = torch.where(mask, observed, raw)
    exact[2, 18] = 1.0
    with pytest.raises(RuntimeError, match="uncontrolled_mismatches=1"):
        _write_motion_output(
            tmp_path,
            case,
            shard_id=0,
            prediction=raw,
            exact_prediction=exact,
            observed=observed,
            mask=mask,
            overwrite=False,
        )


def test_metric_replay_rejects_main_and_control_value_mutation() -> None:
    row = load_motionfix_rows(MANIFEST)[0]
    target_native = torch.from_numpy(
        np.load(row["target_motion"]["k273_asset"]["path"])
    ).float()
    phi = 0.37
    target, _ = _to_gauge(target_native, phi)
    case = EditCase(
        row_index=0,
        pair_id=str(row["pair"]["official_pair_id"]),
        system=SYSTEM_EDIT_CONTROL,
        sample_seed=17,
        control_subtype=EDIT_CONTROL_SUBTYPES[0],
    )
    constraint, evaluator = _compile_edit_control(
        target, case, max_sparse_keyframes=20
    )
    main_metrics = evaluate_motionfix_internal_case(target, target, target)
    control_metrics = {
        pass_name: _control_metrics_with_safety(
            evaluator(target, target, constraint), main_metrics
        )
        for pass_name in ("generated_raw", "diagnostic_exact_clamp", "ground_truth")
    }
    record = {
        "case_key": case.key,
        "row_index": case.row_index,
        "pair_id": case.pair_id,
        "system": case.system,
        "sample_seed": case.sample_seed,
        "control_subtype": case.control_subtype,
        "assets": {"target_k273": row["target_motion"]["k273_asset"]},
        "output_gauge_phi": phi,
        "length_relation": "equal",
        "metrics": main_metrics,
        "control": _control_identity(
            str(case.control_subtype),
            constraint,
            constraint.motion_mask,
            constraint.observed_motion,
        ),
        "control_metrics": control_metrics,
    }
    kwargs = {
        "raw": target,
        "exact": target,
        "aligned_source": target,
        "observed": constraint.observed_motion,
        "mask": constraint.motion_mask,
        "max_sparse_keyframes": 20,
    }
    _replay_record_metrics(record, **kwargs)

    changed_main = copy.deepcopy(record)
    changed_main["metrics"]["global_joint_target_error_m"] = 123.0
    with pytest.raises(RuntimeError, match="metrics values differ"):
        _replay_record_metrics(changed_main, **kwargs)

    changed_control = copy.deepcopy(record)
    metric_name = next(iter(changed_control["control_metrics"]["generated_raw"]))
    changed_control["control_metrics"]["generated_raw"][metric_name] = 123.0
    with pytest.raises(RuntimeError, match="control_metrics.generated_raw values differ"):
        _replay_record_metrics(changed_control, **kwargs)


def test_renderer_small_budget_is_system_and_contact_stratified() -> None:
    cases = [
        {"case_key": "instruction", "system": SYSTEM_INSTRUCTION_ONLY},
        {"case_key": "shuffle", "system": SYSTEM_SOURCE_SHUFFLE},
    ]
    for system in (SYSTEM_EDIT_CONTROL, SYSTEM_STANDALONE_CONTROL):
        for subtype in EDIT_CONTROL_SUBTYPES[-6:]:
            cases.append(
                {
                    "case_key": f"{system}:{subtype}",
                    "system": system,
                    "control_subtype": subtype,
                }
            )
    selected = _stratified_render_cases(cases, max_cases=14)
    assert {row["case_key"] for row in selected[:2]} == {"instruction", "shuffle"}
    contact_rows = [row for row in selected if row.get("control_subtype")]
    assert len(contact_rows) == 12
    assert {row["system"] for row in contact_rows} == {
        SYSTEM_EDIT_CONTROL,
        SYSTEM_STANDALONE_CONTROL,
    }


def test_physical_exact_clamp_is_bit_exact_for_all_control_subtypes() -> None:
    target = torch.from_numpy(
        np.load(load_motionfix_rows(MANIFEST)[0]["target_motion"]["k273_asset"]["path"])
    ).float()
    normalizer = HY273Normalizer(
        torch.linspace(-0.2, 0.3, 273),
        torch.linspace(0.4, 1.7, 273),
        variance_eps=1.0e-5,
    )
    exercised = set()
    for index, subtype in enumerate(EDIT_CONTROL_SUBTYPES):
        case = EditCase(
            row_index=0,
            pair_id="000004",
            system=SYSTEM_EDIT_CONTROL,
            sample_seed=100 + index,
            control_subtype=subtype,
        )
        constraint, evaluator = _compile_edit_control(
            target,
            case,
            max_sparse_keyframes=20,
        )
        roundtrip = normalizer.denormalize(
            normalizer.normalize(constraint.observed_motion.unsqueeze(0))
        )[0]
        exact = _physical_exact_clamp(
            roundtrip,
            constraint.observed_motion,
            constraint.motion_mask,
        )
        assert torch.equal(
            exact[constraint.motion_mask],
            constraint.observed_motion[constraint.motion_mask],
        )
        exercised.add(subtype)
        for metric_name in evaluator(target, target, constraint):
            assert _control_metric_direction(metric_name) in {
                "higher",
                "lower",
                "count",
            }
    assert exercised == set(EDIT_CONTROL_SUBTYPES)


def test_matched_edit_control_summary_is_direction_aware() -> None:
    shared = {
        "pair_id": "000004",
        "sample_seed": 17,
        "control_subtype": EDIT_CONTROL_SUBTYPES[0],
        "length_relation": "equal",
        "seen_strata": {"source_base_seen": True},
        "control": {
            "motion_mask_sha256": "mask",
            "observed_motion_sha256": "observed",
        },
        "sampling_protocol": {
            "initial_continuous_noise_sha256": "continuous",
            "initial_contact_noise_sha256": "contact",
        },
    }
    edit = {
        **shared,
        "system": SYSTEM_EDIT_CONTROL,
        "control_metrics": {
            "generated_raw": {
                "constraint_root2d_err": 1.0,
                "constraint_root2d_acc": 0.9,
            }
        },
    }
    standalone = {
        **shared,
        "system": SYSTEM_STANDALONE_CONTROL,
        "control_metrics": {
            "generated_raw": {
                "constraint_root2d_err": 2.0,
                "constraint_root2d_acc": 0.8,
            }
        },
    }
    summary = _paired_edit_control_summary(
        [edit, standalone], SimpleNamespace(seed=3407, bootstrap_samples=100)
    )
    all_row = next(
        row
        for row in summary["rows"]
        if row["level"] == "all" and row["name"] == "all"
    )
    assert all_row["metrics"]["constraint_root2d_err"]["direction"] == "lower"
    assert (
        all_row["metrics"]["constraint_root2d_err"]
        ["directional_edit_minus_standalone"]["mean"]
        == 1.0
    )
    assert all_row["metrics"]["constraint_root2d_acc"]["direction"] == "higher"


def _scientific_identity_record() -> dict:
    return {
        "case_key": "pair_x__source_instruction_model",
        "case_uid": "motionfix:x",
        "source_frames": 60,
        "target_frames": 60,
        "length_relation": "equal",
        "target_length_protocol": "equal_length_only",
        "frame_policy_id": "independent_sequence_frame_v1",
        "shared_world_frame": False,
        "assets": {
            "reference_source_k273": {"path": "/source", "sha256": "s"},
            "reference_source_hy201": None,
            "target_k273": {"path": "/target", "sha256": "t"},
            "conditioning_source_k273": {"path": "/source", "sha256": "s"},
        },
        "condition_provenance": {
            "source_condition_present": True,
            "source_role": "original_pair_source",
        },
        "output_gauge_phi": 0.25,
        "source_applied_yaw_delta": 0.1,
        "model_source_applied_yaw_delta": 0.1,
        "target_applied_yaw_delta": -0.2,
        "aligned_source_applied_yaw_delta": 0.1,
        "instruction": "raise the left arm",
        "model_instruction": "raise the left arm",
        "seen_strata": {"source_base_seen": True, "target_base_seen": False},
        "source_copy_protocol": {"protocol": "native_equal_length_k273_identity_v1"},
        "regional_metric_protocol": "equal_length_frozen_regions_v1",
        "sampling_protocol": {
            "route": "edit",
            "branch_names": ["empty", "source", "joint"],
            "text_cfg_scale": 2.0,
            "source_cfg_scale": 1.0,
            "initial_continuous_noise_sha256": "continuous",
            "initial_contact_noise_sha256": "contact",
        },
        "metrics": {"protocol": "metric_v1", "target_error": 1.0},
        "control": None,
        "control_metrics": None,
        "aligned_reference_source": {
            "format": "hy273_evaluator_aligned_source_npy_v1",
            "path": "/aligned",
            "sha256": "file",
            "shape": [60, 273],
            "tensor_sha256": "tensor",
        },
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("length_relation", "unequal"),
        ("seen_strata", {"source_base_seen": False, "target_base_seen": False}),
        ("regional_metric_protocol", "forged"),
        ("sampling_protocol", {"route": "generate"}),
        ("output_gauge_phi", 1.25),
    ],
)
def test_scientific_identity_rejects_material_metadata_mutation(
    field: str, value: object
) -> None:
    record = _scientific_identity_record()
    expected = _record_scientific_identity(record)
    mutated = copy.deepcopy(record)
    mutated[field] = value
    with pytest.raises(RuntimeError, match="scientific identity mismatch"):
        _validate_scientific_identity(mutated, expected)


def test_shard_ownership_rejects_foreign_record() -> None:
    record = {"case_key": "case", "shard_id": 1}
    expected = {"expected_shard_id": 0}
    with pytest.raises(RuntimeError, match="foreign case"):
        _validate_shard_ownership(record, expected, 0)


def test_counterfactual_pairing_rejects_same_seed_different_noise() -> None:
    def make(system: str) -> dict:
        return {
            "pair_id": "000004",
            "system": system,
            "sample_seed": 17,
            "length_relation": "equal",
            "seen_strata": {"source_base_seen": True},
            "sampling_protocol": {
                "initial_continuous_noise_sha256": "continuous",
                "initial_contact_noise_sha256": "contact",
            },
            "metrics": {
                "source_target_position_delta_m": 1.0,
                "global_joint_target_error_m": 0.5,
                "global_joint_source_error_m": 0.5,
                "global_rotation_target_error_deg": 2.0,
            },
        }

    records = [
        make(SYSTEM_MODEL),
        make(SYSTEM_SOURCE_SHUFFLE),
        make(SYSTEM_INSTRUCTION_SHUFFLE),
        make(SYSTEM_INSTRUCTION_DROP),
    ]
    records[1]["sampling_protocol"]["initial_contact_noise_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="actual initial noise"):
        _paired_counterfactual_summary(
            records,
            SimpleNamespace(seed=3407, bootstrap_samples=20),
            {"counterfactual_source_fail_closed_pairs": []},
        )


def test_counterfactual_summary_allows_pure_edit_system_subset() -> None:
    record = {
        "pair_id": "000004",
        "system": SYSTEM_MODEL,
        "sample_seed": 17,
        "length_relation": "equal",
        "seen_strata": {"source_base_seen": True},
        "sampling_protocol": {
            "initial_continuous_noise_sha256": "continuous",
            "initial_contact_noise_sha256": "contact",
        },
        "metrics": {
            "source_target_position_delta_m": 1.0,
            "global_joint_target_error_m": 0.4,
            "global_joint_source_error_m": 0.6,
            "global_rotation_target_error_deg": 2.0,
        },
    }
    summary = _paired_counterfactual_summary(
        [record],
        SimpleNamespace(seed=3407, bootstrap_samples=20),
        {
            "systems": [SYSTEM_SOURCE_COPY, SYSTEM_MODEL],
            "counterfactual_source_fail_closed_pairs": [],
        },
    )
    assert summary["requested_comparison_systems"] == []
    assert summary["subsets"]["all"]["comparisons"] == {}
    assert summary["subsets"]["all"]["edit_gain"]["count"] == 1


def test_paired_control_rejects_one_sided_missing_metric() -> None:
    shared = {
        "pair_id": "000004",
        "sample_seed": 17,
        "control_subtype": EDIT_CONTROL_SUBTYPES[0],
        "length_relation": "equal",
        "seen_strata": {"source_base_seen": True},
        "control": {
            "motion_mask_sha256": "mask",
            "observed_motion_sha256": "observed",
        },
        "sampling_protocol": {
            "initial_continuous_noise_sha256": "continuous",
            "initial_contact_noise_sha256": "contact",
        },
    }
    edit = {
        **shared,
        "system": SYSTEM_EDIT_CONTROL,
        "control_metrics": {
            "generated_raw": {
                "constraint_root2d_err": 1.0,
                "constraint_root2d_acc": 0.9,
            }
        },
    }
    standalone = {
        **shared,
        "system": SYSTEM_STANDALONE_CONTROL,
        "control_metrics": {
            "generated_raw": {"constraint_root2d_acc": 0.8}
        },
    }
    with pytest.raises(RuntimeError, match="metric schema differs"):
        _paired_edit_control_summary(
            [edit, standalone], SimpleNamespace(seed=3407, bootstrap_samples=20)
        )
