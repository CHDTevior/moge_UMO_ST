from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.raw_motion.hy273_slices import GLOBAL_ROT_SLICE, matrix_to_cont6d
from models.raw_motion.hy273_normalizer import HY273Normalizer
from tools.eval_hy273_t2m_nonregression import (
    ChunkCache,
    DEFAULT_GT272_ROOT,
    DEFAULT_K273_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_TEST_SPLIT,
    DEFAULT_TEXT_ROOT,
    MAX_EVAL_FRAMES,
    _initial_noise,
    _atomic_npz,
    _quality,
    _recover_partial_batches,
    _retrieval_rows,
    _sample_batch,
    _sampling_identity,
    checkpoint_kind,
    load_plan,
    sha256_file,
)
from models.raw_motion.hy273_t2m_eval import hy273_to_motionstreamer272
from train_hy273_multitask import CHECKPOINT_FORMAT


def test_retrieval_rows_uses_each_pairs_diagonal_rank() -> None:
    text = np.asarray(
        [[0.0, 0.0], [1.0, 2.0], [4.0, 1.0], [6.0, 5.0]],
        dtype=np.float64,
    )
    motion = np.asarray(
        [[2.0, 0.0], [1.0, 1.0], [5.0, 3.0], [7.0, 8.0]],
        dtype=np.float64,
    )

    top, matching = _retrieval_rows(text, motion, batch_size=4)

    distance = np.linalg.norm(text[:, None] - motion[None], axis=-1)
    expected_rank = np.asarray(
        [int(np.flatnonzero(np.argsort(distance[index]) == index)[0]) for index in range(4)]
    )
    expected_top = np.stack([expected_rank <= k for k in range(3)], axis=1)
    np.testing.assert_array_equal(top, expected_top)
    np.testing.assert_allclose(matching, np.linalg.norm(text - motion, axis=1))


def test_quality_decodes_k273_before_kimodo_metrics() -> None:
    motion = torch.zeros(5, 273)
    motion[:, 3] = 1.0
    identity = matrix_to_cont6d(torch.eye(3).expand(22, 3, 3)).reshape(-1)
    motion[:, GLOBAL_ROT_SLICE] = identity

    metrics = _quality(motion)

    assert {
        "foot_skate_from_height",
        "foot_skate_from_pred_contacts",
        "foot_skate_max_vel",
        "foot_skate_ratio",
        "foot_contact_consistency",
        "fk_jerk_mps3",
        "position_channel_jerk_mps3",
    } == set(metrics)
    assert all(np.isfinite(value) for value in metrics.values())


def test_checkpoint_kind_accepts_current_multitask_v2_and_rejects_unknown() -> None:
    assert checkpoint_kind({"format": CHECKPOINT_FORMAT}) == "multitask"
    with pytest.raises(RuntimeError, match="Unsupported"):
        checkpoint_kind({"format": "hy273_multitask_checkpoint_v999"})


def test_fixed_per_case_noise_is_batch_and_length_invariant() -> None:
    first = {"case_key": "a", "sample_seed": 11, "length": 10}
    second = {"case_key": "b", "sample_seed": 22, "length": 217}
    continuous_a, contacts_a, evidence_a = _initial_noise([first])
    continuous_ba, contacts_ba, evidence_ba = _initial_noise([second, first])
    assert continuous_a.shape == (1, MAX_EVAL_FRAMES, 269)
    assert contacts_a.shape == (1, MAX_EVAL_FRAMES, 4)
    assert torch.equal(continuous_a[0], continuous_ba[1])
    assert torch.equal(contacts_a[0], contacts_ba[1])
    assert evidence_a[0] == evidence_ba[1]


class _ZeroMultitaskModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, model_in: torch.Tensor, **_kwargs) -> torch.Tensor:
        return torch.zeros_like(model_in[..., :273]) + self.anchor * 0.0


@pytest.mark.parametrize("unified", [False, True])
def test_multitask_t2m_routes_noise_from_contact_protocol(unified: bool) -> None:
    args = SimpleNamespace(
        num_steps=1, cfg_scale=2.0, batch_size=1, weight_source="ema"
    )
    identity = _sampling_identity(
        "multitask", args, unified_273_flow=unified
    )
    assert identity["initial_noise"] == (
        "per_case_unified_gaussian_cpu_float32_fixed300_v3"
        if unified
        else "per_case_two_stream_cpu_float32_fixed300_v2"
    )
    normalizer = HY273Normalizer(
        torch.zeros(273), torch.ones(273), normalize_contacts=unified
    )
    output, evidence = _sample_batch(
        kind="multitask",
        model=_ZeroMultitaskModel(),
        normalizer=normalizer,
        runtime={"unified_273_flow": unified},
        cases=[{"length": 3, "caption": "walk", "sample_seed": 17}],
        args=args,
    )
    assert output.shape == (1, MAX_EVAL_FRAMES, 273)
    assert evidence[0]["initial_noise_protocol"] == identity["initial_noise"]


@pytest.mark.skipif(
    not Path(DEFAULT_MANIFEST).is_file() or not Path(DEFAULT_TEST_SPLIT).is_file(),
    reason="Canonical HY273/HumanML3D evaluation assets absent",
)
def test_canonical_plan_has_4042_cases_and_four_short_supplements() -> None:
    plan = load_plan(
        DEFAULT_MANIFEST,
        DEFAULT_GT272_ROOT,
        DEFAULT_TEST_SPLIT,
        DEFAULT_K273_ROOT,
        DEFAULT_TEXT_ROOT,
        seed=3407,
        audit_gt=False,
    )
    assert len(plan) == 4_042
    supplements = [row for row in plan if row["supplemental_short_case"]]
    assert {row["motion_id"] for row in supplements} == {
        "003790",
        "012941",
        "M003790",
        "M012941",
    }
    assert {row["length"] for row in supplements} == {10}


def _recovery_case(key: str, seed: int, length: int) -> dict:
    return {
        "case_key": key,
        "motion_id": key,
        "caption": f"caption {key}",
        "text_id": f"text:{key}",
        "length": length,
        "sample_seed": seed,
        "k273_path": f"/tmp/{key}.npy",
        "k273_sha256": "1" * 64,
        "gt272_path": f"/tmp/{key}_272.npy",
        "gt272_sha256": "2" * 64,
        "supplemental_short_case": False,
    }


def test_partial_batch_recovers_missing_json_record_from_atomic_chunk(tmp_path: Path) -> None:
    cases = [_recovery_case("a", 11, 5), _recovery_case("b", 22, 6)]
    raw = np.zeros((2, MAX_EVAL_FRAMES, 273), dtype=np.float32)
    raw[:, :, 3] = 1.0
    identity = matrix_to_cont6d(torch.eye(3).expand(22, 3, 3)).reshape(-1).numpy()
    raw[:, :, GLOBAL_ROT_SLICE] = identity
    generated272 = np.zeros((2, MAX_EVAL_FRAMES, 272), dtype=np.float32)
    for index, case in enumerate(cases):
        length = case["length"]
        generated272[index, :length] = hy273_to_motionstreamer272(raw[index, :length])
    chunk_path = tmp_path / "batch.npz"
    _atomic_npz(
        chunk_path,
        case_keys=np.asarray([case["case_key"] for case in cases]),
        lengths=np.asarray([case["length"] for case in cases], dtype=np.int64),
        generated272=generated272,
        generated_k273=raw,
    )
    chunk_sha = sha256_file(chunk_path)
    protocol_sha, preflight_sha, weight_sha = "a" * 64, "b" * 64, "c" * 64
    noise = _initial_noise([cases[0]])[2][0]
    first = {
        "status": "ok",
        **cases[0],
        "quality": _quality(torch.from_numpy(raw[0, : cases[0]["length"]])),
        **noise,
        "chunk_path": str(chunk_path),
        "chunk_sha256": chunk_sha,
        "chunk_row": 0,
        "chunk_frames": MAX_EVAL_FRAMES,
        "shard_id": 0,
        "protocol_manifest_sha256": protocol_sha,
        "preflight_sha256": preflight_sha,
        "selected_weight_state_sha256": weight_sha,
    }
    record_path = tmp_path / "shard.jsonl"
    record_path.write_text(json.dumps(first, sort_keys=True) + "\n", encoding="utf-8")
    existing = [first]
    by_key = {"a": first}
    _recover_partial_batches(
        record_path=record_path,
        shard_cases=cases,
        batch_size=2,
        existing=existing,
        by_key=by_key,
        chunk_cache=ChunkCache(),
        shard_id=0,
        protocol_sha=protocol_sha,
        preflight_sha=preflight_sha,
        selected_weight_sha=weight_sha,
    )
    assert set(by_key) == {"a", "b"}
    assert by_key["b"]["recovered_from_atomic_chunk"] is True
    assert by_key["b"]["chunk_row"] == 1
    assert len(record_path.read_text(encoding="utf-8").splitlines()) == 2
