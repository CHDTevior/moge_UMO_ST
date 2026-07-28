from __future__ import annotations

import json

from tools.monitor_hy273_training import assess, expected_gpu_indices


def _metrics() -> list[dict[str, float]]:
    return [{"step": 100.0, "loss": 0.1, "flow": 0.08}]


def _gpu_rows() -> list[dict[str, int]]:
    return [
        {
            "index": index,
            "memory_used_mb": 21000 if index < 4 else 0,
            "memory_total_mb": 81920,
            "util_gpu_pct": 80 if index < 4 else 0,
        }
        for index in range(8)
    ]


def test_assess_accepts_busy_expected_ddp_subset() -> None:
    status, reasons = assess(_metrics(), _gpu_rows(), expected_gpus=[0, 1, 2, 3])
    assert status == "ok"
    assert reasons == []


def test_assess_rejects_idle_expected_gpu() -> None:
    status, reasons = assess(_metrics(), _gpu_rows(), expected_gpus=[0, 1, 2, 3, 4])
    assert status == "bad"
    assert reasons == ["expected GPUs have <=1GB allocated: [4]"]


def test_expected_gpu_indices_uses_trace_contract(tmp_path) -> None:
    (tmp_path / "trace_contract.json").write_text(
        json.dumps({"cuda_visible_devices": "4,5,6,7"}), encoding="utf-8"
    )
    assert expected_gpu_indices(tmp_path, _gpu_rows()) == [4, 5, 6, 7]
