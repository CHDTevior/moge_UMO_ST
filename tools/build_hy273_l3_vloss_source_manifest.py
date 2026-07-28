#!/usr/bin/env python3
"""Build the immutable source manifest used by the L3 JiT-v-loss gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LOCAL_PATHS = (
    "train_hy273_raw_flow.py",
    "sample_hy273_raw.py",
    "data/kimodo273_datasets.py",
    "models/raw_motion/flow_schedule.py",
    "models/raw_motion/hy273_constraints.py",
    "models/raw_motion/hy273_normalizer.py",
    "models/raw_motion/hy273_slices.py",
    "models/raw_motion/raw_flow_dit.py",
    "models/raw_motion/hytext_cache.py",
    "models/raw_motion/text_condition.py",
    "models/codeflow/dit_blocks.py",
    "configs/raw_flow_hy273_hytext_l3_vloss_scratch.yaml",
    "scripts/launch/train_hy273_raw_flow_l3_vloss_pilot_ddp4.sh",
    "scripts/launch/train_hy273_raw_flow_l3_vloss_scratch_ddp4.sh",
    "scripts/launch/train_hy273_raw_flow_stage1_x0_hytext_ddp8.sh",
    "scripts/launch/train_hy273_raw_flow_ddp8.sh",
    "tools/build_hy273_l3_vloss_source_manifest.py",
    "tools/calibrate_hy273_l2_l3_losses.py",
    "tools/verify_hy273_l3_vloss_preflight.py",
    "tests/test_raw_flow_model.py",
    "tests/test_raw_flow_sampling.py",
    "tests/test_hy273_constraints.py",
    "run_logs/hy273_l3_vloss_motion_payload.sha256",
    "run_logs/hy273_l3_vloss_text_payload.sha256",
    "run_logs/hy273_l3_vloss_hytext_payload.sha256",
    "run_logs/hy273_l3_vloss_jitpm08ps08_calibration_train_t_seed3407_n4096.json",
    "run_logs/hy273_l3_vloss_jitpm08ps08_calibration_bins_seed3407_n16_alpha0939702.json",
    "external_repos/kimodo/kimodo/assets/skeletons/smplx22/joints.p",
)

PAYLOAD_SPECS = (
    (
        "run_logs/hy273_l3_vloss_motion_payload.sha256",
        Path(
            "/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22/motion_data"
        ),
        "*.npy",
    ),
    (
        "run_logs/hy273_l3_vloss_text_payload.sha256",
        Path("/mnt/afs/mogo_base/datasets/HumanML3D/texts"),
        "*.txt",
    ),
    (
        "run_logs/hy273_l3_vloss_hytext_payload.sha256",
        Path(
            "/mnt/afs/mogo_base/datasets/HumanML3D/hytext_qwen3_clipL_mlen128/shards"
        ),
        "**/*.npy",
    ),
)

EXTERNAL_PATHS = (
    "/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22/split_existing/train.txt",
    "/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22/manifest.jsonl",
    "/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22/stats/Mean.npy",
    "/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22/stats/Std.npy",
    "/mnt/afs/mogo_base/datasets/HumanML3D/hytext_qwen3_clipL_mlen128/index.json",
    "/mnt/afs/mogo_base/datasets/HumanML3D/hytext_qwen3_clipL_mlen128/manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("run_logs/hy273_l3_vloss_source_manifest.sha256"),
    )
    parser.add_argument(
        "--reuse_payload_manifests",
        action="store_true",
        help=(
            "Reuse existing payload manifests when only source files changed. "
            "Pilot/production verification still re-hashes every payload file."
        ),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    payload_summaries: dict[str, dict[str, object]] = {}
    for output_name, payload_root, pattern in PAYLOAD_SPECS:
        payload_output = repo_root / output_name
        if args.reuse_payload_manifests:
            if not payload_output.is_file():
                raise FileNotFoundError(
                    f"Cannot reuse missing payload manifest: {payload_output}"
                )
            payload_count = sum(
                bool(line.strip())
                for line in payload_output.read_text(encoding="utf-8").splitlines()
            )
        else:
            payload_files = sorted(
                path for path in payload_root.glob(pattern) if path.is_file()
            )
            if not payload_files:
                raise RuntimeError(
                    f"No payload files matched {pattern!r} under {payload_root}"
                )
            payload_output.parent.mkdir(parents=True, exist_ok=True)
            payload_output.write_text(
                "".join(f"{sha256_file(path)}  {path}\n" for path in payload_files),
                encoding="utf-8",
            )
            payload_count = len(payload_files)
        payload_summaries[output_name] = {
            "files": payload_count,
            "manifest_sha256": sha256_file(payload_output),
        }
    rows: list[str] = []
    for filename in LOCAL_PATHS:
        rows.append(f"{sha256_file(repo_root / filename)}  {filename}")
    for filename in EXTERNAL_PATHS:
        rows.append(f"{sha256_file(Path(filename))}  {filename}")
    output = args.output if args.output.is_absolute() else repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "entries": len(rows),
                "manifest_sha256": sha256_file(output),
                "payload_manifests": payload_summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
