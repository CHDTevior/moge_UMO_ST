"""Build strict source/instruction derangements for MotionFix edit evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_hy273_multitask import canonical_sha, sha256_file


FORMAT = "motionfix_edit_counterfactual_manifest_v1"
TEXT_NORMALIZER = "unicode_preserving_whitespace_collapse_v1"


def normalize_text(value: str) -> str:
    return " ".join(str(value).split())


def stable_key(seed: int, left: str, right: str, kind: str) -> str:
    return hashlib.sha256(f"{FORMAT}:{seed}:{kind}:{left}:{right}".encode()).hexdigest()


def perfect_matching(
    rows: list[dict[str, Any]],
    *,
    valid,
    seed: int,
    kind: str,
    candidate_key=None,
) -> dict[str, str]:
    """Deterministic augmenting-path bipartite matching; fail if no derangement exists."""

    by_uid = {row["uid"]: row for row in rows}
    left_order = sorted(by_uid, key=lambda uid: stable_key(seed, uid, uid, kind + ":left"))
    neighbors = {
        uid: sorted(
            [candidate for candidate in by_uid if valid(by_uid[uid], by_uid[candidate])],
            key=lambda candidate: (
                ()
                if candidate_key is None
                else candidate_key(by_uid[uid], by_uid[candidate])
            )
            + (stable_key(seed, uid, candidate, kind),),
        )
        for uid in left_order
    }
    empty = [uid for uid, values in neighbors.items() if not values]
    if empty:
        raise RuntimeError(f"{kind} has rows without a legal donor: {empty[:8]}")
    donor_to_left: dict[str, str] = {}

    def augment(left: str, seen: set[str]) -> bool:
        for donor in neighbors[left]:
            if donor in seen:
                continue
            seen.add(donor)
            previous = donor_to_left.get(donor)
            if previous is None or augment(previous, seen):
                donor_to_left[donor] = left
                return True
        return False

    for left in left_order:
        if not augment(left, set()):
            raise RuntimeError(f"No complete deterministic {kind} matching at {left}")
    output = {left: donor for donor, left in donor_to_left.items()}
    if set(output) != set(by_uid) or len(set(output.values())) != len(rows):
        raise AssertionError(f"{kind} matching is not bijective")
    return output


def load_motionfix_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") == "motionfix_k273":
                rows.append(row)
    rows.sort(key=lambda row: row["uid"])
    return rows


def source_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    source = row["source_motion"]
    asset = source["k273_asset"]
    return (
        str(source["motion_uid"]),
        str(source["base_motion_id"]),
        str(asset["path"]),
        str(asset["sha256"]),
    )


def target_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    target = row["target_motion"]
    asset = target["k273_asset"]
    return (
        str(target["motion_uid"]),
        str(target["base_motion_id"]),
        str(asset["path"]),
        str(asset["sha256"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_manifest",
        default="/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/hy273_multitask_v1/test.jsonl",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary", required=True)
    args = parser.parse_args()
    manifest = Path(args.test_manifest).expanduser().resolve()
    rows = load_motionfix_rows(manifest)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * len(rows) + 100))
    by_uid = {row["uid"]: row for row in rows}
    source_ids = {row["uid"]: source_identity(row) for row in rows}
    target_ids = {row["uid"]: target_identity(row) for row in rows}
    source_lengths = {
        row["uid"]: int(row["source_motion"]["k273_asset"]["frames"])
        for row in rows
    }
    normalized_instructions = {
        row["uid"]: normalize_text(row["texts"][0]["value"]) for row in rows
    }

    def valid_source(left: dict[str, Any], donor: dict[str, Any]) -> bool:
        if left["uid"] == donor["uid"]:
            return False
        left_source = source_ids[left["uid"]]
        left_target = target_ids[left["uid"]]
        donor_source = source_ids[donor["uid"]]
        left_bucket = (source_lengths[left["uid"]] - 1) // 8
        donor_bucket = (source_lengths[donor["uid"]] - 1) // 8
        return (
            left_bucket == donor_bucket
            and all(a != b for a, b in zip(donor_source, left_source))
            and all(a != b for a, b in zip(donor_source, left_target))
        )

    def valid_instruction(left: dict[str, Any], donor: dict[str, Any]) -> bool:
        return (
            left["uid"] != donor["uid"]
            and normalized_instructions[left["uid"]]
            != normalized_instructions[donor["uid"]]
        )

    active_source_rows = list(rows)
    excluded_source_uids: set[str] = set()
    while True:
        active_uids = {row["uid"] for row in active_source_rows}
        newly_excluded = {
            row["uid"]
            for row in active_source_rows
            if not any(
                candidate["uid"] in active_uids and valid_source(row, candidate)
                for candidate in active_source_rows
            )
        }
        if not newly_excluded:
            break
        excluded_source_uids.update(newly_excluded)
        active_source_rows = [
            row for row in active_source_rows if row["uid"] not in newly_excluded
        ]
    source_match = perfect_matching(
        active_source_rows,
        valid=valid_source,
        seed=args.seed,
        kind="source_shuffle",
        candidate_key=lambda left, donor: (
            abs(
                source_lengths[left["uid"]] - source_lengths[donor["uid"]]
            ),
        ),
    )
    instruction_match = perfect_matching(
        rows, valid=valid_instruction, seed=args.seed, kind="instruction_shuffle"
    )
    output_rows = []
    for row in rows:
        source_donor = (
            None
            if row["uid"] in excluded_source_uids
            else by_uid[source_match[row["uid"]]]
        )
        instruction_donor = by_uid[instruction_match[row["uid"]]]
        source_ref = (
            None if source_donor is None else source_donor["source_motion"]["k273_asset"]
        )
        source_len = int(row["source_motion"]["k273_asset"]["frames"])
        donor_len = None if source_ref is None else int(source_ref["frames"])
        target_len = int(row["target_motion"]["k273_asset"]["frames"])
        record = {
            "format": FORMAT,
            "case_uid": row["uid"],
            "official_pair_id": row["pair"]["official_pair_id"],
            "requested_target_len": target_len,
            "source_len": source_len,
            "source_length_bucket_8": (source_len - 1) // 8,
            "source_shuffle": {
                "status": (
                    "failed_no_legal_same_bucket_donor"
                    if source_donor is None
                    else "matched"
                ),
                "donor_uid": None if source_donor is None else source_donor["uid"],
                "donor_source_motion_uid": (
                    None if source_donor is None else source_donor["source_motion"]["motion_uid"]
                ),
                "donor_source_base_motion_id": (
                    None if source_donor is None else source_donor["source_motion"]["base_motion_id"]
                ),
                "donor_source_path": None if source_ref is None else str(source_ref["path"]),
                "donor_source_sha256": None if source_ref is None else source_ref["sha256"],
                "donor_source_len": donor_len,
                "donor_length_bucket_8": (
                    None if donor_len is None else (donor_len - 1) // 8
                ),
                "target_to_source_time_map_policy": "normalized_progress_v1",
            },
            "instruction_shuffle": {
                "donor_uid": instruction_donor["uid"],
                "donor_text_id": instruction_donor["texts"][0]["text_id"],
                "donor_instruction": normalize_text(instruction_donor["texts"][0]["value"]),
                "donor_instruction_sha256": hashlib.sha256(
                    normalize_text(instruction_donor["texts"][0]["value"]).encode("utf-8")
                ).hexdigest(),
                "encoding_profile": "hytext_relative_edit_v1",
            },
        }
        if (
            source_donor is not None
            and not valid_source(row, source_donor)
        ) or not valid_instruction(row, instruction_donor):
            raise AssertionError("Counterfactual record violated its matching predicate")
        output_rows.append(record)
    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    summary = {
        "format": FORMAT,
        "status": "passed",
        "seed": int(args.seed),
        "test_manifest": str(manifest),
        "test_manifest_sha256": sha256_file(manifest),
        "case_count": len(output_rows),
        "source_shuffle_matched_cases": len(source_match),
        "source_shuffle_fail_closed_cases": len(excluded_source_uids),
        "source_shuffle_fail_closed_uids": sorted(excluded_source_uids),
        "source_shuffle_unique_donors": len(set(source_match.values())),
        "instruction_shuffle_unique_donors": len(set(instruction_match.values())),
        "text_normalizer": TEXT_NORMALIZER,
        "builder_code_sha256": sha256_file(Path(__file__).resolve()),
        "rows_sha256": canonical_sha(output_rows),
        "jsonl_sha256": sha256_file(output_path),
    }
    summary_path = Path(args.output_summary).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**summary, "output_jsonl": str(output_path), "output_summary": str(summary_path)}))


if __name__ == "__main__":
    main()
