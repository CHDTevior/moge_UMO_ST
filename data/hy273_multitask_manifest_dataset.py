"""Manifest-backed HML/MotionFix materialization for HY273 multitask training."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from data.hy273_multitask_scheduler import (
    HIGH_LEVEL_SCHEDULE_VERSION,
    EditConditionPattern,
    SamplePlan,
    bernoulli_from_draw,
    ease_present_from_draw,
    edit_pattern_from_draw,
    hml_capability_from_draw,
    sample_key_u64,
)
from models.raw_motion.hy273_multitask_condition import (
    CapabilityId,
    ConditionBatch,
    FramePolicy,
    SourceRole,
    TargetOp,
    TaskId,
    TrainStream,
)
from models.raw_motion.hy273_ease import ease_from_k273
from models.raw_motion.hy273_normalizer import apply_yaw_rotation, root_origin_shift
from models.raw_motion.hy273_slices import CONTACT_SLICE, DIM_HY273, HEADING_SLICE


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _u64_to_phi(value: int) -> float:
    unit = (int(value) + 0.5) / float(1 << 64)
    return (2.0 * unit - 1.0) * math.pi


def _transform_to_gauge(motion: torch.Tensor, phi: float) -> tuple[torch.Tensor, float]:
    if motion.ndim != 2 or motion.shape[-1] != DIM_HY273:
        raise ValueError(f"Expected physical K273 [T,{DIM_HY273}], got {motion.shape}")
    shifted = root_origin_shift(motion)
    heading = shifted[0, HEADING_SLICE]
    current = torch.atan2(heading[1], heading[0])
    delta = torch.as_tensor(phi, dtype=shifted.dtype) - current
    return apply_yaw_rotation(shifted, delta), float(delta.item())


class HY273MultitaskManifestDataset:
    """One logical stream from a unified split manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        stream: TrainStream,
        *,
        verify_payload_hash: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.stream = TrainStream(stream)
        expected_dataset = (
            "humanml3d_k273"
            if self.stream == TrainStream.HML_MIXED
            else "motionfix_k273"
        )
        rows: list[dict[str, Any]] = []
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema_version") != "hy273_multitask_manifest_v1":
                    raise ValueError(
                        f"Manifest schema mismatch at {self.manifest_path}:{line_number}"
                    )
                if row.get("dataset") == expected_dataset:
                    rows.append(row)
        if not rows:
            raise RuntimeError(
                f"No {expected_dataset} rows in unified manifest {self.manifest_path}"
            )
        self.rows = rows
        self.verify_payload_hash = bool(verify_payload_hash)
        self._verified_paths: set[str] = set()

    def __len__(self) -> int:
        return len(self.rows)

    def uid(self, index: int) -> str:
        return str(self.rows[index]["uid"])

    def caption_count(self, index: int) -> int:
        return len(self.rows[index]["texts"])

    @staticmethod
    def _bucket(length: int) -> int:
        return max(0, (int(length) - 1) // 32)

    def bucket_key(self, index: int) -> tuple[int, int]:
        """Return the row-level pre-sort hint.

        HML rows can resolve to a caption-specific segment, so this hint must not
        be treated as the final batch bucket. ``plan_bucket_key`` is authoritative
        after the stateless caption choice has been made.
        """
        row = self.rows[index]
        target_len = int(row["target_motion"]["k273_asset"]["frames"])
        source = row.get("source_motion")
        source_len = 0 if source is None else int(source["k273_asset"]["frames"])
        return self._bucket(target_len), self._bucket(source_len)

    @property
    def bucket_keys(self) -> tuple[tuple[int, int], ...]:
        return tuple(self.bucket_key(index) for index in range(len(self)))

    def plan_lengths(self, plan: SamplePlan) -> tuple[int, int]:
        """Resolve target/source lengths without loading either NPY payload."""

        if plan.train_stream_id != self.stream:
            raise ValueError("SamplePlan stream does not match dataset")
        row = self.rows[int(plan.row_index)]
        if str(row["uid"]) != plan.uid:
            raise ValueError("SamplePlan UID does not match row index")
        if self.stream == TrainStream.HML_MIXED:
            if plan.caption_index is None:
                raise ValueError("HML SamplePlan requires caption_index")
            target_ref = row["texts"][int(plan.caption_index)]["target_k273_asset"]
            return int(target_ref["frames"]), 0
        if plan.caption_index is not None:
            raise ValueError("MotionFix SamplePlan cannot carry caption_index")
        if plan.edit_pattern is None:
            raise ValueError("MotionFix SamplePlan requires an edit condition pattern")
        source_len = int(row["source_motion"]["k273_asset"]["frames"])
        if plan.edit_pattern == EditConditionPattern.SOURCE_IDENTITY:
            return source_len, source_len
        return int(row["target_motion"]["k273_asset"]["frames"]), source_len

    def plan_bucket_key(self, plan: SamplePlan) -> tuple[int, int]:
        target_len, source_len = self.plan_lengths(plan)
        return self._bucket(target_len), self._bucket(source_len)

    def _load_asset(self, ref: dict[str, Any]) -> torch.Tensor:
        path = Path(ref["path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest asset is missing: {path}")
        if self.verify_payload_hash and str(path) not in self._verified_paths:
            expected = str(ref.get("sha256", ""))
            if not expected or expected == "not-computed":
                raise ValueError(f"Manifest has no usable payload SHA for {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(
                    f"Payload SHA mismatch for {path}: expected={expected}, actual={actual}"
                )
            self._verified_paths.add(str(path))
        array = np.load(path).astype(np.float32, copy=False)
        expected_shape = (int(ref["frames"]), int(ref["feature_dim"]))
        if array.shape != expected_shape or array.shape[1] != DIM_HY273:
            raise ValueError(f"Asset shape mismatch for {path}: {array.shape}/{expected_shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"Asset contains non-finite values: {path}")
        contacts = array[:, CONTACT_SLICE]
        if not np.logical_or(contacts == 0.0, contacts == 1.0).all():
            raise ValueError(f"Asset contacts are not exact binary 0/1: {path}")
        return torch.from_numpy(array.copy())

    def materialize(self, plan: SamplePlan) -> dict[str, Any]:
        if plan.train_stream_id != self.stream:
            raise ValueError("SamplePlan stream does not match dataset")
        row = self.rows[int(plan.row_index)]
        if str(row["uid"]) != plan.uid:
            raise ValueError("SamplePlan UID does not match row index")
        phi = _u64_to_phi(plan.yaw_u64)

        if self.stream == TrainStream.HML_MIXED:
            if plan.caption_index is None:
                raise ValueError("HML SamplePlan requires caption_index")
            text_row = row["texts"][int(plan.caption_index)]
            target_ref = text_row["target_k273_asset"]
            target, target_delta = _transform_to_gauge(self._load_asset(target_ref), phi)
            source = torch.zeros(1, 1, DIM_HY273, dtype=torch.float32)
            source_present = torch.zeros(1, dtype=torch.bool)
            source_valid = torch.zeros(1, 1, dtype=torch.bool)
            source_value_mask = torch.zeros(1, 1, DIM_HY273, dtype=torch.bool)
            source_roles = torch.full((1,), int(SourceRole.NULL), dtype=torch.long)
            source_lengths = torch.zeros(1, dtype=torch.long)
            source_deltas = torch.zeros(1, dtype=torch.float32)
            task = TaskId.GENERATE
            op = TargetOp.GENERATE
            frame_policy = FramePolicy.INDEPENDENT_SEQUENCE
            text = "" if plan.text_drop else str(text_row["value"])
            profile = str(text_row["encoding_profile"])
            ease_present = bool(plan.ease_present)
            ease_physical = (
                ease_from_k273(target)
                if ease_present
                else torch.zeros(6, dtype=torch.float32)
            )
        else:
            if plan.edit_pattern is None:
                raise ValueError("MotionFix SamplePlan requires an edit condition pattern")
            text_row = row["texts"][0]
            source_motion: torch.Tensor | None = None
            source_delta: float | None = None
            if plan.edit_pattern == EditConditionPattern.SOURCE_IDENTITY:
                source_motion, source_delta = _transform_to_gauge(
                    self._load_asset(row["source_motion"]["k273_asset"]), phi
                )
                # The source-only CFG branch has an explicit no-edit meaning.
                # Training it against the paired target lets a unique source ID
                # leak the edit and collapses (source+text - source) guidance.
                target = source_motion.clone()
                target_delta = source_delta
            else:
                target, target_delta = _transform_to_gauge(
                    self._load_asset(row["target_motion"]["k273_asset"]), phi
                )
            if plan.edit_pattern.uses_source:
                if source_motion is None or source_delta is None:
                    source_motion, source_delta = _transform_to_gauge(
                        self._load_asset(row["source_motion"]["k273_asset"]), phi
                    )
                source = source_motion.unsqueeze(0)
                source_present = torch.ones(1, dtype=torch.bool)
                source_valid = torch.ones(1, source_motion.shape[0], dtype=torch.bool)
                source_value_mask = torch.ones(
                    1, source_motion.shape[0], DIM_HY273, dtype=torch.bool
                )
                source_roles = torch.full(
                    (1,), int(SourceRole.SELF), dtype=torch.long
                )
                source_lengths = torch.tensor(
                    [source_motion.shape[0]], dtype=torch.long
                )
                source_deltas = torch.tensor([source_delta], dtype=torch.float32)
            else:
                # Edit CFG needs task-local text-only and unconditional branches.
                # Keep the finite K=1/Ts=1 sentinel used by source-free generation.
                source = torch.zeros(1, 1, DIM_HY273, dtype=torch.float32)
                source_present = torch.zeros(1, dtype=torch.bool)
                source_valid = torch.zeros(1, 1, dtype=torch.bool)
                source_value_mask = torch.zeros(
                    1, 1, DIM_HY273, dtype=torch.bool
                )
                source_roles = torch.full(
                    (1,), int(SourceRole.NULL), dtype=torch.long
                )
                source_lengths = torch.zeros(1, dtype=torch.long)
                source_deltas = torch.zeros(1, dtype=torch.float32)
            task = TaskId.EDIT
            op = TargetOp.EDIT
            frame_policy_name = str(row["pair"]["frame_policy_id"])
            if frame_policy_name != "independent_sequence_frame_v1":
                raise ValueError(f"Unsupported v1 MotionFix frame policy {frame_policy_name}")
            frame_policy = FramePolicy.INDEPENDENT_SEQUENCE
            text = "" if plan.text_drop else str(text_row["value"])
            profile = str(text_row["encoding_profile"])
            if plan.ease_present:
                raise ValueError("MotionFix Edit replay cannot carry Ease in v1")
            ease_present = False
            ease_physical = torch.zeros(6, dtype=torch.float32)

        target_valid = torch.ones(target.shape[0], dtype=torch.bool)
        target_ops = torch.full((target.shape[0],), int(op), dtype=torch.long)
        gauge = torch.tensor([math.cos(phi), math.sin(phi)], dtype=torch.float32)
        return {
            "uid": str(row["uid"]),
            "dataset": str(row["dataset"]),
            "train_stream_id": int(self.stream),
            "task_id": int(task),
            "capability_id": int(plan.capability_id),
            "text": text,
            "text_encoding_profile": profile,
            "target_motion": target,
            "target_valid": target_valid,
            "target_op_id": target_ops,
            "source_motion": source,
            "source_present": source_present,
            "source_time_valid": source_valid,
            "source_value_mask": source_value_mask,
            "source_role_id": source_roles,
            "source_native_lengths": source_lengths,
            "requested_target_len": int(target.shape[0]),
            "frame_gauge_dir": gauge,
            "frame_policy_id": int(frame_policy),
            "ease_physical": ease_physical,
            "ease_present": ease_present,
            "target_applied_yaw_delta": float(target_delta),
            "source_applied_yaw_deltas": source_deltas,
            "plan": plan,
        }


def build_global_sample_plans(
    *,
    dataset: HY273MultitaskManifestDataset,
    row_indices: Sequence[int],
    global_step: int,
    first_global_ordinal: int,
    run_seed: int,
    schedule_version: str = HIGH_LEVEL_SCHEDULE_VERSION,
) -> list[SamplePlan]:
    plans: list[SamplePlan] = []
    for offset, row_index in enumerate(row_indices):
        ordinal = int(first_global_ordinal) + offset
        uid = dataset.uid(row_index)
        task = TaskId.GENERATE if dataset.stream == TrainStream.HML_MIXED else TaskId.EDIT

        def draw(name: str) -> int:
            return sample_key_u64(
                manifest_sha256=dataset.manifest_sha256,
                run_seed=run_seed,
                global_sample_ordinal=ordinal,
                train_stream_id=dataset.stream,
                task_id=int(task),
                uid=uid,
                random_stream_id=name,
            )

        if dataset.stream == TrainStream.HML_MIXED:
            capability = hml_capability_from_draw(
                global_step,
                draw("hml_capability"),
                schedule_version,
            )
            caption_index = draw("caption") % dataset.caption_count(row_index)
            text_drop = bernoulli_from_draw(draw("hml_text_dropout"), 0.10)
            ease_present = ease_present_from_draw(
                capability,
                draw("ease_presence"),
                schedule_version,
            )
            edit_pattern = None
        else:
            edit_pattern = edit_pattern_from_draw(
                draw("edit_condition_pattern"), schedule_version
            )
            capability = (
                CapabilityId.MOTION_EDIT_CONTROL
                if edit_pattern.uses_control
                else CapabilityId.MOTION_EDIT
            )
            caption_index = None
            text_drop = not edit_pattern.uses_text
            ease_present = False
        plans.append(
            SamplePlan(
                global_step=int(global_step),
                global_sample_ordinal=ordinal,
                train_stream_id=dataset.stream,
                capability_id=capability,
                row_index=int(row_index),
                uid=uid,
                caption_index=None if caption_index is None else int(caption_index),
                yaw_u64=draw("yaw_phi"),
                control_u64=draw("control_pattern"),
                text_drop=bool(text_drop),
                edit_pattern=edit_pattern,
                ease_present=bool(ease_present),
            )
        )
    return plans


def collate_hy273_multitask(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty multitask batch")
    stream_ids = {int(sample["train_stream_id"]) for sample in samples}
    if len(stream_ids) != 1:
        raise ValueError("Multitask batches must be high-level stream homogeneous")
    bsz = len(samples)
    target_frames = max(int(sample["target_motion"].shape[0]) for sample in samples)
    source_slots = max(int(sample["source_motion"].shape[0]) for sample in samples)
    source_frames = max(int(sample["source_motion"].shape[1]) for sample in samples)

    target = torch.zeros(bsz, target_frames, DIM_HY273, dtype=torch.float32)
    target_valid = torch.zeros(bsz, target_frames, dtype=torch.bool)
    target_ops = torch.full(
        (bsz, target_frames), int(TargetOp.PRESERVE), dtype=torch.long
    )
    source = torch.zeros(
        bsz, source_slots, source_frames, DIM_HY273, dtype=torch.float32
    )
    source_present = torch.zeros(bsz, source_slots, dtype=torch.bool)
    source_valid = torch.zeros(bsz, source_slots, source_frames, dtype=torch.bool)
    source_value_mask = torch.zeros(
        bsz, source_slots, source_frames, DIM_HY273, dtype=torch.bool
    )
    source_roles = torch.full(
        (bsz, source_slots), int(SourceRole.NULL), dtype=torch.long
    )
    source_lengths = torch.zeros(bsz, source_slots, dtype=torch.long)

    for index, sample in enumerate(samples):
        target_len = int(sample["target_motion"].shape[0])
        slots, source_len = sample["source_motion"].shape[:2]
        target[index, :target_len] = sample["target_motion"]
        target_valid[index, :target_len] = sample["target_valid"]
        target_ops[index, :target_len] = sample["target_op_id"]
        source[index, :slots, :source_len] = sample["source_motion"]
        source_present[index, :slots] = sample["source_present"]
        source_valid[index, :slots, :source_len] = sample["source_time_valid"]
        source_value_mask[index, :slots, :source_len] = sample["source_value_mask"]
        source_roles[index, :slots] = sample["source_role_id"]
        source_lengths[index, :slots] = sample["source_native_lengths"]

    condition = ConditionBatch(
        train_stream_id=torch.tensor(
            [sample["train_stream_id"] for sample in samples], dtype=torch.long
        ),
        task_id=torch.tensor([sample["task_id"] for sample in samples], dtype=torch.long),
        capability_id=torch.tensor(
            [sample["capability_id"] for sample in samples], dtype=torch.long
        ),
        text_encoding_profile=tuple(
            str(sample["text_encoding_profile"]) for sample in samples
        ),
        target_valid=target_valid,
        target_op_id=target_ops,
        source_motion=source,
        source_present=source_present,
        source_time_valid=source_valid,
        source_value_mask=source_value_mask,
        source_role_id=source_roles,
        source_native_lengths=source_lengths,
        requested_target_len=torch.tensor(
            [sample["requested_target_len"] for sample in samples], dtype=torch.long
        ),
        frame_gauge_dir=torch.stack(
            [sample["frame_gauge_dir"] for sample in samples]
        ),
        frame_policy_id=torch.tensor(
            [sample["frame_policy_id"] for sample in samples], dtype=torch.long
        ),
        ease_physical=torch.stack(
            [sample["ease_physical"] for sample in samples]
        ),
        ease_present=torch.tensor(
            [sample["ease_present"] for sample in samples], dtype=torch.bool
        ),
        target_to_source_time_map=None,
    )
    has_source_free_conditional = bool(
        (
            (
                (condition.task_id == int(TaskId.EDIT))
                | (condition.task_id == int(TaskId.REACTION))
            )
            & ~condition.source_present.any(dim=1)
        ).any()
    )
    condition.validate(v1_strict=not has_source_free_conditional)
    return {
        "target_motion": target,
        "condition": condition,
        "texts": [str(sample["text"]) for sample in samples],
        "uids": [str(sample["uid"]) for sample in samples],
        "plans": [sample["plan"] for sample in samples],
        "target_applied_yaw_delta": torch.tensor(
            [sample["target_applied_yaw_delta"] for sample in samples],
            dtype=torch.float32,
        ),
    }
