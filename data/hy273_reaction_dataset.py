"""Inter-X actor-to-reactor samples in the shared single-target K273 contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch

from data.hy273_multitask_manifest_dataset import (
    collate_hy273_multitask,
    sha256_file,
)
from data.hy273_multitask_scheduler import sample_key_u64
from models.raw_motion.hy273_multitask_condition import (
    CapabilityId,
    FramePolicy,
    INTERACTION_TEXT_PROFILE,
    SourceRole,
    TargetOp,
    TaskId,
    TrainStream,
)
from models.raw_motion.hy273_normalizer import apply_yaw_rotation
from models.raw_motion.hy273_slices import (
    CONTACT_SLICE,
    DIM_HY273,
    HEADING_SLICE,
    ROOT_SLICE,
)


REACTION_EXCLUDED_TEST_CLIPS = {"G046T007A038R019"}


class ReactionConditionPattern(IntEnum):
    SOURCE_AND_TEXT = 0
    SOURCE_ONLY = 1
    UNCONDITIONAL = 2

    @property
    def uses_source(self) -> bool:
        return self != ReactionConditionPattern.UNCONDITIONAL

    @property
    def uses_text(self) -> bool:
        return self == ReactionConditionPattern.SOURCE_AND_TEXT


def reaction_pattern_from_draw(draw: int) -> ReactionConditionPattern:
    """Use 90/5/5 joint, actor-only, and task-unconditional branches."""

    bucket = int(draw) % 100
    if bucket < 90:
        return ReactionConditionPattern.SOURCE_AND_TEXT
    if bucket < 95:
        return ReactionConditionPattern.SOURCE_ONLY
    return ReactionConditionPattern.UNCONDITIONAL


def _u64_to_phi(value: int) -> float:
    unit = (int(value) + 0.5) / float(1 << 64)
    return (2.0 * unit - 1.0) * math.pi


def apply_shared_reaction_gauge(
    source: torch.Tensor,
    target: torch.Tensor,
    phi: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Center on the observed actor and rotate both motions by one shared yaw."""

    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("Reaction source/target must share shape [T,273]")
    if source.shape[-1] != DIM_HY273:
        raise ValueError(f"Reaction feature dim must be {DIM_HY273}")
    pair = torch.stack([source, target], dim=0).clone()
    anchor_x = pair[0, 0, ROOT_SLICE.start].clone()
    anchor_z = pair[0, 0, ROOT_SLICE.start + 2].clone()
    pair[..., ROOT_SLICE.start] -= anchor_x
    pair[..., ROOT_SLICE.start + 2] -= anchor_z
    heading = pair[0, 0, HEADING_SLICE]
    current = torch.atan2(heading[1], heading[0])
    delta = torch.as_tensor(phi, dtype=pair.dtype) - current
    pair = apply_yaw_rotation(pair, delta)
    return pair[0], pair[1], float(delta.item())


@dataclass(frozen=True)
class ReactionSamplePlan:
    global_step: int
    global_sample_ordinal: int
    row_index: int
    uid: str
    caption_index: int
    crop_start: int
    yaw_u64: int
    condition_pattern: ReactionConditionPattern


class HY273ReactionDataset:
    """Observed Inter-X actor plus text mapped to one reactor target."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        min_frames: int = 16,
        max_frames: int = 300,
        exclude_overlength: bool = True,
        exclude_known_test_anomalies: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / "manifest.jsonl"
        self.order_path = self.root / "interaction_order.pkl"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing Inter-X manifest: {self.manifest_path}")
        if not self.order_path.is_file():
            raise FileNotFoundError(f"Missing Inter-X actor order: {self.order_path}")
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.split = str(split)
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.exclude_overlength = bool(exclude_overlength)
        with self.order_path.open("rb") as handle:
            order = pickle.load(handle)
        if not isinstance(order, dict):
            raise TypeError("interaction_order.pkl must contain a dict")
        self.actor_order = {str(key): int(value) for key, value in order.items()}

        rows: list[dict[str, Any]] = []
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != self.split:
                    continue
                if row.get("dataset") != "interx":
                    raise ValueError(
                        f"Reaction dataset accepts Inter-X only, line {line_number}"
                    )
                if not bool(row.get("has_text")) or not row.get("texts"):
                    continue
                if int(row.get("dim", -1)) != DIM_HY273:
                    raise ValueError(f"Reaction dim mismatch at line {line_number}")
                if float(row.get("fps", -1.0)) != 30.0:
                    raise ValueError(f"Reaction FPS mismatch at line {line_number}")
                frames = int(row.get("frames", 0))
                if frames < self.min_frames:
                    continue
                if self.exclude_overlength and frames > self.max_frames:
                    continue
                clip_id = str(row.get("clip_id", ""))
                if clip_id not in self.actor_order:
                    raise KeyError(f"Missing actor/reactor order for {clip_id}")
                if self.actor_order[clip_id] not in {0, 1}:
                    raise ValueError(f"Invalid actor order for {clip_id}")
                if (
                    exclude_known_test_anomalies
                    and self.split == "test"
                    and clip_id in REACTION_EXCLUDED_TEST_CLIPS
                ):
                    continue
                rows.append(row)
        if not rows:
            raise RuntimeError(f"No usable Inter-X Reaction rows for split={self.split}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def uid(self, index: int) -> str:
        return str(self.rows[int(index)]["clip_id"])

    @staticmethod
    def _bucket(length: int) -> int:
        return max(0, (int(length) - 1) // 32)

    @property
    def bucket_keys(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                self._bucket(min(int(row["frames"]), self.max_frames)),
                self._bucket(min(int(row["frames"]), self.max_frames)),
            )
            for row in self.rows
        )

    def build_plan(
        self,
        *,
        row_index: int,
        global_step: int,
        global_sample_ordinal: int,
        run_seed: int,
    ) -> ReactionSamplePlan:
        row = self.rows[int(row_index)]
        uid = str(row["clip_id"])

        def draw(name: str) -> int:
            return sample_key_u64(
                manifest_sha256=self.manifest_sha256,
                run_seed=int(run_seed),
                global_sample_ordinal=int(global_sample_ordinal),
                train_stream_id=int(TrainStream.REACTION),
                task_id=int(TaskId.REACTION),
                uid=uid,
                random_stream_id=name,
            )

        frames = int(row["frames"])
        crop_choices = max(1, frames - self.max_frames + 1)
        return ReactionSamplePlan(
            global_step=int(global_step),
            global_sample_ordinal=int(global_sample_ordinal),
            row_index=int(row_index),
            uid=uid,
            caption_index=int(draw("caption") % len(row["texts"])),
            crop_start=int(draw("paired_crop") % crop_choices),
            yaw_u64=draw("shared_yaw_phi"),
            condition_pattern=reaction_pattern_from_draw(draw("condition_pattern")),
        )

    def _load_actor(self, relative_path: str, expected_frames: int) -> torch.Tensor:
        path = (self.root / relative_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing Inter-X actor file: {path}")
        array = np.load(path, allow_pickle=False)
        if array.shape != (expected_frames, DIM_HY273) or array.dtype != np.float32:
            raise ValueError(f"Bad Inter-X tensor {path}: {array.shape}/{array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite Inter-X tensor: {path}")
        contacts = array[..., CONTACT_SLICE]
        if not np.logical_or(contacts == 0.0, contacts == 1.0).all():
            raise ValueError(f"Non-binary Inter-X contacts: {path}")
        return torch.from_numpy(array.copy())

    def materialize(self, plan: ReactionSamplePlan) -> dict[str, Any]:
        row = self.rows[int(plan.row_index)]
        if str(row["clip_id"]) != plan.uid:
            raise ValueError("Reaction plan UID does not match its row")
        frames = int(row["frames"])
        people = (
            self._load_actor(str(row["person1"]), frames),
            self._load_actor(str(row["person2"]), frames),
        )
        actor_index = self.actor_order[plan.uid]
        source = people[actor_index]
        target = people[1 - actor_index]
        start = int(plan.crop_start)
        length = min(frames, self.max_frames)
        source = source[start : start + length]
        target = target[start : start + length]
        phi = _u64_to_phi(plan.yaw_u64)
        source, target, yaw_delta = apply_shared_reaction_gauge(
            source, target, phi
        )

        pattern = ReactionConditionPattern(plan.condition_pattern)
        if pattern.uses_source:
            source_motion = source.unsqueeze(0)
            source_present = torch.ones(1, dtype=torch.bool)
            source_valid = torch.ones(1, length, dtype=torch.bool)
            source_value_mask = torch.ones(
                1, length, DIM_HY273, dtype=torch.bool
            )
            source_role = torch.full(
                (1,),
                int(
                    SourceRole.OTHER_ACTOR_FIRST_PERSON
                    if actor_index == 0
                    else SourceRole.OTHER_ACTOR_SECOND_PERSON
                ),
                dtype=torch.long,
            )
            source_lengths = torch.tensor([length], dtype=torch.long)
            source_deltas = torch.tensor([yaw_delta], dtype=torch.float32)
        else:
            source_motion = torch.zeros(1, 1, DIM_HY273, dtype=torch.float32)
            source_present = torch.zeros(1, dtype=torch.bool)
            source_valid = torch.zeros(1, 1, dtype=torch.bool)
            source_value_mask = torch.zeros(
                1, 1, DIM_HY273, dtype=torch.bool
            )
            source_role = torch.full(
                (1,), int(SourceRole.NULL), dtype=torch.long
            )
            source_lengths = torch.zeros(1, dtype=torch.long)
            source_deltas = torch.zeros(1, dtype=torch.float32)

        text = (
            str(row["texts"][plan.caption_index]) if pattern.uses_text else ""
        )
        return {
            "uid": plan.uid,
            "dataset": "interx_k273_reaction",
            "train_stream_id": int(TrainStream.REACTION),
            "task_id": int(TaskId.REACTION),
            "capability_id": int(CapabilityId.TEXT_REACTION),
            "text": text,
            "text_encoding_profile": INTERACTION_TEXT_PROFILE,
            "target_motion": target,
            "target_valid": torch.ones(length, dtype=torch.bool),
            "target_op_id": torch.full(
                (length,), int(TargetOp.GENERATE), dtype=torch.long
            ),
            "source_motion": source_motion,
            "source_present": source_present,
            "source_time_valid": source_valid,
            "source_value_mask": source_value_mask,
            "source_role_id": source_role,
            "source_native_lengths": source_lengths,
            "requested_target_len": length,
            "frame_gauge_dir": torch.tensor(
                [math.cos(phi), math.sin(phi)], dtype=torch.float32
            ),
            "frame_policy_id": int(FramePolicy.SHARED_WORLD),
            "ease_physical": torch.zeros(6, dtype=torch.float32),
            "ease_present": False,
            "target_applied_yaw_delta": yaw_delta,
            "source_applied_yaw_deltas": source_deltas,
            "actor_person_index": actor_index,
            "condition_pattern": pattern,
            "plan": plan,
        }


def collate_hy273_reaction(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return collate_hy273_multitask(samples)
