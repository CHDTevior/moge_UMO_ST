"""Manifest-backed paired-actor K273 data for text-to-interaction training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from data.hy273_multitask_scheduler import bernoulli_from_draw, sample_key_u64
from models.raw_motion.hy273_multitask_condition import (
    CapabilityId,
    INTERACTION_TEXT_PROFILE,
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


INTERACTION_EXCLUDED_TEST_CLIPS = {
    "G046T007A038R019": {
        "reason": (
            "Confirmed source defect: P2 late teleport, root flip, root speed "
            ">10 m/s, and >150 degree heading step are present in both raw "
            "SMPL-X and the independent released 64-joint geometry."
        ),
        "qa_report": (
            "/mnt/afs/mogo_base/datasets/InteractionK273/reports/"
            "FINAL_QA_REPORT_20260716.md (section 6, line 120)"
        ),
    }
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _u64_to_phi(value: int) -> float:
    unit = (int(value) + 0.5) / float(1 << 64)
    return (2.0 * unit - 1.0) * math.pi


def apply_shared_interaction_gauge(
    motion: torch.Tensor, phi: float
) -> tuple[torch.Tensor, float]:
    """Center/rotate a pair without changing its relative world geometry."""

    if motion.ndim != 3 or motion.shape[0] != 2 or motion.shape[-1] != DIM_HY273:
        raise ValueError("Interaction motion must have shape [2,T,273]")
    output = motion.clone()
    root_x = output[0, 0, ROOT_SLICE.start].clone()
    root_z = output[0, 0, ROOT_SLICE.start + 2].clone()
    output[..., ROOT_SLICE.start] -= root_x
    output[..., ROOT_SLICE.start + 2] -= root_z
    heading = output[0, 0, HEADING_SLICE]
    current = torch.atan2(heading[1], heading[0])
    delta = torch.as_tensor(phi, dtype=output.dtype) - current
    output = apply_yaw_rotation(output, delta)
    return output, float(delta.item())


@dataclass(frozen=True)
class InteractionSamplePlan:
    global_step: int
    global_sample_ordinal: int
    row_index: int
    uid: str
    caption_index: int
    crop_start: int
    yaw_u64: int
    actor_swap: bool
    text_drop: bool


class HY273InteractionDataset:
    """Text-bearing InterHuman/Inter-X pairs with synchronized augmentation."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        min_frames: int = 16,
        max_frames: int = 300,
        exclude_known_test_anomalies: bool = True,
        exclude_overlength: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / "manifest.jsonl"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing Interaction manifest: {self.manifest_path}")
        self.manifest_sha256 = _sha256_file(self.manifest_path)
        self.split = str(split)
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.exclude_overlength = bool(exclude_overlength)
        if self.min_frames < 1 or self.max_frames < self.min_frames:
            raise ValueError("Invalid Interaction frame bounds")

        rows: list[dict[str, Any]] = []
        self.excluded_rows: list[dict[str, str]] = []
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != self.split:
                    continue
                if not bool(row.get("has_text")) or not row.get("texts"):
                    continue
                if int(row.get("dim", -1)) != DIM_HY273:
                    raise ValueError(
                        f"Interaction dim mismatch at manifest line {line_number}"
                    )
                if float(row.get("fps", -1.0)) != 30.0:
                    raise ValueError(
                        f"Interaction FPS mismatch at manifest line {line_number}"
                    )
                if int(row.get("frames", 0)) < self.min_frames:
                    continue
                if (
                    self.exclude_overlength
                    and int(row.get("frames", 0)) > self.max_frames
                ):
                    continue
                if (
                    exclude_known_test_anomalies
                    and self.split == "test"
                    and str(row.get("source_clip_id"))
                    in INTERACTION_EXCLUDED_TEST_CLIPS
                ):
                    source_clip_id = str(row["source_clip_id"])
                    self.excluded_rows.append(
                        {
                            "clip_id": str(row["clip_id"]),
                            "source_clip_id": source_clip_id,
                            **INTERACTION_EXCLUDED_TEST_CLIPS[source_clip_id],
                        }
                    )
                    continue
                rows.append(row)
        if not rows:
            raise RuntimeError(f"No usable text Interaction rows for split={self.split}")
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
            (self._bucket(min(int(row["frames"]), self.max_frames)), 0)
            for row in self.rows
        )

    def build_plan(
        self,
        *,
        row_index: int,
        global_step: int,
        global_sample_ordinal: int,
        run_seed: int,
    ) -> InteractionSamplePlan:
        row_index = int(row_index)
        row = self.rows[row_index]
        uid = str(row["clip_id"])

        def draw(name: str) -> int:
            return sample_key_u64(
                manifest_sha256=self.manifest_sha256,
                run_seed=int(run_seed),
                global_sample_ordinal=int(global_sample_ordinal),
                train_stream_id=TrainStream.INTERACTION,
                task_id=int(TaskId.INTERACTION),
                uid=uid,
                random_stream_id=name,
            )

        frames = int(row["frames"])
        crop_choices = max(1, frames - self.max_frames + 1)
        return InteractionSamplePlan(
            global_step=int(global_step),
            global_sample_ordinal=int(global_sample_ordinal),
            row_index=row_index,
            uid=uid,
            caption_index=int(draw("caption") % len(row["texts"])),
            crop_start=int(draw("paired_crop") % crop_choices),
            yaw_u64=draw("shared_yaw_phi"),
            actor_swap=bernoulli_from_draw(draw("actor_swap"), 0.5),
            text_drop=bernoulli_from_draw(draw("text_dropout"), 0.10),
        )

    def _load_actor(self, relative_path: str, expected_frames: int) -> torch.Tensor:
        path = (self.root / relative_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing Interaction actor file: {path}")
        array = np.load(path, allow_pickle=False)
        if array.shape != (expected_frames, DIM_HY273) or array.dtype != np.float32:
            raise ValueError(
                f"Bad Interaction actor tensor {path}: {array.shape}/{array.dtype}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite Interaction actor tensor: {path}")
        contacts = array[..., CONTACT_SLICE]
        if not np.logical_or(contacts == 0.0, contacts == 1.0).all():
            raise ValueError(f"Non-binary Interaction contacts: {path}")
        return torch.from_numpy(array.copy())

    def materialize(self, plan: InteractionSamplePlan) -> dict[str, Any]:
        row = self.rows[int(plan.row_index)]
        if str(row["clip_id"]) != plan.uid:
            raise ValueError("Interaction plan UID does not match its row")
        frames = int(row["frames"])
        pair = torch.stack(
            [
                self._load_actor(str(row["person1"]), frames),
                self._load_actor(str(row["person2"]), frames),
            ],
            dim=0,
        )
        start = int(plan.crop_start)
        length = min(frames, self.max_frames)
        pair = pair[:, start : start + length]
        phi = _u64_to_phi(plan.yaw_u64)
        pair, yaw_delta = apply_shared_interaction_gauge(pair, phi)
        # The shared scene gauge is defined before actor ordering. Swapping
        # afterward makes augmentation a true permutation of the paired target,
        # matching the actor-exchange equivariance used by InterGen-style models.
        if plan.actor_swap:
            pair = pair.flip(dims=(0,))
        text = "" if plan.text_drop else str(row["texts"][plan.caption_index])
        return {
            "uid": str(row["clip_id"]),
            "source_dataset": str(row["source_dataset"]),
            "source_clip_id": str(row["source_clip_id"]),
            "motion": pair,
            "valid": torch.ones(2, pair.shape[1], dtype=torch.bool),
            "text": text,
            "text_encoding_profile": INTERACTION_TEXT_PROFILE,
            "task_id": int(TaskId.INTERACTION),
            "capability_id": int(CapabilityId.TEXT_INTERACTION),
            "frame_gauge_dir": torch.tensor(
                [math.cos(phi), math.sin(phi)], dtype=torch.float32
            ),
            "actor_swap": bool(plan.actor_swap),
            "crop_start": start,
            "applied_yaw_delta": yaw_delta,
            "plan": plan,
        }


def collate_hy273_interaction(
    samples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty Interaction batch")
    batch = len(samples)
    frames = max(int(sample["motion"].shape[1]) for sample in samples)
    target = torch.zeros(batch, 2, frames, DIM_HY273, dtype=torch.float32)
    valid = torch.zeros(batch, 2, frames, dtype=torch.bool)
    for index, sample in enumerate(samples):
        length = int(sample["motion"].shape[1])
        target[index, :, :length] = sample["motion"]
        valid[index, :, :length] = sample["valid"]
    return {
        "target_motion": target,
        "target_valid": valid,
        "texts": [str(sample["text"]) for sample in samples],
        "text_profiles": tuple(
            str(sample["text_encoding_profile"]) for sample in samples
        ),
        "task_id": torch.tensor(
            [sample["task_id"] for sample in samples], dtype=torch.long
        ),
        "capability_id": torch.tensor(
            [sample["capability_id"] for sample in samples], dtype=torch.long
        ),
        "frame_gauge_dir": torch.stack(
            [sample["frame_gauge_dir"] for sample in samples]
        ),
        "uids": [str(sample["uid"]) for sample in samples],
        "source_datasets": [str(sample["source_dataset"]) for sample in samples],
        "plans": [sample["plan"] for sample in samples],
    }
