"""Dataset wrappers for converted HumanML3D/MotionFix Kimodo273 motion files."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from models.raw_motion.hy273_slices import DIM_HY273, FALLBACK_SHORT_CLIPS


_MASK64 = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def deterministic_item_value(seed: int, epoch: int, index: int, stream: int) -> int:
    value = int(seed) & _MASK64
    value ^= (int(epoch) * 0xD2B74407B1CE6E93) & _MASK64
    value ^= (int(index) * 0xCA5A826395121157) & _MASK64
    value ^= (int(stream) * 0x9E3779B97F4A7C15) & _MASK64
    return _splitmix64(value)


def parse_hml_text_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if "#" in line:
        return line.split("#", 1)[0].strip()
    return line


CaptionPolicy = Literal["default", "first", "first_full_motion"]


@dataclass(frozen=True)
class HMLCaption:
    text: str
    from_tag: float | None
    to_tag: float | None
    source_line_number: int

    @property
    def is_full_motion(self) -> bool:
        return (
            self.from_tag is not None
            and self.to_tag is not None
            and self.from_tag == 0.0
            and self.to_tag == 0.0
        )


def _parse_hml_tag(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    # HumanML3D loaders conventionally map NaN span tags to zero, meaning that
    # the caption describes the complete motion.
    return 0.0 if math.isnan(parsed) else parsed


def _read_caption_entries(path: Path) -> list[HMLCaption]:
    if not path.is_file():
        return [HMLCaption("", 0.0, 0.0, 0)]
    captions: list[HMLCaption] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split("#")
            text = fields[0].strip()
            if not text:
                continue
            if len(fields) >= 4:
                from_tag = _parse_hml_tag(fields[2])
                to_tag = _parse_hml_tag(fields[3])
            else:
                # Untagged captions can only describe the complete clip.
                from_tag = 0.0
                to_tag = 0.0
            captions.append(
                HMLCaption(
                    text=text,
                    from_tag=from_tag,
                    to_tag=to_tag,
                    source_line_number=line_number,
                )
            )
    return captions or [HMLCaption("", 0.0, 0.0, 0)]


def _read_captions(path: Path) -> list[str]:
    return [caption.text for caption in _read_caption_entries(path)]


@dataclass
class Kimodo273Item:
    motion: torch.Tensor
    length: int
    text: str
    rel_path: str
    motion_id: str


class Kimodo273TextDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        text_root: str | Path | None = None,
        max_frames: int = 300,
        min_frames: int = 16,
        random_crop: bool = True,
        exclude_fallback_short_clips: bool = True,
        deterministic_text: bool = False,
        caption_policy: CaptionPolicy = "default",
        cache_manifest: bool = True,
        trace_seed: int | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = str(split)
        self.text_root = Path(text_root).expanduser().resolve() if text_root else self.data_root.parent / "texts"
        self.max_frames = int(max_frames)
        self.min_frames = int(min_frames)
        self.random_crop = bool(random_crop)
        self.deterministic_text = bool(deterministic_text)
        if caption_policy not in {"default", "first", "first_full_motion"}:
            raise ValueError(f"Unknown caption policy: {caption_policy}")
        self.caption_policy: CaptionPolicy = caption_policy
        self.trace_seed = None if trace_seed is None else int(trace_seed)
        self.trace_epoch = 0
        split_path = self.data_root / "split_existing" / f"{self.split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        frame_by_rel: dict[str, int] = {}
        fallback_by_rel: dict[str, bool] = {}
        manifest_path = self.data_root / "manifest.jsonl"
        if cache_manifest and manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    rel = str(row["relative_path"])
                    frame_by_rel[rel] = int(row.get("frames", 0))
                    fallback_by_rel[rel] = bool(row.get("smooth_root_fallback", False))
        records: list[dict[str, Any]] = []
        for motion_id in ids:
            rel = f"motion_data/{motion_id}.npy"
            path = self.data_root / rel
            if not path.is_file():
                continue
            frames = frame_by_rel.get(rel)
            if frames is None:
                frames = int(np.load(path, mmap_mode="r").shape[0])
            if frames < self.min_frames:
                continue
            if exclude_fallback_short_clips and (rel in FALLBACK_SHORT_CLIPS or fallback_by_rel.get(rel, False)):
                continue
            records.append({"id": motion_id, "rel": rel, "frames": frames, "path": path})
        if not records:
            raise RuntimeError(f"No usable records for split={split} under {self.data_root}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def set_trace_epoch(self, epoch: int) -> None:
        self.trace_epoch = int(epoch)

    def _crop(self, motion: np.ndarray, deterministic_value: int | None = None) -> tuple[np.ndarray, int]:
        if self.max_frames <= 0 or motion.shape[0] <= self.max_frames:
            return motion, 0
        span = motion.shape[0] - self.max_frames
        if self.random_crop:
            start = (
                random.randint(0, span)
                if deterministic_value is None
                else int(deterministic_value % (span + 1))
            )
        else:
            start = span // 2
        return motion[start : start + self.max_frames], int(start)

    def caption_metadata(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        motion_id = str(rec["id"])
        captions = _read_caption_entries(self.text_root / f"{motion_id}.txt")
        if self.caption_policy == "first_full_motion":
            caption_index = next(
                (
                    caption_index
                    for caption_index, caption in enumerate(captions)
                    if caption.is_full_motion
                ),
                0,
            )
        elif self.caption_policy == "first" or self.deterministic_text:
            caption_index = 0
        elif self.trace_seed is not None:
            caption_value = deterministic_item_value(
                self.trace_seed, self.trace_epoch, index, stream=1
            )
            caption_index = int(caption_value % len(captions))
        else:
            caption_index = random.randrange(len(captions))
        caption = captions[caption_index]
        return {
            "text": caption.text,
            "caption_index": int(caption_index),
            "caption_line_number": int(caption.source_line_number),
            "caption_from_tag": caption.from_tag,
            "caption_to_tag": caption.to_tag,
            "caption_is_full_motion": bool(caption.is_full_motion),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        motion = np.load(rec["path"]).astype(np.float32, copy=False)
        if motion.ndim != 2 or motion.shape[1] != DIM_HY273:
            raise ValueError(f"Expected [T,{DIM_HY273}] at {rec['path']}, got {motion.shape}")
        crop_value = None
        if self.trace_seed is not None:
            crop_value = deterministic_item_value(self.trace_seed, self.trace_epoch, index, stream=0)
        motion, crop_start = self._crop(motion, deterministic_value=crop_value)
        motion_id = rec["id"]
        caption = self.caption_metadata(index)
        return {
            "motion": torch.from_numpy(motion.copy()),
            "length": int(motion.shape[0]),
            "text": caption["text"],
            "rel_path": rec["rel"],
            "motion_id": motion_id,
            "dataset_index": int(index),
            "crop_start": int(crop_start),
            **caption,
        }


def collate_kimodo273_text(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    max_len = max(int(item["length"]) for item in batch)
    motion = torch.zeros(len(batch), max_len, DIM_HY273, dtype=torch.float32)
    valid = torch.zeros(len(batch), max_len, dtype=torch.bool)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    texts: list[str] = []
    rel_paths: list[str] = []
    motion_ids: list[str] = []
    dataset_indices = torch.zeros(len(batch), dtype=torch.long)
    crop_starts = torch.zeros(len(batch), dtype=torch.long)
    caption_indices = torch.zeros(len(batch), dtype=torch.long)
    caption_line_numbers = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        cur = item["motion"].float()
        length = int(cur.shape[0])
        motion[i, :length] = cur
        valid[i, :length] = True
        lengths[i] = length
        texts.append(str(item["text"]))
        rel_paths.append(str(item["rel_path"]))
        motion_ids.append(str(item["motion_id"]))
        dataset_indices[i] = int(item.get("dataset_index", -1))
        crop_starts[i] = int(item.get("crop_start", 0))
        caption_indices[i] = int(item.get("caption_index", 0))
        caption_line_numbers[i] = int(item.get("caption_line_number", 0))
    return {
        "motion": motion,
        "lengths": lengths,
        "valid": valid,
        "texts": texts,
        "rel_paths": rel_paths,
        "motion_ids": motion_ids,
        "dataset_indices": dataset_indices,
        "crop_starts": crop_starts,
        "caption_indices": caption_indices,
        "caption_line_numbers": caption_line_numbers,
    }
