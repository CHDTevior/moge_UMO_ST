from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data.kimodo273_datasets import Kimodo273TextDataset, collate_kimodo273_text


def _make_dataset(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "kimodo273"
    motion_dir = data_root / "motion_data"
    split_dir = data_root / "split_existing"
    text_root = tmp_path / "texts"
    motion_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    text_root.mkdir(parents=True)

    motion = np.arange(40 * 273, dtype=np.float32).reshape(40, 273)
    np.save(motion_dir / "sample.npy", motion)
    (split_dir / "train.txt").write_text("sample\n", encoding="utf-8")
    (text_root / "sample.txt").write_text(
        "caption zero#token#0#0\ncaption one#token#0#0\ncaption two#token#0#0\n",
        encoding="utf-8",
    )
    manifest = {
        "relative_path": "motion_data/sample.npy",
        "frames": 40,
        "smooth_root_fallback": False,
    }
    (data_root / "manifest.jsonl").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return data_root, text_root


def test_trace_seed_replays_crop_caption_and_collate_metadata(tmp_path: Path):
    data_root, text_root = _make_dataset(tmp_path)
    first = Kimodo273TextDataset(
        data_root,
        text_root=text_root,
        max_frames=16,
        min_frames=1,
        trace_seed=3407,
    )
    second = Kimodo273TextDataset(
        data_root,
        text_root=text_root,
        max_frames=16,
        min_frames=1,
        trace_seed=3407,
    )

    first.set_trace_epoch(17)
    second.set_trace_epoch(17)
    item_a = first[0]
    item_b = second[0]
    assert item_a["crop_start"] == item_b["crop_start"]
    assert item_a["caption_index"] == item_b["caption_index"]
    assert item_a["text"] == item_b["text"]
    assert torch.equal(item_a["motion"], item_b["motion"])

    batch = collate_kimodo273_text([item_a])
    assert batch["dataset_indices"].tolist() == [0]
    assert batch["crop_starts"].tolist() == [item_a["crop_start"]]
    assert batch["caption_indices"].tolist() == [item_a["caption_index"]]


def test_trace_epoch_changes_deterministic_item_stream(tmp_path: Path):
    data_root, text_root = _make_dataset(tmp_path)
    dataset = Kimodo273TextDataset(
        data_root,
        text_root=text_root,
        max_frames=16,
        min_frames=1,
        trace_seed=3407,
    )
    signatures = set()
    for epoch in range(8):
        dataset.set_trace_epoch(epoch)
        item = dataset[0]
        signatures.add((item["crop_start"], item["caption_index"]))
    assert len(signatures) > 1


def test_first_full_motion_caption_policy_preserves_source_index(tmp_path: Path):
    data_root, text_root = _make_dataset(tmp_path)
    (text_root / "sample.txt").write_text(
        "segment caption#token#1.0#8.0\n"
        "full caption#token#0.0#0.0\n"
        "nan means full#token#nan#nan\n",
        encoding="utf-8",
    )
    dataset = Kimodo273TextDataset(
        data_root,
        text_root=text_root,
        max_frames=40,
        min_frames=1,
        random_crop=False,
        caption_policy="first_full_motion",
    )
    item = dataset[0]
    assert item["text"] == "full caption"
    assert item["caption_index"] == 1
    assert item["caption_line_number"] == 2
    assert item["caption_from_tag"] == 0.0
    assert item["caption_to_tag"] == 0.0
    assert item["caption_is_full_motion"] is True
