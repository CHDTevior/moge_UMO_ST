from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from models.raw_motion.hytext_cache import (
    CachedHYTextEncoder,
    HYTextMemmapCache,
    PROFILE_CACHE_FORMAT,
    hytext_profile_key,
)


ENCODER_ID = "unit-test-encoder"
PROMPT_VERSION = "unit-test-prompts-v1"
ABSOLUTE = "hytext_absolute_motion_v1"
RELATIVE = "hytext_relative_edit_v1"


def _build_cache(root: Path) -> None:
    shard = root / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    ctxt = np.zeros((3, 4, 6), dtype=np.float32)
    ctxt[0] = 1.0
    ctxt[1] = 2.0
    ctxt[2] = 3.0
    vtxt = np.zeros((3, 1, 5), dtype=np.float32)
    vtxt[0] = 10.0
    vtxt[1] = 20.0
    vtxt[2] = 30.0
    np.save(shard / "ctxt.npy", ctxt)
    np.save(shard / "vtxt.npy", vtxt)
    np.save(shard / "ctxt_len.npy", np.array([1, 2, 3], dtype=np.int16))
    rows = [(ABSOLUTE, ""), (RELATIVE, ""), (RELATIVE, "move faster")]
    index = {}
    for row, (profile, text) in enumerate(rows):
        key = hytext_profile_key(
            text,
            profile,
            encoder_identity=ENCODER_ID,
            prompt_template_version=PROMPT_VERSION,
        )
        index[key] = {
            "shard": "shard_00000",
            "row": row,
            "text": text,
            "profile": profile,
        }
    (root / "index.json").write_text(json.dumps(index))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": PROFILE_CACHE_FORMAT,
                "encoder_identity": ENCODER_ID,
                "prompt_template_version": PROMPT_VERSION,
                "ctxt_dim": 6,
                "vtxt_dim": 5,
                "max_length_llm": 4,
            }
        )
    )


def test_profile_keys_and_empty_rows_are_distinct(tmp_path: Path) -> None:
    _build_cache(tmp_path)
    cache = HYTextMemmapCache(tmp_path)
    vtxt, ctxt, lengths = cache.lookup_rows(["", ""], [ABSOLUTE, RELATIVE])
    assert not torch.equal(ctxt[0], ctxt[1])
    assert not torch.equal(vtxt[0], vtxt[1])
    assert lengths.tolist() == [1, 2]


def test_relative_dropout_uses_relative_empty(tmp_path: Path) -> None:
    _build_cache(tmp_path)
    encoder = CachedHYTextEncoder(
        hidden_dim=8,
        cache_dir=tmp_path,
        max_text_tokens=4,
        ctxt_dim=6,
        vtxt_dim=5,
    )
    captured = {}
    original = encoder.cache.lookup_rows

    def lookup(texts, profiles=None):
        captured["texts"] = list(texts)
        captured["profiles"] = list(profiles or [])
        return original(captured["texts"], captured["profiles"])

    encoder.cache.lookup_rows = lookup  # type: ignore[method-assign]
    encoder(
        ["move faster"],
        profiles=[RELATIVE],
        device=torch.device("cpu"),
        dtype=torch.float32,
        force_drop=True,
    )
    assert captured == {"texts": [""], "profiles": [RELATIVE]}


def test_profile_cache_fails_closed_without_profile(tmp_path: Path) -> None:
    _build_cache(tmp_path)
    cache = HYTextMemmapCache(tmp_path)
    with pytest.raises(ValueError, match="requires one profile"):
        cache.lookup_rows(["move faster"])


def test_runtime_rows_handle_cache_miss_without_mutating_index(tmp_path: Path) -> None:
    _build_cache(tmp_path)
    cache = HYTextMemmapCache(tmp_path)
    text = "a dancer pivots, then crouches"
    runtime_vtxt = torch.full((1, 1, 5), 41.0)
    runtime_ctxt = torch.full((1, 4, 6), 17.0)
    runtime_lengths = torch.tensor([4])

    assert cache.text_source(text, ABSOLUTE) is None
    cache.add_runtime_rows(
        [text],
        runtime_vtxt,
        runtime_ctxt,
        runtime_lengths,
        profiles=[ABSOLUTE],
    )

    assert cache.text_source(text, ABSOLUTE) == "runtime_encoder"
    assert len(cache.index) == 3
    vtxt, ctxt, lengths = cache.lookup_rows([text], [ABSOLUTE])
    torch.testing.assert_close(vtxt, runtime_vtxt)
    torch.testing.assert_close(ctxt, runtime_ctxt)
    assert lengths.tolist() == [4]
