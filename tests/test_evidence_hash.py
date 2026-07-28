from __future__ import annotations

from collections import OrderedDict

import torch

from models.raw_motion.evidence_hash import state_dict_sha256, tensor_sha256


def test_tensor_hash_supports_scalars_empty_and_noncontiguous_values() -> None:
    values = (
        torch.tensor(1, dtype=torch.int64),
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(True),
        torch.empty(0),
        torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1),
    )
    for value in values:
        digest = tensor_sha256(value)
        assert len(digest) == 64
        assert digest == tensor_sha256(value.clone())


def test_state_dict_hash_is_name_sorted_and_binds_dtype_shape_and_values() -> None:
    first = OrderedDict(
        (("weight", torch.arange(4.0)), ("counter", torch.tensor(3)))
    )
    reordered = OrderedDict(reversed(list(first.items())))
    assert state_dict_sha256(first) == state_dict_sha256(reordered)

    changed = OrderedDict(first)
    changed["counter"] = torch.tensor(4)
    assert state_dict_sha256(first) != state_dict_sha256(changed)
