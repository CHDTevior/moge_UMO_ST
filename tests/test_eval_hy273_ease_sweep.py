import numpy as np
import pytest
import torch

from models.raw_motion.hy273_ease import HY273EaseNormalizer
from tools.eval_hy273_ease_sweep import (
    build_sweep_requests,
    spearman_correlation,
)


def _normalizer() -> HY273EaseNormalizer:
    return HY273EaseNormalizer(
        mean=torch.tensor([0.0, 0.1, 0.0, 0.0, -0.2, 0.0]),
        std=torch.tensor([0.5, 0.25, 0.5, 0.75, 0.4, 0.75]),
    )


def test_spearman_handles_monotonic_and_tied_values() -> None:
    assert spearman_correlation([0, 1, 2], [3, 4, 5]) == pytest.approx(1.0)
    assert spearman_correlation([0, 1, 2], [5, 4, 3]) == pytest.approx(-1.0)
    assert np.isfinite(spearman_correlation([0, 1, 2], [4, 4, 5]))
    assert spearman_correlation([0, 1, 2], [4, 4, 4]) is None


def test_build_sweep_requests_isolates_each_normalized_half() -> None:
    normalizer = _normalizer()
    target_normalized = torch.tensor([1.0, -2.0, 0.5, -1.0, 0.25, 2.0])
    target_physical = normalizer.denormalize(target_normalized)
    rows, physical, present, normalized = build_sweep_requests(
        normalizer,
        target_physical,
        [0.0, 0.5, 1.0],
    )

    assert len(rows) == 7
    assert rows[0]["axis"] == "absent"
    assert not bool(present[0])
    assert torch.equal(physical[0], torch.zeros(6))
    assert torch.allclose(normalizer.normalize(physical[0]), normalized[0])
    assert bool(present[1:].all())
    assert torch.allclose(normalized[1, :3], torch.zeros(3))
    assert torch.allclose(normalized[1, 3:], torch.zeros(3))
    assert torch.allclose(normalized[3, :3], target_normalized[:3])
    assert torch.allclose(normalized[3, 3:], torch.zeros(3))
    assert torch.allclose(normalized[6, :3], torch.zeros(3))
    assert torch.allclose(normalized[6, 3:], target_normalized[3:])
