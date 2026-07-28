from __future__ import annotations

import torch

from models.raw_motion.kimodo_context_flow_dit import align_projected_source


def test_mixed_native_lengths_share_one_padded_batch_without_crosstalk():
    # Cases: equal, off-by-one, material duration change, Ts=1, Tt=1, absent.
    source_lengths = torch.tensor([[5], [4], [2], [1], [4], [0]])
    target_lengths = torch.tensor([5, 5, 5, 5, 1, 5])
    bsz, slots, source_frames, hidden, target_frames = 6, 1, 5, 2, 5
    source = torch.zeros(bsz, slots, source_frames, hidden)
    for b in range(bsz):
        for t in range(int(source_lengths[b, 0])):
            source[b, 0, t] = torch.tensor([100.0 * b + t, -100.0 * b - t])
    source_valid = torch.arange(source_frames)[None, None] < source_lengths[..., None]
    target_valid = torch.arange(target_frames)[None] < target_lengths[:, None]

    aligned, valid = align_projected_source(
        source,
        source_valid,
        source_lengths,
        target_valid,
        target_lengths,
    )
    assert torch.equal(aligned[0, 0], source[0, 0])
    assert torch.allclose(aligned[1, 0, :, 0], torch.tensor([100.0, 100.75, 101.5, 102.25, 103.0]))
    assert torch.allclose(aligned[2, 0, :, 0], torch.linspace(200.0, 201.0, 5))
    assert torch.equal(aligned[3, 0, :, 0], torch.full((5,), 300.0))
    assert aligned[4, 0, 0, 0].item() == 400.0
    assert torch.count_nonzero(aligned[4, 0, 1:]) == 0
    assert torch.count_nonzero(aligned[5]) == 0
    assert not valid[5].any()


def test_invalid_neighbor_is_excluded_and_weights_are_renormalized():
    source = torch.tensor([[[[0.0], [10.0], [20.0]]]])
    source_valid = torch.tensor([[[True, False, True]]])
    source_lengths = torch.tensor([[3]])
    target_valid = torch.ones(1, 5, dtype=torch.bool)
    target_lengths = torch.tensor([5])
    aligned, valid = align_projected_source(
        source,
        source_valid,
        source_lengths,
        target_valid,
        target_lengths,
    )
    assert aligned[0, 0, 1, 0].item() == 0.0
    assert not valid[0, 0, 2]
    assert aligned[0, 0, 3, 0].item() == 20.0

