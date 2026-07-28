from __future__ import annotations

import torch

from models.raw_motion.hy273_unified_edit_losses import (
    UnifiedEditLossWeights,
    compute_source_target_discrepancy_x0_loss,
    compute_unified_edit_loss,
)


def _inputs(frames: int = 5):
    target = torch.zeros(2, frames, 273)
    valid = torch.ones(2, frames, dtype=torch.bool)
    mask = torch.zeros(2, frames, 273, dtype=torch.bool)
    return target, valid, mask


def test_correct_instruction_is_preferred_without_source_target_alignment():
    target, valid, mask = _inputs()
    correct = target[..., :269].clone()
    shuffled = correct.clone()
    shuffled[..., 5:8] = 1.0
    loss = compute_unified_edit_loss(
        correct_x0_hat_cont=correct,
        shuffled_x0_hat_cont=shuffled,
        x0_target_norm=target,
        target_valid=valid,
        hard_mask=mask,
        weights=UnifiedEditLossWeights(
            target_x0_scale=0.0,
            hard_x0_scale=0.0,
            instruction_rank_scale=1.0,
            instruction_relative_margin=0.1,
        ),
    )
    assert loss.correct_distance == 0
    assert loss.shuffled_distance > 0
    assert loss.instruction_rank_raw == 0


def test_identical_wrong_predictions_activate_instruction_ranking():
    target, valid, mask = _inputs()
    prediction = target[..., :269].clone()
    prediction[..., :3] = 0.5
    loss = compute_unified_edit_loss(
        correct_x0_hat_cont=prediction,
        shuffled_x0_hat_cont=prediction.clone(),
        x0_target_norm=target,
        target_valid=valid,
        hard_mask=mask,
        weights=UnifiedEditLossWeights(
            target_x0_scale=0.0,
            hard_x0_scale=0.0,
            instruction_rank_scale=1.0,
            instruction_relative_margin=0.1,
        ),
    )
    assert loss.instruction_rank_raw > 0
    assert loss.instruction_rank_active_fraction == 1


def test_masked_softplus_keeps_same_source_assignment_gradient_alive():
    target, valid, mask = _inputs()
    correct = target[..., :269].clone()
    correct[..., :3] = 0.1
    correct.requires_grad_(True)
    sibling = correct.detach().clone()
    sibling[..., :3] = 1.0
    sibling.requires_grad_(True)

    hinge = compute_unified_edit_loss(
        correct_x0_hat_cont=correct,
        shuffled_x0_hat_cont=sibling,
        x0_target_norm=target,
        target_valid=valid,
        hard_mask=mask,
        weights=UnifiedEditLossWeights(
            target_x0_scale=0.0,
            hard_x0_scale=0.0,
            instruction_rank_scale=1.0,
            instruction_relative_margin=0.1,
        ),
        instruction_rank_mode="hinge",
        instruction_rank_sample_mask=torch.tensor([True, False]),
    )
    assert hinge.instruction_rank_raw == 0
    assert hinge.instruction_rank_eligible_fraction == 0.5

    softplus = compute_unified_edit_loss(
        correct_x0_hat_cont=correct,
        shuffled_x0_hat_cont=sibling,
        x0_target_norm=target,
        target_valid=valid,
        hard_mask=mask,
        weights=UnifiedEditLossWeights(
            target_x0_scale=0.0,
            hard_x0_scale=0.0,
            instruction_rank_scale=1.0,
            instruction_relative_margin=0.1,
        ),
        instruction_rank_mode="softplus",
        instruction_rank_temperature=0.01,
        instruction_rank_sample_mask=torch.tensor([True, False]),
    )
    assert softplus.instruction_rank_ce_raw > 0
    assert softplus.instruction_rank_raw < softplus.instruction_rank_ce_raw
    assert 0 < softplus.instruction_rank_slope < 1
    softplus.total.backward()
    assert correct.grad is not None and bool((correct.grad[0] != 0).any())
    assert sibling.grad is not None and bool((sibling.grad[0] != 0).any())
    assert not bool((correct.grad[1] != 0).any())
    assert not bool((sibling.grad[1] != 0).any())


def test_global_eligible_denominator_matches_single_process_global_mean():
    target = torch.zeros(4, 2, 273)
    correct = torch.zeros(4, 2, 269, requires_grad=True)
    sibling = torch.ones(4, 2, 269, requires_grad=True)
    valid = torch.ones(4, 2, dtype=torch.bool)
    hard_mask = torch.zeros_like(target, dtype=torch.bool)
    eligible = torch.tensor([True, False, True, True])
    weights = UnifiedEditLossWeights(
        target_x0_scale=0.0,
        hard_x0_scale=0.0,
        instruction_rank_scale=1.0,
        instruction_relative_margin=0.1,
    )

    reference = compute_unified_edit_loss(
        correct_x0_hat_cont=correct,
        shuffled_x0_hat_cont=sibling,
        x0_target_norm=target,
        target_valid=valid,
        hard_mask=hard_mask,
        weights=weights,
        instruction_rank_sample_mask=eligible,
    )
    reference_grads = torch.autograd.grad(reference.total, (correct, sibling))

    shard_grads = []
    per_rank_denominator = float(eligible.sum()) / 2.0
    for shard in (slice(0, 2), slice(2, 4)):
        shard_correct = correct.detach()[shard].requires_grad_(True)
        shard_sibling = sibling.detach()[shard].requires_grad_(True)
        loss = compute_unified_edit_loss(
            correct_x0_hat_cont=shard_correct,
            shuffled_x0_hat_cont=shard_sibling,
            x0_target_norm=target[shard],
            target_valid=valid[shard],
            hard_mask=hard_mask[shard],
            weights=weights,
            instruction_rank_sample_mask=eligible[shard],
            instruction_rank_denominator=per_rank_denominator,
        )
        shard_grads.append(
            torch.autograd.grad(loss.total, (shard_correct, shard_sibling))
        )

    # DDP averages rank gradients. Local parameter gradients therefore include
    # a 1/world factor relative to the concatenated input-gradient reference.
    for tensor_index in range(2):
        ddp_gradient = torch.cat(
            [grad_pair[tensor_index] / 2.0 for grad_pair in shard_grads], dim=0
        )
        torch.testing.assert_close(ddp_gradient, reference_grads[tensor_index])


def test_subunit_per_rank_denominator_preserves_one_global_eligible_row():
    target = torch.zeros(1, 2, 273)
    correct = torch.ones(1, 2, 269, requires_grad=True)
    sibling = torch.ones(1, 2, 269, requires_grad=True)
    valid = torch.ones(1, 2, dtype=torch.bool)
    hard_mask = torch.zeros_like(target, dtype=torch.bool)
    weights = UnifiedEditLossWeights(
        target_x0_scale=0.0,
        hard_x0_scale=0.0,
        instruction_rank_scale=1.0,
        instruction_relative_margin=0.1,
    )
    reference = compute_unified_edit_loss(
        correct_x0_hat_cont=correct,
        shuffled_x0_hat_cont=sibling,
        x0_target_norm=target,
        target_valid=valid,
        hard_mask=hard_mask,
        weights=weights,
        instruction_rank_sample_mask=torch.tensor([True]),
    )
    distributed = compute_unified_edit_loss(
        correct_x0_hat_cont=correct,
        shuffled_x0_hat_cont=sibling,
        x0_target_norm=target,
        target_valid=valid,
        hard_mask=hard_mask,
        weights=weights,
        instruction_rank_sample_mask=torch.tensor([True]),
        instruction_rank_denominator=0.25,
    )
    torch.testing.assert_close(distributed.total / 4.0, reference.total)


def test_discrepancy_global_denominator_matches_ddp_global_mean():
    target = torch.zeros(4, 2, 273)
    prediction = torch.stack(
        [torch.full((2, 269), value) for value in (0.25, 0.5, 0.75, 1.0)]
    ).requires_grad_(True)
    mask = torch.zeros(4, 2, 269, dtype=torch.bool)
    mask[0, :, :3] = True
    mask[2, :, 5:8] = True

    reference = compute_source_target_discrepancy_x0_loss(
        correct_x0_hat_cont=prediction,
        x0_target_norm=target,
        discrepancy_mask=mask,
        scale=0.2,
    )
    reference_grad = torch.autograd.grad(reference.total, prediction)[0]

    shard_losses = []
    shard_grads = []
    per_rank_denominator = float(mask.reshape(4, -1).any(dim=-1).sum()) / 2.0
    for shard in (slice(0, 2), slice(2, 4)):
        shard_prediction = prediction.detach()[shard].requires_grad_(True)
        shard_loss = compute_source_target_discrepancy_x0_loss(
            correct_x0_hat_cont=shard_prediction,
            x0_target_norm=target[shard],
            discrepancy_mask=mask[shard],
            scale=0.2,
            active_denominator=per_rank_denominator,
        )
        shard_losses.append(shard_loss.total.detach())
        shard_grads.append(torch.autograd.grad(shard_loss.total, shard_prediction)[0])

    torch.testing.assert_close(torch.stack(shard_losses).mean(), reference.total)
    ddp_gradient = torch.cat([gradient / 2.0 for gradient in shard_grads], dim=0)
    torch.testing.assert_close(ddp_gradient, reference_grad)


def test_discrepancy_subunit_denominator_preserves_one_global_active_row():
    target = torch.zeros(1, 2, 273)
    prediction = torch.ones(1, 2, 269, requires_grad=True)
    mask = torch.zeros(1, 2, 269, dtype=torch.bool)
    mask[..., :3] = True
    reference = compute_source_target_discrepancy_x0_loss(
        correct_x0_hat_cont=prediction,
        x0_target_norm=target,
        discrepancy_mask=mask,
        scale=0.2,
    )
    distributed = compute_source_target_discrepancy_x0_loss(
        correct_x0_hat_cont=prediction,
        x0_target_norm=target,
        discrepancy_mask=mask,
        scale=0.2,
        active_denominator=0.25,
    )
    torch.testing.assert_close(distributed.total / 4.0, reference.total)


def test_hard_target_loss_emphasizes_sparse_errors_and_respects_controls():
    target, valid, mask = _inputs()
    correct = target[..., :269].clone()
    correct[:, 0, 5] = 2.0
    loss = compute_unified_edit_loss(
        correct_x0_hat_cont=correct,
        shuffled_x0_hat_cont=correct.clone(),
        x0_target_norm=target,
        target_valid=valid,
        hard_mask=mask,
        weights=UnifiedEditLossWeights(instruction_rank_scale=0.0),
    )
    assert loss.hard_x0_raw > loss.target_x0_raw > 0

    masked = compute_unified_edit_loss(
        correct_x0_hat_cont=correct,
        shuffled_x0_hat_cont=correct.clone(),
        x0_target_norm=target,
        target_valid=valid,
        hard_mask=torch.ones_like(mask),
        weights=UnifiedEditLossWeights(instruction_rank_scale=0.0),
    )
    assert masked.total == 0
