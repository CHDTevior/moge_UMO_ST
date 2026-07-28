#!/usr/bin/env python
"""Real-asset capability smoke and fixed 32-sample overfit gate for R11."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.hy273_multitask_manifest_dataset import (
    HY273MultitaskManifestDataset,
    build_global_sample_plans,
    collate_hy273_multitask,
)
from data.hy273_multitask_scheduler import EditConditionPattern
from models.raw_motion.hy273_multitask_condition import (
    CapabilityId,
    ConditionBatch,
    TaskId,
    TrainStream,
)
from models.raw_motion.hy273_normalizer import HY273Normalizer
from models.raw_motion.hy273_slices import CONTACT_SLICE, CONT_DIM
from train_hy273_multitask import (
    apply_optimizer_phase,
    assert_and_mask_context_gradients,
    build_hard_controls,
    build_stateless_flow_inputs,
    cfg_get,
    create_model,
    load_config,
    optimizer_groups,
    seed_model_initialization,
    validate_assets,
    validate_frozen_contract,
)
from models.raw_motion.flow_schedule import build_flow_state
from models.raw_motion.hy273_multitask_losses import compute_hy273_multitask_loss


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    stream: TrainStream
    capability: CapabilityId
    edit_patterns: tuple[EditConditionPattern, ...]
    row_offset: int


CAPABILITIES = (
    CapabilitySpec("t2m", TrainStream.HML_MIXED, CapabilityId.T2M, (), 0),
    CapabilitySpec(
        "control",
        TrainStream.HML_MIXED,
        CapabilityId.KIMODO_CONTROL,
        (),
        8,
    ),
    CapabilitySpec(
        "edit",
        TrainStream.MOTION_EDIT,
        CapabilityId.MOTION_EDIT,
        (
            EditConditionPattern.SOURCE_TEXT,
            EditConditionPattern.SOURCE_ONLY,
        ),
        0,
    ),
    CapabilitySpec(
        "edit_control",
        TrainStream.MOTION_EDIT,
        CapabilityId.MOTION_EDIT_CONTROL,
        (
            EditConditionPattern.SOURCE_TEXT_CONTROL,
            EditConditionPattern.SOURCE_CONTROL,
        ),
        8,
    ),
)

CONTROL_GATE_MODES = (
    "root_sparse",
    "root_dense",
    "endpoints",
    "fullpose",
    "contact",
    "mixed",
    "mixed",
    "mixed",
)


def _materialize_batch(
    dataset: HY273MultitaskManifestDataset,
    spec: CapabilitySpec,
    *,
    sample_count: int,
    run_seed: int,
    global_ordinal: int,
) -> dict[str, Any]:
    row_indices = [
        (spec.row_offset + index) % len(dataset) for index in range(sample_count)
    ]
    plans = build_global_sample_plans(
        dataset=dataset,
        row_indices=row_indices,
        global_step=400_000,
        first_global_ordinal=global_ordinal,
        run_seed=run_seed,
    )
    forced = []
    for index, plan in enumerate(plans):
        pattern = (
            None
            if not spec.edit_patterns
            else spec.edit_patterns[index % len(spec.edit_patterns)]
        )
        forced.append(
            replace(
                plan,
                capability_id=spec.capability,
                text_drop=False if pattern is None else not pattern.uses_text,
                edit_pattern=pattern,
            )
        )
    return collate_hy273_multitask([dataset.materialize(plan) for plan in forced])


def _prepare(
    batch: dict[str, Any],
    *,
    device: torch.device,
    normalizer: HY273Normalizer,
    config: dict[str, Any],
    manifest_sha256: str,
    run_seed: int,
    forced_control_modes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    target_physical = batch["target_motion"].to(device=device, dtype=torch.float32)
    condition = batch["condition"].to(device)
    condition.validate(max_target_frames=int(cfg_get(config, "data.max_target_frames")))
    observed_physical, hard_mask, control_modes = build_hard_controls(
        target_physical=target_physical,
        condition=condition,
        plans=batch["plans"],
        global_step=400_000,
        config=config,
        manifest_sha256=manifest_sha256,
        run_seed=run_seed,
        forced_mode_schedule=forced_control_modes,
    )
    x0_norm = normalizer.normalize(target_physical)
    observed_norm = normalizer.normalize(observed_physical)
    timesteps, continuous_noise, contact_aux = build_stateless_flow_inputs(
        plans=batch["plans"],
        x0_norm=x0_norm,
        manifest_sha256=manifest_sha256,
        run_seed=run_seed,
        config=config,
    )
    flow_state = build_flow_state(
        x0_norm,
        observed_norm,
        hard_mask,
        timesteps,
        noise_cont=continuous_noise,
        contact_aux=contact_aux,
    )
    return {
        "target_physical": target_physical,
        "condition": condition,
        "observed_norm": observed_norm,
        "hard_mask": hard_mask,
        "control_modes": control_modes,
        "x0_norm": x0_norm,
        "timesteps": timesteps,
        "flow_state": flow_state,
        "continuous_noise": continuous_noise,
        "contact_aux": contact_aux,
        "texts": batch["texts"],
        "uids": batch["uids"],
        "edit_patterns": [
            None if plan.edit_pattern is None else plan.edit_pattern.name
            for plan in batch["plans"]
        ],
    }


def _forward_prediction(
    model: torch.nn.Module,
    prepared: dict[str, Any],
    *,
    autocast_enabled: bool,
    condition: ConditionBatch | None = None,
    texts: list[str] | None = None,
    flow_state: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    condition = prepared["condition"] if condition is None else condition
    texts = prepared["texts"] if texts is None else texts
    flow_state = prepared["flow_state"] if flow_state is None else flow_state
    with torch.autocast(
        device_type=prepared["x0_norm"].device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        prediction = model(
            flow_state["model_in"],
            t=prepared["timesteps"],
            c_dir=condition.frame_gauge_dir,
            text=texts,
            length_mask=condition.target_valid,
            x_self_cond=None,
            text_drop_prob=0.0,
            condition=condition,
        )
    return prediction


def _forward_loss(
    model: torch.nn.Module,
    prepared: dict[str, Any],
    *,
    normalizer: HY273Normalizer,
    weights: Any,
    autocast_enabled: bool,
):
    condition = prepared["condition"]
    prediction = _forward_prediction(
        model,
        prepared,
        autocast_enabled=autocast_enabled,
    )
    with torch.autocast(
        device_type=prepared["x0_norm"].device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        bundle = compute_hy273_multitask_loss(
            x0_hat_cont=prediction[..., :CONT_DIM],
            contact_logits=prediction[..., CONTACT_SLICE],
            z_cont_imputed=prepared["flow_state"]["z_cont_imp"],
            x0_target_norm=prepared["x0_norm"],
            x0_target_physical=prepared["target_physical"],
            hard_observed_norm=prepared["observed_norm"],
            hard_mask=prepared["hard_mask"],
            target_valid=condition.target_valid,
            timesteps=prepared["timesteps"],
            normalizer=normalizer,
            global_step=400_000,
            weights=weights,
        )
    return bundle


def _target_probe(prediction: torch.Tensor, prepared: dict[str, Any]) -> float:
    valid = prepared["condition"].target_valid[..., None]
    cont_mask = valid.expand_as(prediction[..., :CONT_DIM])
    contact_mask = valid.expand_as(prediction[..., CONTACT_SLICE])
    cont = (prediction[..., :CONT_DIM] - prepared["x0_norm"][..., :CONT_DIM]).square()
    contact = F.binary_cross_entropy_with_logits(
        prediction[..., CONTACT_SLICE],
        prepared["x0_norm"][..., CONTACT_SLICE],
        reduction="none",
    )
    value = cont.masked_select(cont_mask).mean() + 0.1 * contact.masked_select(
        contact_mask
    ).mean()
    return float(value.detach().float().item())


def _output_sensitivity(
    reference: torch.Tensor,
    altered: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    cont_mask = valid[..., None].expand_as(reference[..., :CONT_DIM])
    contact_mask = valid[..., None].expand_as(reference[..., CONTACT_SLICE])
    cont = (reference[..., :CONT_DIM] - altered[..., :CONT_DIM]).square()
    contact = (
        torch.sigmoid(reference[..., CONTACT_SLICE])
        - torch.sigmoid(altered[..., CONTACT_SLICE])
    ).square()
    return float(
        (
            cont.masked_select(cont_mask).mean()
            + contact.masked_select(contact_mask).mean()
        )
        .detach()
        .float()
        .item()
    )


def _rolled_source_condition(condition: ConditionBatch) -> ConditionBatch:
    kwargs = {}
    for name in (
        "source_motion",
        "source_present",
        "source_time_valid",
        "source_value_mask",
        "source_role_id",
        "source_native_lengths",
    ):
        kwargs[name] = torch.roll(getattr(condition, name), shifts=1, dims=0)
    if condition.target_to_source_time_map is not None:
        kwargs["target_to_source_time_map"] = torch.roll(
            condition.target_to_source_time_map, shifts=1, dims=0
        )
    altered = replace(condition, **kwargs)
    altered.validate()
    return altered


def _zero_source_condition(condition: ConditionBatch) -> ConditionBatch:
    altered = replace(condition, source_motion=torch.zeros_like(condition.source_motion))
    altered.validate()
    return altered


@torch.no_grad()
def _counterfactual_report(
    model: torch.nn.Module,
    spec: CapabilitySpec,
    prepared: dict[str, Any],
    *,
    autocast_enabled: bool,
    minimum_sensitivity: float,
) -> tuple[dict[str, Any], bool]:
    reference = _forward_prediction(
        model, prepared, autocast_enabled=autocast_enabled
    )
    reference_probe = _target_probe(reference, prepared)
    cases: dict[str, tuple[ConditionBatch | None, list[str] | None, dict[str, torch.Tensor] | None]] = {}
    if spec.capability == CapabilityId.T2M or spec.stream == TrainStream.MOTION_EDIT:
        texts = list(prepared["texts"])
        cases["text_shuffle"] = (None, texts[-1:] + texts[:-1], None)
        cases["text_zero"] = (None, [""] * len(texts), None)
    if spec.stream == TrainStream.MOTION_EDIT:
        cases["source_shuffle"] = (
            _rolled_source_condition(prepared["condition"]),
            None,
            None,
        )
        cases["source_zero"] = (
            _zero_source_condition(prepared["condition"]),
            None,
            None,
        )
    if spec.capability in {
        CapabilityId.KIMODO_CONTROL,
        CapabilityId.MOTION_EDIT_CONTROL,
    }:
        wrong_observed = torch.roll(prepared["x0_norm"], shifts=1, dims=0)
        wrong_flow = build_flow_state(
            prepared["x0_norm"],
            wrong_observed,
            prepared["hard_mask"],
            prepared["timesteps"],
            noise_cont=prepared["continuous_noise"],
            contact_aux=prepared["contact_aux"],
        )
        cases["control_shuffle"] = (None, None, wrong_flow)

    report = {}
    passed = True
    for name, (condition, texts, flow_state) in cases.items():
        altered = _forward_prediction(
            model,
            prepared,
            autocast_enabled=autocast_enabled,
            condition=condition,
            texts=texts,
            flow_state=flow_state,
        )
        sensitivity = _output_sensitivity(
            reference, altered, prepared["condition"].target_valid
        )
        altered_probe = _target_probe(altered, prepared)
        case_passed = math.isfinite(sensitivity) and sensitivity >= minimum_sensitivity
        report[name] = {
            "output_sensitivity_mse": sensitivity,
            "minimum_sensitivity_mse": minimum_sensitivity,
            "reference_target_probe": reference_probe,
            "altered_target_probe": altered_probe,
            "target_probe_ratio": altered_probe / max(reference_probe, 1e-12),
            "passed": case_passed,
        }
        passed = passed and case_passed
    return report, passed


def _loss_report(bundle) -> dict[str, Any]:
    total = float(bundle.total.detach().float().item())
    terms = {}
    groups: dict[str, float] = {}
    for name, term in bundle.terms.items():
        raw = float(term.raw.detach().float().item())
        weighted = float(term.weighted.detach().float().item())
        terms[name] = {
            "raw": raw,
            "weighted": weighted,
            "denominator": float(term.denominator.detach().float().item()),
            "percent_total": 100.0 * weighted / max(total, 1e-12),
        }
        groups[term.group] = groups.get(term.group, 0.0) + weighted
    return {
        "total": total,
        "groups": {
            name: {
                "weighted": value,
                "percent_total": 100.0 * value / max(total, 1e-12),
            }
            for name, value in sorted(groups.items())
        },
        "terms": terms,
        "fk_warmup_factor": float(bundle.fk_warmup_factor),
        "fk_distance_cm": float(bundle.fk_distance_cm.raw.detach().float().item()),
    }


def _assert_capability_contract(
    spec: CapabilitySpec,
    prepared: dict[str, Any],
    bundle,
) -> None:
    source_present = bool(prepared["condition"].source_present.any().item())
    mask_present = bool(prepared["hard_mask"].any().item())
    expects_source = spec.stream == TrainStream.MOTION_EDIT
    expects_control = spec.capability in {
        CapabilityId.KIMODO_CONTROL,
        CapabilityId.MOTION_EDIT_CONTROL,
    }
    if source_present != expects_source:
        raise RuntimeError(f"{spec.name}: source presence contract failed")
    if mask_present != expects_control:
        raise RuntimeError(f"{spec.name}: hard-control presence contract failed")
    continuous_denominator = float(
        bundle.terms["control_continuous"].denominator.detach().item()
    )
    contact_denominator = float(
        bundle.terms["control_contact"].denominator.detach().item()
    )
    if expects_control:
        if continuous_denominator <= 0.0 or contact_denominator <= 0.0:
            raise RuntimeError(
                f"{spec.name}: continuous/contact control denominators must both be nonzero"
            )
        realized = set(prepared["control_modes"])
        required = set(CONTROL_GATE_MODES[:5])
        if not required.issubset(realized):
            raise RuntimeError(
                f"{spec.name}: fixed Kimodo mode coverage failed: {sorted(realized)}"
            )
    elif continuous_denominator != 0.0 or contact_denominator != 0.0:
        raise RuntimeError(f"{spec.name}: no-control loss denominator must be zero")
    if spec.edit_patterns:
        actual_patterns = {value for value in prepared["edit_patterns"] if value is not None}
        expected_patterns = {value.name for value in spec.edit_patterns}
        if actual_patterns != expected_patterns:
            raise RuntimeError(
                f"{spec.name}: edit condition-pattern coverage failed: {actual_patterns}"
            )
    if not bool(torch.isfinite(bundle.total.detach())):
        raise RuntimeError(f"{spec.name}: non-finite total loss")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hy273_multitask_stage_a_t2m.yaml")
    parser.add_argument("--mode", choices=("smoke", "overfit"), default="smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--samples_per_capability", type=int, default=8)
    parser.add_argument("--overfit_steps", type=int, default=400)
    parser.add_argument("--required_reduction", type=float, default=0.10)
    parser.add_argument("--required_term_reduction", type=float, default=0.02)
    parser.add_argument("--minimum_counterfactual_sensitivity", type=float, default=1e-7)
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    weights = validate_frozen_contract(config)
    asset_identity = validate_assets(config)
    run_seed = int(cfg_get(config, "training.seed"))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The real-model capability gate requires CUDA")
    torch.cuda.set_device(device)
    stats_root = Path(cfg_get(config, "data.stats_root"))
    normalizer = HY273Normalizer.from_data_root(
        stats_root / "full",
        stats_dir=stats_root / "full",
        variance_eps=float(cfg_get(config, "model.stats_variance_eps")),
    ).to(device)
    autocast_enabled = str(cfg_get(config, "training.precision")) == "bf16"
    datasets = {
        stream: HY273MultitaskManifestDataset(
            cfg_get(config, "data.train_manifest"), stream
        )
        for stream in TrainStream
    }
    requested_sample_count = 2 if args.mode == "smoke" else int(args.samples_per_capability)
    if requested_sample_count < 1:
        raise ValueError("samples_per_capability must be positive")
    if args.mode == "overfit" and requested_sample_count < len(CONTROL_GATE_MODES):
        raise ValueError(
            f"overfit gate requires at least {len(CONTROL_GATE_MODES)} samples per capability"
        )
    prepared = {}
    sample_counts = {}
    next_ordinal = 0
    for spec_index, spec in enumerate(CAPABILITIES):
        sample_count = (
            max(requested_sample_count, 5)
            if spec.capability
            in {CapabilityId.KIMODO_CONTROL, CapabilityId.MOTION_EDIT_CONTROL}
            else max(requested_sample_count, len(spec.edit_patterns) or 1)
        )
        sample_counts[spec.name] = sample_count
        dataset = datasets[spec.stream]
        batch = _materialize_batch(
            dataset,
            spec,
            sample_count=sample_count,
            run_seed=run_seed,
            global_ordinal=next_ordinal,
        )
        next_ordinal += sample_count
        forced_control_modes = None
        if spec.capability in {
            CapabilityId.KIMODO_CONTROL,
            CapabilityId.MOTION_EDIT_CONTROL,
        }:
            forced_control_modes = tuple(
                CONTROL_GATE_MODES[index % len(CONTROL_GATE_MODES)]
                for index in range(sample_count)
            )
        prepared[spec.name] = _prepare(
            batch,
            device=device,
            normalizer=normalizer,
            config=config,
            manifest_sha256=dataset.manifest_sha256,
            run_seed=run_seed,
            forced_control_modes=forced_control_modes,
        )

    started = time.perf_counter()
    before: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    after: dict[str, Any] = {}
    passed = True
    reductions: dict[str, float] = {}
    term_reductions: dict[str, dict[str, float]] = {}
    context_update_norms: dict[str, float] = {}
    counterfactuals: dict[str, Any] = {}
    peak_gib = 0.0

    for spec_index, spec in enumerate(CAPABILITIES):
        seed_model_initialization(run_seed + spec_index)
        torch.cuda.reset_peak_memory_stats(device)
        model = create_model(config).to(device)
        model.train()
        model.zero_grad(set_to_none=True)
        bundle = _forward_loss(
            model,
            prepared[spec.name],
            normalizer=normalizer,
            weights=weights,
            autocast_enabled=autocast_enabled,
        )
        _assert_capability_contract(spec, prepared[spec.name], bundle)
        before[spec.name] = _loss_report(bundle)

        if args.mode == "smoke":
            bundle.total.backward()
            source_present = bool(
                prepared[spec.name]["condition"].source_present.any().item()
            )
            context_grad_sq = sum(
                float(parameter.grad.float().square().sum().item())
                for parameter in (
                    *model.context_weight_parameters(),
                    *model.context_bias_parameters(),
                )
                if parameter.grad is not None
            )
            if source_present and context_grad_sq <= 0.0:
                raise RuntimeError(f"{spec.name}: source context received no gradient")
            if not source_present and context_grad_sq != 0.0:
                raise RuntimeError(f"{spec.name}: absent context gradient was not exact zero")
            before[spec.name]["context_grad_norm"] = math.sqrt(context_grad_sq)
            after[spec.name] = before[spec.name]
            context_update_norms[spec.name] = 0.0
            peak_gib = max(
                peak_gib, torch.cuda.max_memory_allocated(device) / (1024**3)
            )
            del bundle, model
            torch.cuda.empty_cache()
            continue

        groups, _ = optimizer_groups(model, 400_000)
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        initial_context = [
            parameter.detach().clone()
            for parameter in (
                *model.context_weight_parameters(),
                *model.context_bias_parameters(),
            )
        ]
        for update in range(int(args.overfit_steps)):
            apply_optimizer_phase(optimizer, 400_000)
            optimizer.zero_grad(set_to_none=True)
            bundle = _forward_loss(
                model,
                prepared[spec.name],
                normalizer=normalizer,
                weights=weights,
                autocast_enabled=autocast_enabled,
            )
            _assert_capability_contract(spec, prepared[spec.name], bundle)
            bundle.total.backward()
            condition = prepared[spec.name]["condition"]
            context_active = bool(
                condition.source_present.any().item()
                or (condition.task_id == int(TaskId.EDIT)).any().item()
            )
            assert_and_mask_context_gradients(
                model,
                context_active=context_active,
                global_step=400_000,
                optimizer=optimizer,
            )
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg_get(config, "training.gradient_clip")),
                ).item()
            )
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient at update {update}")
            optimizer.step()
            if update == 0 or (update + 1) % 20 == 0 or update + 1 == args.overfit_steps:
                history.append(
                    {
                        "update": update + 1,
                        "capability": spec.name,
                        "loss": float(bundle.total.detach().float().item()),
                        "grad_norm_preclip": grad_norm,
                    }
                )

        model.eval()
        with torch.no_grad():
            bundle = _forward_loss(
                model,
                prepared[spec.name],
                normalizer=normalizer,
                weights=weights,
                autocast_enabled=autocast_enabled,
            )
            _assert_capability_contract(spec, prepared[spec.name], bundle)
            after[spec.name] = _loss_report(bundle)
        threshold = 1.0 - float(args.required_reduction)
        initial = before[spec.name]["total"]
        final = after[spec.name]["total"]
        reduction = 1.0 - final / max(initial, 1e-12)
        reductions[spec.name] = reduction
        passed = passed and final <= initial * threshold

        required_terms = ["contact_all"]
        if spec.capability in {
            CapabilityId.KIMODO_CONTROL,
            CapabilityId.MOTION_EDIT_CONTROL,
        }:
            required_terms.extend(("control_continuous", "control_contact"))
        term_reductions[spec.name] = {}
        for name in required_terms:
            initial_term = before[spec.name]["terms"][name]["raw"]
            final_term = after[spec.name]["terms"][name]["raw"]
            if initial_term <= 0.0:
                raise RuntimeError(f"{spec.name}: required term {name} was not exercised")
            term_reduction = 1.0 - final_term / initial_term
            term_reductions[spec.name][name] = term_reduction
            passed = passed and term_reduction >= float(args.required_term_reduction)

        context_update_sq = 0.0
        for parameter, initial in zip(
            (*model.context_weight_parameters(), *model.context_bias_parameters()),
            initial_context,
        ):
            context_update_sq += float(
                (parameter.detach() - initial).float().square().sum().item()
            )
        context_update_norm = math.sqrt(context_update_sq)
        context_update_norms[spec.name] = context_update_norm
        expects_source = spec.stream == TrainStream.MOTION_EDIT
        passed = passed and ((context_update_norm > 0.0) == expects_source)
        report, counterfactual_passed = _counterfactual_report(
            model,
            spec,
            prepared[spec.name],
            autocast_enabled=autocast_enabled,
            minimum_sensitivity=float(args.minimum_counterfactual_sensitivity),
        )
        counterfactuals[spec.name] = report
        passed = passed and counterfactual_passed
        peak_gib = max(
            peak_gib, torch.cuda.max_memory_allocated(device) / (1024**3)
        )
        del bundle, optimizer, model
        torch.cuda.empty_cache()

    payload = {
        "format": "hy273_multitask_capability_gate_v1",
        "mode": args.mode,
        "passed": bool(passed),
        "config_path": str(config_path),
        "asset_identity": asset_identity,
        "device": str(device),
        "samples_per_capability": sample_counts,
        "total_fixed_samples": sum(sample_counts.values()),
        "overfit_steps_per_fresh_model": (
            int(args.overfit_steps) if args.mode == "overfit" else 0
        ),
        "required_reduction": float(args.required_reduction),
        "required_term_reduction": float(args.required_term_reduction),
        "minimum_counterfactual_sensitivity": float(
            args.minimum_counterfactual_sensitivity
        ),
        "fresh_model_per_capability": True,
        "loss_before": before,
        "loss_after": after,
        "total_loss_reduction": reductions,
        "term_loss_reduction": term_reductions,
        "context_update_norm": context_update_norms,
        "counterfactuals": counterfactuals,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "max_allocated_gib": peak_gib,
        "uids": {name: value["uids"] for name, value in prepared.items()},
        "control_modes": {
            name: value["control_modes"] for name, value in prepared.items()
        },
        "edit_patterns": {
            name: value["edit_patterns"] for name, value in prepared.items()
        },
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
