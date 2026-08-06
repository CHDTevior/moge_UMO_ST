"""Deterministic stage, stream, and per-sample plans for HY273 multitask training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Sequence

from models.raw_motion.hy273_multitask_condition import CapabilityId, TrainStream


PROB_SCALE = 5_000_000
HIGH_LEVEL_SCHEDULE_VERSION = "hy273_multitask_weighted_deficit_v1"
R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION = (
    "hy273_multitask_r16_stage_b_fixed_10_90_0_v1"
)
KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION = (
    "hy273_kencoder_stage_be_fixed_60_0_40_v1"
)
KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION = (
    "hy273_kencoder_stage_bc_fixed_10_70_20_ease_v1"
)
STAGE_C_SAFE_MIX_SCHEDULE_VERSION = (
    "hy273_multitask_weighted_deficit_stagec_fixed_18_80_02_v1"
)
STAGE_C_EDIT20_SCHEDULE_VERSION = (
    "hy273_multitask_weighted_deficit_stagec_fixed_35_45_20_v1"
)
STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION = (
    "hy273_multitask_weighted_deficit_stage_d_edit_cfg_then_30pct_v2"
)
CONTEXT_ONLY_EDIT_SCHEDULE_VERSION = (
    "hy273_multitask_context_only_edit_identity_cfg_v2"
)
UNIFIED_EDIT_V2_SCHEDULE_VERSION = (
    "hy273_multitask_unified_fixed_30_40_30_joint_edit_v1"
)
UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION = (
    "hy273_multitask_unified_fixed_30_30_40_joint_edit_v1"
)
UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION = (
    "hy273_multitask_unified_fixed_30_40_30_decomposed_cfg_v1"
)
UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION = (
    "hy273_multitask_unified_fixed_10_10_80_decomposed_cfg_v1"
)
DECOMPOSED_CFG_EDIT_SCHEDULE_VERSIONS = {
    KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
    KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION,
    UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION,
    UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION,
}
JOINT_ONLY_UNIFIED_EDIT_SCHEDULE_VERSIONS = {
    UNIFIED_EDIT_V2_SCHEDULE_VERSION,
    UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION,
}
UNIFIED_EDIT_SCHEDULE_VERSIONS = {
    KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
    KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION,
    *JOINT_ONLY_UNIFIED_EDIT_SCHEDULE_VERSIONS,
    UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION,
    UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION,
}
SUPPORTED_HIGH_LEVEL_SCHEDULES = {
    HIGH_LEVEL_SCHEDULE_VERSION,
    R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION,
    STAGE_C_SAFE_MIX_SCHEDULE_VERSION,
    STAGE_C_EDIT20_SCHEDULE_VERSION,
    STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION,
    CONTEXT_ONLY_EDIT_SCHEDULE_VERSION,
    KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION,
    *UNIFIED_EDIT_SCHEDULE_VERSIONS,
}
HML_INNER_SCHEDULE_VERSION = "hy273_hml_stateless_bernoulli_v1"
SAMPLE_RNG_VERSION = "hy273_multitask_sample_key_blake2b_v1"
BUCKET_PLAN_VERSION = "hy273_multitask_plan_first_actual_length_v1"


class TrainingPhase(IntEnum):
    STAGE_A = 0
    STAGE_B1 = 1
    STAGE_B2 = 2
    STAGE_C = 3


class EditConditionPattern(IntEnum):
    SOURCE_TEXT = 0
    SOURCE_ONLY = 1
    SOURCE_TEXT_CONTROL = 2
    SOURCE_CONTROL = 3
    TEXT_ONLY = 4
    UNCONDITIONAL = 5
    SOURCE_IDENTITY = 6

    @property
    def uses_source(self) -> bool:
        return self in {
            self.SOURCE_TEXT,
            self.SOURCE_ONLY,
            self.SOURCE_TEXT_CONTROL,
            self.SOURCE_CONTROL,
            self.SOURCE_IDENTITY,
        }

    @property
    def uses_text(self) -> bool:
        return self in {self.SOURCE_TEXT, self.SOURCE_TEXT_CONTROL, self.TEXT_ONLY}

    @property
    def uses_control(self) -> bool:
        return self in {self.SOURCE_TEXT_CONTROL, self.SOURCE_CONTROL}


@dataclass(frozen=True)
class PhaseProbabilityUnits:
    t2m: int
    control: int
    edit: int

    def __post_init__(self) -> None:
        if min(self.t2m, self.control, self.edit) < 0:
            raise ValueError("Probability units must be non-negative")
        if self.t2m + self.control + self.edit != PROB_SCALE:
            raise ValueError("Capability probability units must sum to PROB_SCALE")

    @property
    def hml(self) -> int:
        return self.t2m + self.control


def phase_for_step(step: int) -> TrainingPhase:
    step = int(step)
    if step < 0:
        raise ValueError("Global step cannot be negative")
    if step < 200_000:
        return TrainingPhase.STAGE_A
    if step < 250_000:
        return TrainingPhase.STAGE_B1
    if step < 400_000:
        return TrainingPhase.STAGE_B2
    return TrainingPhase.STAGE_C


def probability_units_for_step(
    step: int,
    schedule_version: str = HIGH_LEVEL_SCHEDULE_VERSION,
) -> PhaseProbabilityUnits:
    """Return exact capability probabilities for a versioned schedule."""

    if schedule_version not in SUPPORTED_HIGH_LEVEL_SCHEDULES:
        raise ValueError(f"Unknown high-level schedule: {schedule_version!r}")

    if schedule_version == CONTEXT_ONLY_EDIT_SCHEDULE_VERSION:
        return PhaseProbabilityUnits(0, 0, PROB_SCALE)
    if schedule_version == R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION:
        if phase_for_step(step) == TrainingPhase.STAGE_A:
            return PhaseProbabilityUnits(PROB_SCALE, 0, 0)
        return PhaseProbabilityUnits(500_000, 4_500_000, 0)
    if schedule_version == KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION:
        if phase_for_step(step) == TrainingPhase.STAGE_A:
            return PhaseProbabilityUnits(PROB_SCALE, 0, 0)
        return PhaseProbabilityUnits(3_000_000, 0, 2_000_000)
    if schedule_version == KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION:
        phase = phase_for_step(step)
        if phase == TrainingPhase.STAGE_A:
            return PhaseProbabilityUnits(PROB_SCALE, 0, 0)
        if phase == TrainingPhase.STAGE_B1:
            return PhaseProbabilityUnits(3_000_000, 0, 2_000_000)
        return PhaseProbabilityUnits(500_000, 3_500_000, 1_000_000)
    if schedule_version in {
        UNIFIED_EDIT_V2_SCHEDULE_VERSION,
        UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION,
    }:
        return PhaseProbabilityUnits(1_500_000, 2_000_000, 1_500_000)
    if schedule_version == UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION:
        return PhaseProbabilityUnits(500_000, 500_000, 4_000_000)
    if schedule_version == UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION:
        return PhaseProbabilityUnits(1_500_000, 1_500_000, 2_000_000)

    phase = phase_for_step(step)
    if phase == TrainingPhase.STAGE_A:
        return PhaseProbabilityUnits(PROB_SCALE, 0, 0)
    if phase == TrainingPhase.STAGE_B1:
        return PhaseProbabilityUnits(500_000, 4_500_000, 0)
    if phase == TrainingPhase.STAGE_B2:
        offset = int(step) - 250_000
        if offset < 10_000:
            n = offset + 1
            return PhaseProbabilityUnits(
                500_000 + 40 * n,
                4_500_000 - 50 * n,
                10 * n,
            )
        return PhaseProbabilityUnits(900_000, 4_000_000, 100_000)

    if schedule_version == STAGE_C_SAFE_MIX_SCHEDULE_VERSION:
        return PhaseProbabilityUnits(900_000, 4_000_000, 100_000)
    if schedule_version == STAGE_C_EDIT20_SCHEDULE_VERSION:
        return PhaseProbabilityUnits(1_750_000, 2_250_000, 1_000_000)
    if schedule_version == STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION:
        # D1 (500K->510K) changes only EDIT condition dropout. D2 increases
        # edit replay only after the D1 checkpoint has been evaluated.
        if int(step) < 510_000:
            return PhaseProbabilityUnits(1_750_000, 2_250_000, 1_000_000)
        return PhaseProbabilityUnits(1_500_000, 2_000_000, 1_500_000)

    offset = int(step) - 400_000
    if offset < 25_000:
        n = offset + 1
        return PhaseProbabilityUnits(
            900_000 + 34 * n,
            4_000_000 - 70 * n,
            100_000 + 36 * n,
        )
    return PhaseProbabilityUnits(1_750_000, 2_250_000, 1_000_000)


def optimizer_group_hparams(
    step: int,
    schedule_version: str = HIGH_LEVEL_SCHEDULE_VERSION,
) -> dict[str, dict[str, float]]:
    if schedule_version not in SUPPORTED_HIGH_LEVEL_SCHEDULES:
        raise ValueError(f"Unknown high-level schedule: {schedule_version!r}")
    phase = phase_for_step(step)
    if (
        schedule_version == KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION
        and phase == TrainingPhase.STAGE_B1
    ):
        return {
            "G0_existing": {"lr": 5e-5, "weight_decay": 0.01},
            "G1_context_weight": {"lr": 1e-4, "weight_decay": 0.01},
            "G2_context_bias": {"lr": 1e-4, "weight_decay": 0.0},
        }
    if schedule_version == KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION:
        return {
            "G0_existing": {"lr": 5e-5, "weight_decay": 0.01},
            "G1_context_weight": {"lr": 1e-4, "weight_decay": 0.01},
            "G2_context_bias": {"lr": 1e-4, "weight_decay": 0.0},
            "G3_ease_weight": {"lr": 1e-4, "weight_decay": 0.01},
            "G4_ease_bias": {"lr": 1e-4, "weight_decay": 0.0},
        }
    freeze_context_through_stage_b = (
        schedule_version == R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION
        and phase in {TrainingPhase.STAGE_B1, TrainingPhase.STAGE_B2}
    )
    if phase in {TrainingPhase.STAGE_A, TrainingPhase.STAGE_B1} or (
        freeze_context_through_stage_b
    ):
        lr0, lr_context = 1e-4, 0.0
    elif phase == TrainingPhase.STAGE_B2:
        lr0, lr_context = 1e-4, 1e-4
    else:
        lr0, lr_context = 2e-5, 5e-5
    return {
        "G0_existing": {"lr": lr0, "weight_decay": 0.01},
        "G1_context_weight": {"lr": lr_context, "weight_decay": 0.01},
        "G2_context_bias": {"lr": lr_context, "weight_decay": 0.0},
    }


@dataclass
class WeightedDeficitState:
    next_step: int = 0
    debt_hml: int = 0
    debt_edit: int = 0
    realized_hml: int = 0
    realized_edit: int = 0


class WeightedDeficitScheduler:
    """Exact synchronized high-level stream selector with replayable integer state."""

    def __init__(
        self,
        state: WeightedDeficitState | None = None,
        *,
        schedule_version: str = HIGH_LEVEL_SCHEDULE_VERSION,
    ) -> None:
        if schedule_version not in SUPPORTED_HIGH_LEVEL_SCHEDULES:
            raise ValueError(f"Unknown high-level schedule: {schedule_version!r}")
        self.schedule_version = str(schedule_version)
        self.state = state or WeightedDeficitState()

    def choose(self, step: int | None = None) -> TrainStream:
        expected = self.state.next_step
        step = expected if step is None else int(step)
        if step != expected:
            raise ValueError(f"Scheduler expected step {expected}, got {step}")
        units = probability_units_for_step(step, self.schedule_version)
        self.state.debt_hml += units.hml
        self.state.debt_edit += units.edit
        if self.state.debt_hml >= self.state.debt_edit:
            selected = TrainStream.HML_MIXED
            self.state.debt_hml -= PROB_SCALE
            self.state.realized_hml += 1
        else:
            selected = TrainStream.MOTION_EDIT
            self.state.debt_edit -= PROB_SCALE
            self.state.realized_edit += 1
        self.state.next_step += 1
        return selected

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": self.schedule_version,
            "prob_scale": PROB_SCALE,
            **self.state.__dict__,
        }

    def load_state_dict(
        self,
        payload: dict[str, Any],
        *,
        allow_schedule_fork_at_step: int | None = None,
    ) -> None:
        payload_version = payload.get("format")
        if payload_version != self.schedule_version:
            valid_stage_c_fork = (
                (
                    self.schedule_version
                    in {
                        STAGE_C_SAFE_MIX_SCHEDULE_VERSION,
                        STAGE_C_EDIT20_SCHEDULE_VERSION,
                        UNIFIED_EDIT_V2_SCHEDULE_VERSION,
                        UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION,
                    }
                    and payload_version
                    in {
                        HIGH_LEVEL_SCHEDULE_VERSION,
                        R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION,
                    }
                    and allow_schedule_fork_at_step == 400_000
                    and int(payload.get("next_step", -1)) == 400_000
                )
                or (
                    self.schedule_version
                    == R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION
                    and payload_version == HIGH_LEVEL_SCHEDULE_VERSION
                    and allow_schedule_fork_at_step == 200_000
                    and int(payload.get("next_step", -1)) == 200_000
                )
                or (
                    self.schedule_version
                    == KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION
                    and payload_version == HIGH_LEVEL_SCHEDULE_VERSION
                    and allow_schedule_fork_at_step == 200_000
                    and int(payload.get("next_step", -1)) == 200_000
                )
                or (
                    self.schedule_version
                    == KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION
                    and payload_version == KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION
                    and allow_schedule_fork_at_step == 250_000
                    and int(payload.get("next_step", -1)) == 250_000
                )
                or (
                    self.schedule_version
                    == STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION
                    and payload_version == STAGE_C_EDIT20_SCHEDULE_VERSION
                    and allow_schedule_fork_at_step == 500_000
                    and int(payload.get("next_step", -1)) == 500_000
                )
                or (
                    self.schedule_version
                    == UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION
                    and payload_version == UNIFIED_EDIT_V2_SCHEDULE_VERSION
                    and allow_schedule_fork_at_step == 450_000
                    and int(payload.get("next_step", -1)) == 450_000
                )
                or (
                    self.schedule_version
                    == UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION
                    and payload_version
                    == UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION
                    and allow_schedule_fork_at_step == 450_000
                    and int(payload.get("next_step", -1)) == 450_000
                )
            )
            if not valid_stage_c_fork:
                raise ValueError("High-level scheduler version mismatch")
        if int(payload.get("prob_scale", -1)) != PROB_SCALE:
            raise ValueError("High-level scheduler PROB_SCALE mismatch")
        self.state = WeightedDeficitState(
            **{
                name: int(payload[name])
                for name in WeightedDeficitState.__dataclass_fields__
            }
        )


def _stable_u64(*parts: Any) -> int:
    payload = json.dumps(
        [SAMPLE_RNG_VERSION, *parts],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def sample_key_u64(
    *,
    manifest_sha256: str,
    run_seed: int,
    global_sample_ordinal: int,
    train_stream_id: TrainStream,
    task_id: int,
    uid: str,
    random_stream_id: str,
) -> int:
    return _stable_u64(
        str(manifest_sha256),
        int(run_seed),
        int(global_sample_ordinal),
        int(train_stream_id),
        int(task_id),
        str(uid),
        str(random_stream_id),
    )


def _categorical_from_u64(draw: int, cumulative_units: tuple[int, ...], total: int) -> int:
    if total <= 0 or cumulative_units[-1] != total:
        raise ValueError("Invalid categorical unit table")
    # Multiplication avoids modulo bias while remaining deterministic in integers.
    scaled = (int(draw) * int(total)) >> 64
    for index, boundary in enumerate(cumulative_units):
        if scaled < boundary:
            return index
    return len(cumulative_units) - 1


def hml_capability_from_draw(
    step: int,
    draw: int,
    schedule_version: str = HIGH_LEVEL_SCHEDULE_VERSION,
) -> CapabilityId:
    units = probability_units_for_step(step, schedule_version)
    if units.hml <= 0:
        raise ValueError("HML capability requested at a zero-HML step")
    index = _categorical_from_u64(draw, (units.t2m, units.hml), units.hml)
    return CapabilityId.T2M if index == 0 else CapabilityId.KIMODO_CONTROL


def edit_pattern_from_draw(
    draw: int,
    schedule_version: str = HIGH_LEVEL_SCHEDULE_VERSION,
) -> EditConditionPattern:
    if schedule_version in {
        KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
        KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION,
    }:
        # source+instruction / source identity / instruction only /
        # unconditional = 80% / 10% / 5% / 5%. Control is absent here.
        index = _categorical_from_u64(
            draw,
            (4_000_000, 4_500_000, 4_750_000, PROB_SCALE),
            PROB_SCALE,
        )
        return (
            EditConditionPattern.SOURCE_TEXT,
            EditConditionPattern.SOURCE_IDENTITY,
            EditConditionPattern.TEXT_ONLY,
            EditConditionPattern.UNCONDITIONAL,
        )[index]
    if schedule_version in {
        UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION,
        UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION,
    }:
        # Mutually exclusive task-local branches for hierarchical Edit CFG:
        # source+text / source-identity / text-only / unconditional /
        # source+text+control = 75% / 10% / 5% / 5% / 5%.
        index = _categorical_from_u64(
            draw,
            (3_750_000, 4_250_000, 4_500_000, 4_750_000, PROB_SCALE),
            PROB_SCALE,
        )
        return (
            EditConditionPattern.SOURCE_TEXT,
            EditConditionPattern.SOURCE_IDENTITY,
            EditConditionPattern.TEXT_ONLY,
            EditConditionPattern.UNCONDITIONAL,
            EditConditionPattern.SOURCE_TEXT_CONTROL,
        )[index]
    if schedule_version in JOINT_ONLY_UNIFIED_EDIT_SCHEDULE_VERSIONS:
        # Unified Edit always receives source+instruction. Ten percent also
        # composes the target with a Kimodo control condition.
        index = _categorical_from_u64(
            draw,
            (4_500_000, PROB_SCALE),
            PROB_SCALE,
        )
        return (
            EditConditionPattern.SOURCE_TEXT,
            EditConditionPattern.SOURCE_TEXT_CONTROL,
        )[index]
    if schedule_version in {
        STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION,
        CONTEXT_ONLY_EDIT_SCHEDULE_VERSION,
    }:
        # source+text / source-identity / text-only / unconditional /
        # source+text+control = 75% / 10% / 5% / 5% / 5%.
        index = _categorical_from_u64(
            draw,
            (3_750_000, 4_250_000, 4_500_000, 4_750_000, PROB_SCALE),
            PROB_SCALE,
        )
        return (
            EditConditionPattern.SOURCE_TEXT,
            EditConditionPattern.SOURCE_IDENTITY,
            EditConditionPattern.TEXT_ONLY,
            EditConditionPattern.UNCONDITIONAL,
            EditConditionPattern.SOURCE_TEXT_CONTROL,
        )[index]

    # Existing stages retain source+text / source-only /
    # source+text+control / source+control = 70% / 10% / 15% / 5%.
    index = _categorical_from_u64(
        draw,
        (3_500_000, 4_000_000, 4_750_000, PROB_SCALE),
        PROB_SCALE,
    )
    return (
        EditConditionPattern.SOURCE_TEXT,
        EditConditionPattern.SOURCE_ONLY,
        EditConditionPattern.SOURCE_TEXT_CONTROL,
        EditConditionPattern.SOURCE_CONTROL,
    )[index]


def bernoulli_from_draw(draw: int, probability: float) -> bool:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Bernoulli probability must be in [0,1]")
    threshold = int(round(float(probability) * (1 << 64)))
    return int(draw) < min(threshold, 1 << 64)


def orthogonal_control_from_ordinal(
    global_sample_ordinal: int,
    probability: float,
    *,
    phase: int = 0,
) -> bool:
    """Allocate control exactly and replayably, independent of task semantics."""

    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("Control probability must be in [0,1]")
    fraction = Fraction(str(float(probability))).limit_denominator(1000)
    if abs(float(fraction) - float(probability)) > 1e-12:
        raise ValueError("Control probability must have an exact denominator <= 1000")
    if fraction.numerator == 0:
        return False
    if fraction.numerator == fraction.denominator:
        return True
    position = int(global_sample_ordinal) + int(phase)
    before = (position * fraction.numerator) // fraction.denominator
    after = ((position + 1) * fraction.numerator) // fraction.denominator
    return after > before


def ease_present_from_draw(
    capability: CapabilityId,
    draw: int,
    schedule_version: str,
) -> bool:
    """Sample independent Ease coverage for the registered control bootstrap."""

    if schedule_version != KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION:
        return False
    capability = CapabilityId(capability)
    if capability == CapabilityId.T2M:
        probability = 0.25
    elif capability == CapabilityId.KIMODO_CONTROL:
        probability = 0.50
    else:
        probability = 0.0
    return bernoulli_from_draw(draw, probability)


@dataclass
class BernoulliIntegrity:
    expected: float = 0.0
    variance: float = 0.0
    realized: int = 0
    trials: int = 0

    def update(self, probability: float, outcome: bool) -> None:
        probability = float(probability)
        self.expected += probability
        self.variance += probability * (1.0 - probability)
        self.realized += int(bool(outcome))
        self.trials += 1

    @property
    def bound(self) -> float:
        return max(1.0, 6.0 * math.sqrt(max(self.variance, 0.0)))

    @property
    def passed(self) -> bool:
        return abs(self.realized - self.expected) <= self.bound


@dataclass(frozen=True)
class SamplePlan:
    global_step: int
    global_sample_ordinal: int
    train_stream_id: TrainStream
    capability_id: CapabilityId
    row_index: int
    uid: str
    caption_index: int | None
    yaw_u64: int
    control_u64: int
    text_drop: bool
    edit_pattern: EditConditionPattern | None
    control_present: bool = False
    ease_present: bool = False


@dataclass
class StreamCursorState:
    cycle: int = 0
    offset: int = 0
    pending_batches: tuple[tuple[int, ...], ...] = ()


class DeterministicStreamCursor:
    """Sortish length bucketing without changing the row marginal distribution."""

    FORMAT = "hy273_sortish_stream_cursor_v1"

    def __init__(
        self,
        *,
        row_bucket_keys: Sequence[tuple[int, int]],
        manifest_sha256: str,
        run_seed: int,
        stream: TrainStream,
        global_batch_size: int,
        sort_window_batches: int = 8,
        state: StreamCursorState | None = None,
    ) -> None:
        if not row_bucket_keys:
            raise ValueError("Stream cursor requires at least one row")
        self.row_bucket_keys = tuple((int(a), int(b)) for a, b in row_bucket_keys)
        self.manifest_sha256 = str(manifest_sha256)
        self.run_seed = int(run_seed)
        self.stream = TrainStream(stream)
        self.global_batch_size = int(global_batch_size)
        self.sort_window_batches = int(sort_window_batches)
        if self.global_batch_size <= 0 or self.sort_window_batches <= 0:
            raise ValueError("Batch and sort-window sizes must be positive")
        self.state = state or StreamCursorState()
        self._permutation_cache: dict[int, tuple[int, ...]] = {}

    def _permutation(self, cycle: int) -> tuple[int, ...]:
        if cycle not in self._permutation_cache:
            rows = list(range(len(self.row_bucket_keys)))
            rows.sort(
                key=lambda row: _stable_u64(
                    self.manifest_sha256,
                    self.run_seed,
                    int(self.stream),
                    int(cycle),
                    row,
                    "row_permutation",
                )
            )
            self._permutation_cache = {cycle: tuple(rows)}
        return self._permutation_cache[cycle]

    def _take_rows(self, count: int) -> list[int]:
        output: list[int] = []
        while len(output) < count:
            permutation = self._permutation(self.state.cycle)
            available = len(permutation) - self.state.offset
            take = min(count - len(output), available)
            output.extend(permutation[self.state.offset : self.state.offset + take])
            self.state.offset += take
            if self.state.offset == len(permutation):
                self.state.cycle += 1
                self.state.offset = 0
        return output

    def _refill(self) -> None:
        pool_size = self.global_batch_size * self.sort_window_batches
        rows = self._take_rows(pool_size)
        # Python sort is stable, so equal-length rows retain their random cycle order.
        rows.sort(key=lambda row: self.row_bucket_keys[row])
        batches = tuple(
            tuple(rows[start : start + self.global_batch_size])
            for start in range(0, len(rows), self.global_batch_size)
        )
        self.state.pending_batches = batches

    def next_global_batch(self) -> tuple[int, ...]:
        if not self.state.pending_batches:
            self._refill()
        batch = self.state.pending_batches[0]
        self.state.pending_batches = self.state.pending_batches[1:]
        if len(batch) != self.global_batch_size:
            raise RuntimeError("Internal stream cursor produced a partial global batch")
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "manifest_sha256": self.manifest_sha256,
            "run_seed": self.run_seed,
            "stream": int(self.stream),
            "global_batch_size": self.global_batch_size,
            "sort_window_batches": self.sort_window_batches,
            "row_count": len(self.row_bucket_keys),
            "cycle": self.state.cycle,
            "offset": self.state.offset,
            "pending_batches": [list(batch) for batch in self.state.pending_batches],
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        expected = {
            "format": self.FORMAT,
            "manifest_sha256": self.manifest_sha256,
            "run_seed": self.run_seed,
            "stream": int(self.stream),
            "global_batch_size": self.global_batch_size,
            "sort_window_batches": self.sort_window_batches,
            "row_count": len(self.row_bucket_keys),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"Stream cursor contract mismatch for {key}: "
                    f"checkpoint={payload.get(key)!r}, expected={value!r}"
                )
        self.state = StreamCursorState(
            cycle=int(payload["cycle"]),
            offset=int(payload["offset"]),
            pending_batches=tuple(
                tuple(int(row) for row in batch)
                for batch in payload.get("pending_batches", [])
            ),
        )
