"""Replayable task-first batches for T2M, Edit, and Interaction training."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.hy273_interaction_dataset import (
    HY273InteractionDataset,
    collate_hy273_interaction,
)
from data.hy273_reaction_dataset import (
    HY273ReactionDataset,
    collate_hy273_reaction,
)
from data.hy273_multitask_manifest_dataset import (
    HY273MultitaskManifestDataset,
    build_global_sample_plans,
    collate_hy273_multitask,
)
from data.hy273_multitask_scheduler import (
    DeterministicStreamCursor,
    KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
)
from models.raw_motion.hy273_multitask_condition import TrainStream


TASK_ORDER = (
    TrainStream.HML_MIXED,
    TrainStream.MOTION_EDIT,
    TrainStream.INTERACTION,
)


@dataclass
class TaskDeficitState:
    next_step: int = 0
    debt_hml: int = 0
    debt_edit: int = 0
    debt_interaction: int = 0
    realized_hml: int = 0
    realized_edit: int = 0
    realized_interaction: int = 0


class PiecewiseTaskScheduler:
    """Exact weighted-deficit task selection over declared step segments."""

    FORMAT = "hy273_unified_actor_task_schedule_v1"
    SCALE = 100

    def __init__(self, segments: Sequence[Mapping[str, Any]]) -> None:
        paired_keys = {
            "reaction" if "reaction" in segment else "interaction"
            for segment in segments
        }
        if len(paired_keys) != 1:
            raise ValueError(
                "Task schedule must consistently use reaction or interaction"
            )
        self.paired_key = next(iter(paired_keys))
        normalized = []
        previous_end = 0
        for segment in segments:
            start = int(segment["start"])
            end = int(segment["end"])
            weights = {
                TrainStream.HML_MIXED: int(segment["t2m"]),
                TrainStream.MOTION_EDIT: int(segment["edit"]),
                TrainStream.REACTION: int(segment[self.paired_key]),
            }
            if start != previous_end or end <= start:
                raise ValueError("Task schedule segments must be contiguous and non-empty")
            if min(weights.values()) < 0 or sum(weights.values()) != self.SCALE:
                raise ValueError("Every task segment must contain non-negative weights summing to 100")
            normalized.append(
                {
                    "start": start,
                    "end": end,
                    "t2m": weights[TrainStream.HML_MIXED],
                    "edit": weights[TrainStream.MOTION_EDIT],
                    self.paired_key: weights[TrainStream.REACTION],
                }
            )
            previous_end = end
        if not normalized:
            raise ValueError("At least one task schedule segment is required")
        self.segments = tuple(normalized)
        self.state = TaskDeficitState()

    def _weights(self, step: int) -> dict[TrainStream, int]:
        for segment in self.segments:
            if int(segment["start"]) <= step < int(segment["end"]):
                return {
                    TrainStream.HML_MIXED: int(segment["t2m"]),
                    TrainStream.MOTION_EDIT: int(segment["edit"]),
                    TrainStream.REACTION: int(segment[self.paired_key]),
                }
        raise ValueError(f"Step {step} is outside the declared task schedule")

    def choose(self, step: int) -> TrainStream:
        step = int(step)
        if step != self.state.next_step:
            raise ValueError(f"Task scheduler expected step {self.state.next_step}, got {step}")
        weights = self._weights(step)
        debts = {
            TrainStream.HML_MIXED: self.state.debt_hml + weights[TrainStream.HML_MIXED],
            TrainStream.MOTION_EDIT: self.state.debt_edit + weights[TrainStream.MOTION_EDIT],
            TrainStream.INTERACTION: self.state.debt_interaction
            + weights[TrainStream.INTERACTION],
        }
        selected = max(TASK_ORDER, key=lambda stream: (debts[stream], -int(stream)))
        debts[selected] -= self.SCALE
        self.state.debt_hml = debts[TrainStream.HML_MIXED]
        self.state.debt_edit = debts[TrainStream.MOTION_EDIT]
        self.state.debt_interaction = debts[TrainStream.INTERACTION]
        if selected == TrainStream.HML_MIXED:
            self.state.realized_hml += 1
        elif selected == TrainStream.MOTION_EDIT:
            self.state.realized_edit += 1
        else:
            self.state.realized_interaction += 1
        self.state.next_step += 1
        return selected

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "segments": list(self.segments),
            "state": asdict(self.state),
        }

    def load_state_dict(
        self,
        payload: Mapping[str, Any],
        *,
        allow_segment_extension_at_step: int | None = None,
    ) -> None:
        if payload.get("format") != self.FORMAT:
            raise ValueError("Task scheduler format mismatch")
        saved_segments = payload.get("segments")
        segments_match = saved_segments == list(self.segments)
        if not segments_match and allow_segment_extension_at_step is not None:
            boundary = int(allow_segment_extension_at_step)
            saved_state = payload.get("state", {})
            saved_rows = list(saved_segments or [])
            current_rows = list(self.segments)
            if len(saved_rows) == len(current_rows) and saved_rows:
                saved_prefix = saved_rows[:-1]
                current_prefix = current_rows[:-1]
                saved_last = dict(saved_rows[-1])
                current_last = dict(current_rows[-1])
                saved_end = int(saved_last.pop("end", -1))
                current_end = int(current_last.pop("end", -1))
                segments_match = (
                    saved_prefix == current_prefix
                    and saved_last == current_last
                    and saved_end == boundary
                    and current_end > saved_end
                    and int(saved_state.get("next_step", -1)) == boundary
                )
        if not segments_match:
            raise ValueError("Task scheduler segments differ from the checkpoint")
        self.state = TaskDeficitState(
            **{key: int(value) for key, value in payload["state"].items()}
        )


class HY273UnifiedActorStepBatcher:
    """Materialize one synchronized homogeneous task batch per DDP update."""

    FORMAT = "hy273_unified_actor_step_batcher_v1"

    def __init__(
        self,
        *,
        multitask_manifest: str | Path,
        interaction_root: str | Path,
        run_seed: int,
        world_size: int,
        rank: int,
        batch_size_t2m_edit: int,
        batch_size_interaction: int,
        batch_size_edit: int | None = None,
        task_segments: Sequence[Mapping[str, Any]],
        materialize_workers: int = 4,
        sort_window_batches: int = 8,
        verify_payload_hash: bool = False,
        interaction_exclude_overlength: bool = False,
        paired_task: str = "interaction",
        orthogonal_control_probability: float = 0.0,
    ) -> None:
        self.multitask_manifest = Path(multitask_manifest).expanduser().resolve()
        self.interaction_root = Path(interaction_root).expanduser().resolve()
        self.run_seed = int(run_seed)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.materialize_workers = max(1, int(materialize_workers))
        self.interaction_exclude_overlength = bool(
            interaction_exclude_overlength
        )
        self.paired_task = str(paired_task).lower()
        self.orthogonal_control_probability = float(
            orthogonal_control_probability
        )
        if not 0.0 <= self.orthogonal_control_probability <= 1.0:
            raise ValueError("orthogonal_control_probability must be in [0,1]")
        if self.paired_task not in {"interaction", "reaction"}:
            raise ValueError("paired_task must be 'interaction' or 'reaction'")
        self._materialize_pool = (
            None
            if self.materialize_workers == 1
            else ThreadPoolExecutor(max_workers=self.materialize_workers)
        )
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.local_batch_sizes = {
            TrainStream.HML_MIXED: int(batch_size_t2m_edit),
            TrainStream.MOTION_EDIT: int(
                batch_size_t2m_edit
                if batch_size_edit is None
                else batch_size_edit
            ),
            TrainStream.INTERACTION: int(batch_size_interaction),
        }
        if min(self.local_batch_sizes.values()) <= 0:
            raise ValueError("All per-task batch sizes must be positive")
        self.global_batch_sizes = {
            stream: size * self.world_size
            for stream, size in self.local_batch_sizes.items()
        }
        self.datasets = {
            TrainStream.HML_MIXED: HY273MultitaskManifestDataset(
                self.multitask_manifest,
                TrainStream.HML_MIXED,
                verify_payload_hash=verify_payload_hash,
            ),
            TrainStream.MOTION_EDIT: HY273MultitaskManifestDataset(
                self.multitask_manifest,
                TrainStream.MOTION_EDIT,
                verify_payload_hash=verify_payload_hash,
            ),
            TrainStream.REACTION: (
                HY273ReactionDataset(
                    self.interaction_root,
                    split="train",
                    exclude_overlength=self.interaction_exclude_overlength,
                )
                if self.paired_task == "reaction"
                else HY273InteractionDataset(
                    self.interaction_root,
                    split="train",
                    exclude_overlength=self.interaction_exclude_overlength,
                )
            ),
        }
        self.manifest_hashes = {
            int(stream): dataset.manifest_sha256
            for stream, dataset in self.datasets.items()
        }
        self.scheduler = PiecewiseTaskScheduler(task_segments)
        self.cursors = {
            stream: DeterministicStreamCursor(
                row_bucket_keys=dataset.bucket_keys,
                manifest_sha256=dataset.manifest_sha256,
                run_seed=self.run_seed,
                stream=stream,
                global_batch_size=self.global_batch_sizes[stream],
                sort_window_batches=int(sort_window_batches),
            )
            for stream, dataset in self.datasets.items()
        }
        self.next_global_sample_ordinal = {stream: 0 for stream in TASK_ORDER}

    def _local_slice(self, plans: Sequence[Any], stream: TrainStream, step: int) -> list[Any]:
        local_batch = self.local_batch_sizes[stream]
        virtual_rank = (self.rank + int(step)) % self.world_size
        start = virtual_rank * local_batch
        return list(plans[start : start + local_batch])

    def _materialize(self, dataset: Any, plans: Sequence[Any]) -> list[dict[str, Any]]:
        if self._materialize_pool is None:
            return [dataset.materialize(plan) for plan in plans]
        return list(self._materialize_pool.map(dataset.materialize, plans))

    def close(self) -> None:
        if self._materialize_pool is not None:
            self._materialize_pool.shutdown(wait=True)
            self._materialize_pool = None

    def next_batch(
        self, global_step: int
    ) -> tuple[dict[str, Any], TrainStream, str]:
        stream = self.scheduler.choose(int(global_step))
        dataset = self.datasets[stream]
        row_indices = self.cursors[stream].next_global_batch()
        first_ordinal = self.next_global_sample_ordinal[stream]
        if stream == TrainStream.REACTION:
            plans = [
                dataset.build_plan(
                    row_index=row_index,
                    global_step=int(global_step),
                    global_sample_ordinal=first_ordinal + offset,
                    run_seed=self.run_seed,
                    orthogonal_control_probability=(
                        self.orthogonal_control_probability
                    ),
                )
                for offset, row_index in enumerate(row_indices)
            ]
            plans.sort(
                key=lambda plan: min(
                    int(dataset.rows[plan.row_index]["frames"]),
                    dataset.max_frames,
                )
            )
        else:
            plans = build_global_sample_plans(
                dataset=dataset,
                row_indices=row_indices,
                global_step=int(global_step),
                first_global_ordinal=first_ordinal,
                run_seed=self.run_seed,
                schedule_version=KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
                orthogonal_control_probability=(
                    self.orthogonal_control_probability
                ),
            )
            plans.sort(
                key=lambda plan: (
                    *dataset.plan_bucket_key(plan),
                    *dataset.plan_lengths(plan),
                )
            )
        self.next_global_sample_ordinal[stream] += self.global_batch_sizes[stream]
        local_plans = self._local_slice(plans, stream, int(global_step))
        samples = self._materialize(dataset, local_plans)
        batch = (
            (
                collate_hy273_reaction(samples)
                if self.paired_task == "reaction"
                else collate_hy273_interaction(samples)
            )
            if stream == TrainStream.REACTION
            else collate_hy273_multitask(samples)
        )
        trace_payload = [
            {
                **asdict(plan),
                **(
                    {
                        "train_stream_id": int(plan.train_stream_id),
                        "capability_id": int(plan.capability_id),
                        "edit_pattern": (
                            None
                            if plan.edit_pattern is None
                            else int(plan.edit_pattern)
                        ),
                    }
                    if stream != TrainStream.INTERACTION
                    else {}
                ),
            }
            for plan in plans
        ]
        trace = hashlib.sha256(
            json.dumps(
                trace_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return batch, stream, trace

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "multitask_manifest": str(self.multitask_manifest),
            "interaction_root": str(self.interaction_root),
            "run_seed": self.run_seed,
            "world_size": self.world_size,
            "interaction_exclude_overlength": self.interaction_exclude_overlength,
            "paired_task": self.paired_task,
            "local_batch_sizes": {
                str(int(stream)): value
                for stream, value in self.local_batch_sizes.items()
            },
            "manifest_hashes": dict(self.manifest_hashes),
            "scheduler": self.scheduler.state_dict(),
            "cursors": {
                str(int(stream)): cursor.state_dict()
                for stream, cursor in self.cursors.items()
            },
            "next_global_sample_ordinal": {
                str(int(stream)): value
                for stream, value in self.next_global_sample_ordinal.items()
            },
        }

    def load_state_dict(
        self,
        payload: Mapping[str, Any],
        *,
        allow_segment_extension_at_step: int | None = None,
    ) -> None:
        expected = {
            "format": self.FORMAT,
            "multitask_manifest": str(self.multitask_manifest),
            "interaction_root": str(self.interaction_root),
            "run_seed": self.run_seed,
            "world_size": self.world_size,
            "interaction_exclude_overlength": self.interaction_exclude_overlength,
            "local_batch_sizes": {
                str(int(stream)): value
                for stream, value in self.local_batch_sizes.items()
            },
            "manifest_hashes": self.manifest_hashes,
        }
        saved_paired_task = str(payload.get("paired_task", "interaction"))
        if saved_paired_task != self.paired_task:
            raise ValueError(
                "Unified actor batcher resume mismatch for paired_task: "
                f"checkpoint={saved_paired_task!r}, expected={self.paired_task!r}"
            )
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"Unified actor batcher resume mismatch for {key}: "
                    f"checkpoint={payload.get(key)!r}, expected={value!r}"
                )
        self.scheduler.load_state_dict(
            payload["scheduler"],
            allow_segment_extension_at_step=allow_segment_extension_at_step,
        )
        for stream, cursor in self.cursors.items():
            cursor.load_state_dict(payload["cursors"][str(int(stream))])
        self.next_global_sample_ordinal = {
            stream: int(payload["next_global_sample_ordinal"][str(int(stream))])
            for stream in TASK_ORDER
        }
