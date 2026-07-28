"""Replayable stream-first global batch materialization for HY273 multitask DDP."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from data.hy273_multitask_manifest_dataset import (
    HY273MultitaskManifestDataset,
    build_global_sample_plans,
    collate_hy273_multitask,
)
from data.hy273_multitask_scheduler import (
    BUCKET_PLAN_VERSION,
    HIGH_LEVEL_SCHEDULE_VERSION,
    BernoulliIntegrity,
    DeterministicStreamCursor,
    EditConditionPattern,
    WeightedDeficitScheduler,
    probability_units_for_step,
)
from models.raw_motion.hy273_multitask_condition import CapabilityId, TrainStream


class HY273MultitaskStepBatcher:
    """Build one globally planned, rank-sharded batch per optimizer update."""

    FORMAT = "hy273_multitask_step_batcher_v2"

    def __init__(
        self,
        *,
        train_manifest: str | Path,
        run_seed: int,
        world_size: int,
        rank: int,
        batch_size_per_rank: int,
        materialize_workers: int = 4,
        sort_window_batches: int = 8,
        verify_payload_hash: bool = False,
        schedule_version: str = HIGH_LEVEL_SCHEDULE_VERSION,
        allow_schedule_fork_at_step: int | None = None,
    ) -> None:
        self.train_manifest = Path(train_manifest).expanduser().resolve()
        self.run_seed = int(run_seed)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.batch_size_per_rank = int(batch_size_per_rank)
        self.global_batch_size = self.world_size * self.batch_size_per_rank
        self.materialize_workers = max(1, int(materialize_workers))
        self.schedule_version = str(schedule_version)
        self.allow_schedule_fork_at_step = allow_schedule_fork_at_step
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.batch_size_per_rank <= 0:
            raise ValueError("batch_size_per_rank must be positive")

        self.datasets = {
            stream: HY273MultitaskManifestDataset(
                self.train_manifest,
                stream,
                verify_payload_hash=verify_payload_hash,
            )
            for stream in (TrainStream.HML_MIXED, TrainStream.MOTION_EDIT)
        }
        manifest_hashes = {dataset.manifest_sha256 for dataset in self.datasets.values()}
        if len(manifest_hashes) != 1:
            raise RuntimeError("Stream datasets disagree on unified manifest SHA")
        self.manifest_sha256 = next(iter(manifest_hashes))
        self.scheduler = WeightedDeficitScheduler(
            schedule_version=self.schedule_version
        )
        self.cursors = {
            stream: DeterministicStreamCursor(
                row_bucket_keys=self.datasets[stream].bucket_keys,
                manifest_sha256=self.manifest_sha256,
                run_seed=self.run_seed,
                stream=stream,
                global_batch_size=self.global_batch_size,
                sort_window_batches=sort_window_batches,
            )
            for stream in self.datasets
        }
        self.next_global_sample_ordinal = 0
        self.hml_t2m_integrity = BernoulliIntegrity()
        self.capability_counts = {int(capability): 0 for capability in CapabilityId}
        self.edit_pattern_counts = {int(pattern): 0 for pattern in EditConditionPattern}
        # A ratio partition is the sample block over which masked loss
        # numerators and denominators are combined before the DDP average.
        # Normally it is one physical rank. An expansion reshard may preserve
        # the parent's larger partition across several adjacent ranks.
        self.ratio_partition_world_size = self.world_size
        self.ratio_partition_batch_size = self.batch_size_per_rank
        self.ratio_group_size = 1

    def next_batch(self, global_step: int) -> tuple[dict[str, Any], TrainStream, str]:
        if int(global_step) != self.scheduler.state.next_step:
            raise ValueError(
                f"Batcher expected step={self.scheduler.state.next_step}, got {global_step}"
            )
        stream = self.scheduler.choose(global_step)
        row_indices = self.cursors[stream].next_global_batch()
        plans = build_global_sample_plans(
            dataset=self.datasets[stream],
            row_indices=row_indices,
            global_step=global_step,
            first_global_ordinal=self.next_global_sample_ordinal,
            run_seed=self.run_seed,
            schedule_version=self.schedule_version,
        )
        if len(plans) != self.global_batch_size:
            raise RuntimeError("Planner did not emit one full global batch")
        self.next_global_sample_ordinal += self.global_batch_size

        # The caption draw must happen before final bucketing: a HumanML3D row
        # may resolve to a much shorter derived segment than its full motion.
        # Stable sorting keeps equal-length plans in their row-cursor order.
        dataset = self.datasets[stream]
        plans.sort(
            key=lambda plan: (
                *dataset.plan_bucket_key(plan),
                *dataset.plan_lengths(plan),
            )
        )

        if stream == TrainStream.HML_MIXED:
            units = probability_units_for_step(global_step, self.schedule_version)
            q_t2m = units.t2m / float(units.hml)
            for plan in plans:
                self.hml_t2m_integrity.update(
                    q_t2m, plan.capability_id == CapabilityId.T2M
                )
        for plan in plans:
            self.capability_counts[int(plan.capability_id)] += 1
            if plan.edit_pattern is not None:
                self.edit_pattern_counts[int(plan.edit_pattern)] += 1

        # Rotate compact length partitions so one virtual rank does not always
        # own the shortest (or longest) ratio-of-sums batch. After an expansion
        # reshard, adjacent physical ranks remain subchunks of one parent
        # partition; this preserves the parent's loss objective.
        virtual_rank = self.rank // self.ratio_group_size
        rank_within_partition = self.rank % self.ratio_group_size
        partition_index = (
            virtual_rank + int(global_step)
        ) % self.ratio_partition_world_size
        chunk_index = (
            partition_index * self.ratio_group_size + rank_within_partition
        )
        start = chunk_index * self.batch_size_per_rank
        local_plans = plans[start : start + self.batch_size_per_rank]
        if self.materialize_workers == 1:
            samples = [dataset.materialize(plan) for plan in local_plans]
        else:
            with ThreadPoolExecutor(max_workers=self.materialize_workers) as pool:
                samples = list(pool.map(dataset.materialize, local_plans))
        batch = collate_hy273_multitask(samples)
        trace_payload = [
            {
                **asdict(plan),
                "train_stream_id": int(plan.train_stream_id),
                "capability_id": int(plan.capability_id),
                "edit_pattern": (
                    None if plan.edit_pattern is None else int(plan.edit_pattern)
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
            "train_manifest": str(self.train_manifest),
            "manifest_sha256": self.manifest_sha256,
            "run_seed": self.run_seed,
            "world_size": self.world_size,
            "batch_size_per_rank": self.batch_size_per_rank,
            "global_batch_size": self.global_batch_size,
            "bucket_plan_version": BUCKET_PLAN_VERSION,
            "next_global_sample_ordinal": self.next_global_sample_ordinal,
            "scheduler": self.scheduler.state_dict(),
            "cursors": {
                str(int(stream)): cursor.state_dict()
                for stream, cursor in self.cursors.items()
            },
            "hml_t2m_integrity": asdict(self.hml_t2m_integrity),
            "capability_counts": dict(self.capability_counts),
            "edit_pattern_counts": dict(self.edit_pattern_counts),
            "ratio_partition": {
                "format": "hy273_rank_ratio_partition_v1",
                "world_size": self.ratio_partition_world_size,
                "batch_size": self.ratio_partition_batch_size,
            },
        }

    def load_state_dict(
        self,
        payload: dict[str, Any],
        *,
        allow_same_global_batch_reshard: bool = False,
        preserve_source_rank_ratio_objective: bool = False,
    ) -> None:
        expected = {
            "format": self.FORMAT,
            "train_manifest": str(self.train_manifest),
            "manifest_sha256": self.manifest_sha256,
            "run_seed": self.run_seed,
            "global_batch_size": self.global_batch_size,
            "bucket_plan_version": BUCKET_PLAN_VERSION,
        }
        if not allow_same_global_batch_reshard:
            expected.update(
                {
                    "world_size": self.world_size,
                    "batch_size_per_rank": self.batch_size_per_rank,
                }
            )
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"Batcher resume mismatch for {key}: checkpoint={payload.get(key)!r}, expected={value!r}"
                )
        if allow_same_global_batch_reshard:
            previous_world_size = int(payload.get("world_size", 0))
            previous_batch_size = int(payload.get("batch_size_per_rank", 0))
            if previous_world_size <= 0 or previous_batch_size <= 0:
                raise ValueError("Reshard source has an invalid DDP topology")
            if previous_world_size * previous_batch_size != self.global_batch_size:
                raise ValueError("Research reshard must preserve the effective global batch")
        else:
            previous_world_size = int(payload.get("world_size", 0))
            previous_batch_size = int(payload.get("batch_size_per_rank", 0))

        ratio_partition = payload.get("ratio_partition")
        if ratio_partition is None:
            source_ratio_world_size = previous_world_size
            source_ratio_batch_size = previous_batch_size
        else:
            if not isinstance(ratio_partition, dict) or ratio_partition.get(
                "format"
            ) != "hy273_rank_ratio_partition_v1":
                raise ValueError("Batcher resume has an invalid ratio_partition")
            source_ratio_world_size = int(ratio_partition.get("world_size", 0))
            source_ratio_batch_size = int(ratio_partition.get("batch_size", 0))
        if (
            source_ratio_world_size <= 0
            or source_ratio_batch_size <= 0
            or source_ratio_world_size * source_ratio_batch_size
            != self.global_batch_size
        ):
            raise ValueError("Batcher resume has an invalid ratio partition")

        if preserve_source_rank_ratio_objective:
            if not allow_same_global_batch_reshard:
                raise ValueError(
                    "Preserving parent ratio partitions requires an explicit reshard"
                )
            if source_ratio_batch_size % self.batch_size_per_rank != 0:
                raise ValueError(
                    "The new per-rank batch must divide the parent ratio partition"
                )
            ratio_group_size = (
                source_ratio_batch_size // self.batch_size_per_rank
            )
            if ratio_group_size <= 1:
                raise ValueError(
                    "Parent-ratio preservation is only defined for an expansion"
                )
            if self.world_size != source_ratio_world_size * ratio_group_size:
                raise ValueError(
                    "Physical ranks cannot reconstruct the parent ratio partitions"
                )
            self.ratio_partition_world_size = source_ratio_world_size
            self.ratio_partition_batch_size = source_ratio_batch_size
            self.ratio_group_size = ratio_group_size
        elif (
            previous_world_size == self.world_size
            and previous_batch_size == self.batch_size_per_rank
        ):
            if source_ratio_batch_size % self.batch_size_per_rank != 0:
                raise ValueError(
                    "Saved ratio partition is incompatible with the current batch"
                )
            ratio_group_size = (
                source_ratio_batch_size // self.batch_size_per_rank
            )
            if (
                ratio_group_size <= 0
                or self.world_size
                != source_ratio_world_size * ratio_group_size
            ):
                raise ValueError(
                    "Saved ratio partition is incompatible with the current topology"
                )
            self.ratio_partition_world_size = source_ratio_world_size
            self.ratio_partition_batch_size = source_ratio_batch_size
            self.ratio_group_size = ratio_group_size
        else:
            # Legacy research reshard behavior starts a new rank-local ratio
            # objective. The formal Stage-A -> Stage-B expansion opts into the
            # preservation branch above instead.
            self.ratio_partition_world_size = self.world_size
            self.ratio_partition_batch_size = self.batch_size_per_rank
            self.ratio_group_size = 1
        self.scheduler.load_state_dict(
            payload["scheduler"],
            allow_schedule_fork_at_step=self.allow_schedule_fork_at_step,
        )
        for stream, cursor in self.cursors.items():
            cursor.load_state_dict(payload["cursors"][str(int(stream))])
        self.next_global_sample_ordinal = int(payload["next_global_sample_ordinal"])
        integrity = payload["hml_t2m_integrity"]
        self.hml_t2m_integrity = BernoulliIntegrity(
            expected=float(integrity["expected"]),
            variance=float(integrity["variance"]),
            realized=int(integrity["realized"]),
            trials=int(integrity["trials"]),
        )
        self.capability_counts = {
            int(key): int(value) for key, value in payload["capability_counts"].items()
        }
        self.edit_pattern_counts = {int(pattern): 0 for pattern in EditConditionPattern}
        self.edit_pattern_counts.update(
            {
                int(key): int(value)
                for key, value in payload["edit_pattern_counts"].items()
            }
        )
