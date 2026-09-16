"""FLAT_MEMORY: Read-only, rank-local HiCache completion accounting and collection."""

from __future__ import annotations

import math
import os
import uuid
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import HiCacheAck
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


class HiCacheIOCounters:
    def __init__(self):
        self.epoch = str(uuid.uuid4())
        self._totals = {
            direction: {
                "bytes": 0,
                "duration_ns": 0,
                "batches": 0,
                "untimed_batches": 0,
            }
            for direction in ("read", "write")
        }
        self._unsupported = False

    def account_completed(self, direction: str, ack: HiCacheAck) -> float | None:
        # The caller has consumed the ACK after its normal completion check.
        # This method must never query, wait for, or synchronize an event.
        duration_ms = None
        if ack.timing_enabled:
            try:
                elapsed = ack.start_event.elapsed_time(ack.finish_event)
                if math.isfinite(elapsed) and elapsed > 0:
                    duration_ms = elapsed
            except (RuntimeError, NotImplementedError):
                pass
        totals = self._totals[direction]
        totals["batches"] += 1
        if ack.num_bytes is None:
            self._unsupported = True
        else:
            totals["bytes"] += ack.num_bytes
        if duration_ms is None:
            totals["untimed_batches"] += 1
        else:
            totals["duration_ns"] += round(duration_ms * 1_000_000)
        return duration_ms

    def snapshot(self) -> dict:
        result = {"epoch": self.epoch, "status": "ok"}
        if self._unsupported:
            result.update(
                status="unsupported",
                error="A completed L2 transfer has unsupported payload metering",
            )
        else:
            result["l2"] = {name: dict(totals) for name, totals in self._totals.items()}
        return result


def snapshot_unified_hicache(cache) -> dict:
    # This leaf reads live scheduler-owned queues, not worker events or Prometheus.
    result = cache.hicache_io_counters.snapshot()
    cc = cache.cache_controller
    if cc is None or cache.host_memory_mode != "cache":
        return {
            "epoch": result["epoch"],
            "status": "unavailable",
            "error": "HiCache L2 host cache is not enabled",
        }
    result["pending"] = {
        "h2d": len(cc.load_queue) + len(cc.ack_load_queue),
        "d2h": len(cc.write_queue) + len(cc.ack_write_queue),
        "storage_prefetch": len(cache.ongoing_prefetch),
        "storage_backup": len(cache.ongoing_backup),
    }
    result["mooncake"] = {"status": "unavailable", "reason": "Mooncake is not attached"}
    if cc.enable_storage:
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        if isinstance(cc.storage_backend, MooncakeStore):
            result["mooncake"] = cc.storage_backend.get_io_stats_snapshot()
    return result


def gather_hicache_io(
    *,
    snapshot: Callable[[], dict],
    tp_rank: int,
    tp_size: int,
    pp_rank: int,
    pp_size: int,
    dp_rank: int | None,
    dp_size: int,
    gather: Callable[[dict], list[dict]],
) -> dict:
    rank = {
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "dp_rank": 0 if dp_rank is None else dp_rank,
        "pid": os.getpid(),
    }
    try:
        rank.update(snapshot())
    except Exception as exc:
        rank.update(status="error", error=f"{type(exc).__name__}: {exc}")
    # A local snapshot failure still participates in the same management collective.
    ranks = gather(rank)
    ranks.sort(key=lambda item: (item["dp_rank"], item["pp_rank"], item["tp_rank"]))
    return {
        "schema_version": 1,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "dp_size": dp_size,
        "scope": "completed_accounted_window",
        "drain": False,
        "pending_semantics": {
            "h2d_d2h": "queued_operations_plus_unconsumed_ack_batches",
            "storage": "scheduler_owned_operations_not_yet_retired",
            "boundary": "pending_at_boundary_is_excluded_until_normal_ack_accounting",
        },
        "ranks": ranks,
    }


def collect_hicache_io(
    *,
    cache: BasePrefixCache,
    tp_rank: int,
    tp_size: int,
    pp_rank: int,
    pp_size: int,
    dp_rank: int | None,
    dp_size: int,
    attn_cp_size: int,
    attn_dcp_size: int,
    tp_cpu_group,
) -> dict:
    # The management fanout is guaranteed for this topology only. Do not introduce
    # a cross-DP/PP collective on configurations with different control routing.
    if pp_size != 1 or dp_size != 1 or attn_cp_size != 1 or attn_dcp_size != 1:
        return {
            "schema_version": 1,
            "tp_size": tp_size,
            "pp_size": pp_size,
            "dp_size": dp_size,
            "status": "unsupported",
            "error": "HiCache I/O collection requires PP1, DP1 and no CP",
            "ranks": [],
        }

    def snapshot():
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        if not isinstance(cache, UnifiedRadixCache):
            return {"status": "unavailable", "error": "UnifiedRadixCache is required"}
        return snapshot_unified_hicache(cache)

    def gather(rank):
        if tp_size == 1:
            return [rank]
        import torch.distributed as dist

        ranks = [None] * tp_size
        dist.all_gather_object(ranks, rank, group=tp_cpu_group)
        return ranks

    return gather_hicache_io(
        snapshot=snapshot,
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        dp_rank=dp_rank,
        dp_size=dp_size,
        gather=gather,
    )
