"""Native HiCache snapshot and drain coordination on the scheduler thread."""

from __future__ import annotations

import time

import torch


def pending_hicache_operations(cache) -> int:
    pending = sum(
        len(operations)
        for operations in (
            cache.ongoing_write_through,
            cache.ongoing_load_back,
            cache.ongoing_prefetch,
            cache.ongoing_backup,
        )
    )
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if isinstance(cache, UnifiedRadixCache) and cache.buffer_pipeline is not None:
        pending += int(not cache.buffer_pipeline.is_idle())
    controller = cache.cache_controller
    if controller is not None:
        pending += len(controller.write_queue) + len(controller.load_queue)
        pending += controller.pending_storage_io()
    return pending


def collect_hicache_io_state(*, cache, idle: bool) -> dict:
    controller = cache.cache_controller
    cache.check_hicache_events()
    snapshot = controller.host_io_snapshot()
    snapshot["idle"] = bool(idle)
    snapshot["storage_pending"] = pending_hicache_operations(cache)
    # Storage plugins need not expose the native Mooncake snapshot extension.
    storage_snapshot = getattr(
        controller.storage_backend, "get_storage_io_snapshot", None
    )
    if storage_snapshot is not None:
        snapshot["mooncake_io"] = storage_snapshot()
    tp_size = torch.distributed.get_world_size(group=controller.tp_group)
    ranks = [None] * tp_size
    if tp_size > 1:
        torch.distributed.all_gather_object(ranks, snapshot, group=controller.tp_group)
    else:
        ranks[0] = snapshot
    if any(
        rank["tp_rank"] != index or rank["tp_size"] != tp_size
        for index, rank in enumerate(ranks)
    ):
        raise RuntimeError("Inconsistent native HiCache TP snapshots")
    return {"ranks": ranks}


def drain_hicache_io(*, cache, timeout: float) -> bool:
    controller = cache.cache_controller
    deadline = time.monotonic() + timeout
    while True:
        cache.check_hicache_events()
        snapshot = controller.host_io_snapshot()
        pending = pending_hicache_operations(cache) + snapshot["pending"]
        status = torch.tensor(
            [int(pending > 0), int(time.monotonic() >= deadline)], dtype=torch.int
        )
        torch.distributed.all_reduce(
            status, op=torch.distributed.ReduceOp.MAX, group=controller.tp_group
        )
        if status[0].item() == 0:
            return True
        if status[1].item() != 0:
            return False
        time.sleep(0.01)
