"""Completed native L2 copies, separate from storage and network I/O."""

from __future__ import annotations

import math
from collections import deque
from typing import Any


class HostIOMetrics:
    def __init__(self, *, enabled: bool):
        self.enabled = enabled
        self.generation = 0
        self.error: str | None = None
        self._pending = {direction: deque() for direction in ("read", "write")}
        self._totals = self._empty_totals()

    @staticmethod
    def _empty_totals():
        return {
            direction: {"bytes": 0, "batches": 0, "elapsed_ms": 0.0}
            for direction in ("read", "write")
        }

    def record(self, *, direction: str, completion: Any, num_bytes: int | None):
        if not self.enabled:
            return
        if num_bytes is None:
            self.error = "Native L2 transfer has unsupported payload metering"
            return
        if num_bytes <= 0:
            return
        self.collect()
        if not completion.timing_enabled:
            self.error = "Native L2 completion events do not support timing"
            return
        # FLAT_MEMORY: Keep the engine's private events after the cache ACK is reaped.
        self._pending[direction].append((completion, num_bytes))

    def collect(self):
        for direction, pending in self._pending.items():
            while pending:
                completion, num_bytes = pending[0]
                if not completion.finish_event.query():
                    break
                elapsed_ms = float(
                    completion.start_event.elapsed_time(completion.finish_event)
                )
                pending.popleft()
                if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                    self.error = "Native L2 copy reported a nonpositive duration"
                    continue
                totals = self._totals[direction]
                totals["bytes"] += num_bytes
                totals["batches"] += 1
                totals["elapsed_ms"] += elapsed_ms

    def snapshot(self):
        self.collect()
        return {
            "enabled": self.enabled and self.error is None,
            "generation": self.generation,
            "error": self.error,
            **{direction: totals.copy() for direction, totals in self._totals.items()},
        }

    def reset(self):
        self.collect()
        if any(self._pending.values()):
            raise RuntimeError(
                "Cannot reset native Host metrics before copy completion"
            )
        self.generation += 1
        self._totals = self._empty_totals()


def _unique_buffer_nbytes(buffers, seen):
    total = 0
    for buffer in buffers:
        # Count the registered view, not unrelated regions in its backing storage.
        key = (buffer.device, buffer.data_ptr(), buffer.nbytes)
        if key not in seen:
            total += int(buffer.nbytes)
            seen.add(key)
    return total


def host_pool_capacities(*, host_pool, device_pool):
    from sglang.srt.mem_cache.memory_pool_host import (
        DeepSeekV4PagedHostPool,
        DeepSeekV4StateHostPool,
        LogicalHostPool,
    )
    from sglang.srt.mem_cache.pool_host import HostPoolGroup

    if isinstance(host_pool, HostPoolGroup):
        pairs = [(entry.host_pool, entry.device_pool) for entry in host_pool.entries]
        anchor = host_pool.anchor_entry.host_pool
    else:
        pairs = [(host_pool, device_pool)]
        anchor = host_pool
    host_bytes = 0
    device_bytes = 0
    seen_host = set()
    seen_device = set()
    seen_device_buffers = set()
    for host, device in pairs:
        if isinstance(host, LogicalHostPool):
            continue
        if isinstance(host, (DeepSeekV4PagedHostPool, DeepSeekV4StateHostPool)):
            # V4 size_per_token describes a per-layer page, not a token.
            if id(host) not in seen_host:
                host_bytes += sum(
                    int(buffer.nbytes) for buffer in host.get_hybrid_pool_buffer()
                )
                seen_host.add(id(host))
            buffers = (
                host.device_buffers
                if isinstance(host, DeepSeekV4PagedHostPool)
                else host.device_page_views
            )
            device_bytes += _unique_buffer_nbytes(buffers, seen_device_buffers)
            continue
        if id(host) not in seen_host:
            host_bytes += int(host.size) * int(host.size_per_token)
            seen_host.add(id(host))
        if id(device) not in seen_device:
            device_bytes += int(device.size) * int(host.size_per_token)
            seen_device.add(id(device))
    return {
        "host_capacity_bytes": host_bytes,
        "device_capacity_bytes": device_bytes,
        "bytes_per_token": int(anchor.size_per_token),
        "capacity_scope": "sum of unique registered HiCache pool capacities",
        "bytes_per_token_scope": "primary index anchor; I/O uses actual transferred bytes",
    }
