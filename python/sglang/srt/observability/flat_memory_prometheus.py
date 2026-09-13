"""Flat metric names and separately labelled Mooncake client-side I/O."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# FLAT_MEMORY: Keep the original metric names for existing experiment consumers.
_BANDWIDTH_FIELDS = {
    **{
        f"{medium}_{op}_bw_gbps": f"{medium}_{op}_bw_gbps"
        for medium in ("dram", "ssd")
        for op in ("read", "write")
    },
    **{
        f"{medium}_{op}_total_bytes": f"{medium}_{op}_total_bytes"
        for medium in ("dram", "ssd")
        for op in ("read", "write")
    },
    **{
        f"{medium}_{op}_ops": f"{medium}_{op}_count"
        for medium in ("dram", "ssd")
        for op in ("read", "write")
    },
    **{
        name: name
        for name in (
            "dram_used_bytes",
            "ssd_used_bytes",
            "total_blocks",
            "dram_overflow_count",
            "dram_overflow_bytes",
            "duplicate_key_skips",
            "delete_count",
            "delete_bytes",
            "put_failures",
            "dram_utilization_pct",
            "ssd_utilization_pct",
        )
    },
}
_TRANSFER_COUNTERS = {
    "d2h_tokens_total": "Tokens copied from GPU payload buffers to host payload buffers.",
    "storage_write_tokens_total": "Tokens whose writes to external storage completed.",
    "host_eviction_ops_total": "Completed host-cache eviction operations.",
    "host_eviction_tokens_total": "Tokens evicted from the host cache.",
}


class FlatStorageMetrics:
    def __init__(
        self,
        *,
        labels: Mapping[str, str],
        counter_cls: Any,
        gauge_cls: Any,
        histogram_cls: Any,
    ) -> None:
        self.labels = labels
        self.gauges = {
            metric: gauge_cls(
                name=f"sglang:flat_memory_{metric}",
                documentation=f"Native per-rank Flat storage {metric}; NaN means unavailable.",
                labelnames=labels.keys(),
            )
            for metric in _BANDWIDTH_FIELDS
        }
        self.counters = {
            metric: counter_cls(
                name=f"sglang:{metric}",
                documentation=help_text,
                labelnames=labels.keys(),
            )
            for metric, help_text in _TRANSFER_COUNTERS.items()
        }
        self.prefetch_latency_ms = histogram_cls(
            name="sglang:prefetch_latency_ms",
            documentation="Completed storage-prefetch latency in milliseconds.",
            labelnames=labels.keys(),
            buckets=[10, 50, 100, 250, 500, 1000, 2000, 5000, 10000],
        )
        self.client_io = {
            (operation, field): counter_cls(
                name=f"sglang:mooncake_client_rpc_{operation}_{field}_total",
                documentation=f"Caller-visible Mooncake {operation} RPC {field}; storage medium is unknown.",
                labelnames=labels.keys(),
            )
            for operation in ("read", "write")
            for field in ("bytes", "nanoseconds", "operations")
        }

    def log_transfer(self, name: str, value: int) -> None:
        if value > 0:
            self.counters[name].labels(**self.labels).inc(value)

    def log_prefetch_latency(self, latency_ms: float) -> None:
        if latency_ms >= 0 and math.isfinite(latency_ms):
            self.prefetch_latency_ms.labels(**self.labels).observe(latency_ms)

    def log(
        self,
        *,
        bandwidth: Mapping[str, Any] | None,
        client_io: Mapping[str, Any] | None,
    ) -> None:
        for metric, key in _BANDWIDTH_FIELDS.items():
            value = bandwidth.get(key) if bandwidth is not None else None
            available = (
                type(value) in (int, float) and math.isfinite(value) and value >= 0
            )
            self.gauges[metric].labels(**self.labels).set(
                value if available else math.nan
            )
        if client_io is not None:
            for (operation, field), counter in self.client_io.items():
                key = {"bytes": "bytes", "nanoseconds": "ns", "operations": "ops"}[
                    field
                ]
                value = client_io.get(f"{operation}_{key}")
                if type(value) is int and value > 0:
                    counter.labels(**self.labels).inc(value)
