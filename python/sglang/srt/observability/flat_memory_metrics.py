"""Flat cache accounting and the optional PD metadata extension."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from typing import Protocol

FLAT_CACHE_FIELDS = ("flat_dram", "flat_ssd", "flat_mixed")
FLAT_PREFETCH_FIELDS = (
    "flat_prefetch_dram_ms",
    "flat_prefetch_dram_ops",
    "flat_prefetch_ssd_ms",
    "flat_prefetch_ssd_ops",
)
FLAT_TRANSFER_FIELDS = (
    "kvcache_page_size",
    "kvcache_bytes_per_page",
    "storage_read_latency_ms",
    "storage_read_tokens",
    "d2h_tokens",
    "storage_write_tokens",
)
# FLAT_MEMORY: Slots 0..6 belong to upstream cache/MM counts on both PD endpoints.
FLAT_PD_FIRST_SLOT = 7
FLAT_PD_MAGIC = 0x464D
FLAT_PD_VERSION = 1
_INT32_MAX = (1 << 31) - 1


class FlatRequestMetrics(Protocol):
    flat_storage_backend: bool
    flat_prefetch_stats: dict[str, int | float | str | None]
    flat_cached_tokens: dict[str, int | None]
    storage_hit_length: int
    cached_tokens_device: int
    cached_tokens_host: int
    cached_tokens_storage: int
    kvcache_page_size: int | None
    kvcache_bytes_per_page: int | None
    storage_read_latency_ms: float | None
    storage_read_tokens: int | None
    d2h_tokens: int | None
    storage_write_tokens: int | None


def initialize_flat_request(
    req: FlatRequestMetrics, *, page_size: int, bytes_per_page: int | None
) -> None:
    """Enable measured zeroes only after selecting the Flat backend."""
    if not req.flat_storage_backend:
        req.flat_prefetch_stats = dict.fromkeys(
            (*FLAT_CACHE_FIELDS, *FLAT_PREFETCH_FIELDS), 0
        )
        req.flat_cached_tokens = dict.fromkeys(FLAT_CACHE_FIELDS, 0)
        req.storage_read_latency_ms = 0.0
        req.storage_read_tokens = 0
        req.d2h_tokens = 0
        # FLAT_MEMORY: Shared asynchronous offloads are not attributed to a request.
        req.storage_write_tokens = None
    req.flat_storage_backend = True
    req.kvcache_page_size = page_size
    req.kvcache_bytes_per_page = bytes_per_page


def finalize_flat_cache_accounting(req: FlatRequestMetrics, *, prefix_len: int) -> None:
    # FLAT_MEMORY: Direct restores are L3 hits even without a host-cache match.
    storage = min(prefix_len, max(0, req.storage_hit_length))
    req.cached_tokens_device = prefix_len - storage
    req.cached_tokens_host = 0
    req.cached_tokens_storage = storage
    counts = {name: req.flat_prefetch_stats[name] for name in FLAT_CACHE_FIELDS}
    values = list(counts.values())
    valid = all(type(value) is int and value >= 0 for value in values)
    if valid:
        dram, ssd, mixed = values
        valid = dram + ssd == storage and mixed <= ssd
    req.flat_cached_tokens = counts if valid else dict.fromkeys(FLAT_CACHE_FIELDS, None)


def flat_cached_tokens_details(
    req: FlatRequestMetrics,
) -> dict[str, int | float | str | None]:
    details: dict[str, int | float | str | None] = {}
    if req.flat_storage_backend:
        details.update(
            storage=req.cached_tokens_storage, storage_backend="FlatMemoryStore"
        )
        details.update(req.flat_cached_tokens)
        details.update(
            {name: req.flat_prefetch_stats[name] for name in FLAT_PREFETCH_FIELDS}
        )
        details.update(
            (name, value)
            for name, value in req.flat_prefetch_stats.items()
            if name.startswith("flat_restore_")
        )
    for field, key in zip(
        FLAT_TRANSFER_FIELDS,
        ("page_size", "bytes_per_page", *FLAT_TRANSFER_FIELDS[2:]),
    ):
        value = getattr(req, field)
        if value is not None:
            details[key] = value
    return details


def encode_flat_pd_metrics(req: FlatRequestMetrics) -> list[int]:
    """Encode six optional fields without overflowing the shared int32 buffer."""
    payload = [0] * len(FLAT_TRANSFER_FIELDS)
    valid_mask = 0
    for index, name in enumerate(FLAT_TRANSFER_FIELDS):
        value = getattr(req, name)
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            continue
        limit = _INT32_MAX / 1000 if name == "storage_read_latency_ms" else _INT32_MAX
        if (
            value < 0
            or value > limit
            or (isinstance(value, float) and not math.isfinite(value))
        ):
            continue
        wire_value = (
            int(value * 1000) if name == "storage_read_latency_ms" else int(value)
        )
        if wire_value > _INT32_MAX or (
            name != "storage_read_latency_ms" and wire_value != value
        ):
            continue
        payload[index] = wire_value
        valid_mask |= 1 << index
    return [*payload, FLAT_PD_MAGIC, FLAT_PD_VERSION, valid_mask]


def decode_flat_pd_metrics(values: Sequence[int]) -> dict[str, int | float | None]:
    decoded = dict.fromkeys(FLAT_TRANSFER_FIELDS, None)
    if len(values) < 16 or values[13] != FLAT_PD_MAGIC or values[14] != FLAT_PD_VERSION:
        return decoded
    mask = values[15]
    if not isinstance(mask, int) or mask < 0 or mask >= 1 << len(FLAT_TRANSFER_FIELDS):
        return decoded
    for index, name in enumerate(FLAT_TRANSFER_FIELDS):
        value = values[FLAT_PD_FIRST_SLOT + index]
        if mask & (1 << index) and type(value) is int and 0 <= value <= _INT32_MAX:
            decoded[name] = (
                value / 1000.0 if name == "storage_read_latency_ms" else value
            )
    return decoded


class MooncakeClientIOStats:
    """Caller-visible RPC I/O; these measurements do not identify a storage tier."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats = self._empty()

    @staticmethod
    def _empty() -> dict[str, int]:
        return {
            f"{operation}_{field}": 0
            for operation in ("read", "write")
            for field in ("bytes", "ns", "ops")
        }

    def record(self, *, operation: str, completed_bytes: int, elapsed_ns: int) -> None:
        if operation not in ("read", "write"):
            raise ValueError(f"Unknown I/O operation: {operation}")
        if completed_bytes <= 0:
            return
        with self._lock:
            self._stats[f"{operation}_bytes"] += completed_bytes
            self._stats[f"{operation}_ns"] += max(0, elapsed_ns)
            self._stats[f"{operation}_ops"] += 1

    def snapshot_and_reset(self) -> dict[str, int | str]:
        with self._lock:
            stats, self._stats = self._stats, self._empty()
        return {"source": "python_client_rpc", "medium": "unknown", **stats}


def native_mooncake_bandwidth(
    stats: Mapping[str, object],
) -> dict[str, int | float | str]:
    """Keep absent native medium measurements absent, rather than inventing zeroes."""
    result: dict[str, int | float | str] = {"source": "mooncake_native"}
    for medium in ("dram", "ssd"):
        for operation in ("read", "write"):
            prefix = f"{medium}_{operation}"
            byte_count = stats.get(f"{prefix}_bytes")
            elapsed_ns = stats.get(f"{prefix}_ns")
            ops = stats.get(f"{prefix}_ops")
            if type(byte_count) is int and byte_count >= 0:
                result[f"{prefix}_total_bytes"] = byte_count
                if type(elapsed_ns) is int and elapsed_ns > 0:
                    result[f"{prefix}_bw_gbps"] = byte_count / elapsed_ns
            if type(ops) is int and ops >= 0:
                result[f"{prefix}_count"] = ops
    return result
