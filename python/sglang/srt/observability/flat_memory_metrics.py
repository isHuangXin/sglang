"""Flat storage metrics with explicit measurement provenance."""

import threading
from typing import Mapping


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

