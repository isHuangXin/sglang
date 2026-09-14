"""Flat benchmark accounting; telemetry is separate from request execution."""

import asyncio
import math
import os
import re
import time
import uuid
import warnings

import aiohttp
import requests

FLAT_DETAIL_FIELDS = (
    "flat_dram",
    "flat_ssd",
    "flat_mixed",
    "flat_prefetch_dram_ms",
    "flat_prefetch_dram_ops",
    "flat_prefetch_ssd_ms",
    "flat_prefetch_ssd_ops",
)
_WINDOW_COUNTERS = (
    "read_completed_ops",
    "read_completed_bytes",
    "write_completed_ops",
    "write_completed_bytes",
    "durable_write_batches",
    "durable_write_bytes",
    "io_errors",
)
_IO_RESULT_FIELDS = (
    "flat_io_window_seconds",
    "flat_dram_read_bw_gbps",
    "flat_dram_write_bw_gbps",
    "flat_ssd_read_bw_gbps",
    "flat_ssd_write_bw_gbps",
    "flat_ssd_read_peak_bw_gbps",
    "flat_ssd_write_peak_bw_gbps",
    "flat_ssd_read_io_count",
    "flat_ssd_write_io_count",
    "flat_ssd_durable_write_batches",
    "flat_ssd_new_kv_blocks",
)


def _flat_counter(value):
    if type(value) is not int or value < 0:
        raise ValueError(f"Missing or invalid Flat counter: {value!r}")
    return value


def _flat_optional_sum(outputs, name, integer=True):
    values = [getattr(output, name) for output in outputs]
    types = (int,) if integer else (int, float)
    if not values or any(
        type(value) not in types or value < 0 or not math.isfinite(value)
        for value in values
    ):
        return None
    return sum(values)


def consume_native_usage(output, meta):
    # FLAT_MEMORY: Native SSE usage is cumulative, including metadata-only chunks.
    for wire, field in (
        ("prompt_tokens", "server_prompt_tokens"),
        ("completion_tokens", "server_completion_tokens"),
        ("cached_tokens", "server_cached_tokens"),
    ):
        if wire in meta:
            value = meta[wire]
            setattr(output, field, value if type(value) is int and value >= 0 else None)
    if output.server_cached_tokens is not None:
        output.cached_tokens = output.server_cached_tokens
    if "cached_tokens_details" not in meta:
        return
    details = meta["cached_tokens_details"]
    output.cached_tokens_details = details if isinstance(details, dict) else None
    details = output.cached_tokens_details or {}
    for medium in ("device", "host", "storage"):
        value = details.get(medium)
        value = value if type(value) is int and value >= 0 else None
        setattr(output, f"server_cached_tokens_{medium}", value)
        setattr(output, f"cached_tokens_{medium}", value)
    for name in FLAT_DETAIL_FIELDS:
        value = details.get(name)
        types = (int, float) if name.endswith("_ms") else (int,)
        setattr(
            output,
            name,
            (
                value
                if type(value) in types and value >= 0 and math.isfinite(value)
                else None
            ),
        )
    for field, wire in (
        ("kvcache_page_size", "page_size"),
        ("kvcache_bytes_per_page", "bytes_per_page"),
        ("storage_read_latency_ms", "storage_read_latency_ms"),
        ("storage_read_tokens", "storage_read_tokens"),
        ("d2h_tokens", "d2h_tokens"),
        ("storage_write_tokens", "storage_write_tokens"),
    ):
        value = details.get(wire)
        types = (int, float) if field.endswith("_ms") else (int,)
        setattr(
            output,
            field,
            (
                value
                if type(value) in types and value >= 0 and math.isfinite(value)
                else None
            ),
        )


def summarize_flat_cache(outputs, total_prompt_tokens):
    successful = [output for output in outputs if output.success]
    dram = _flat_optional_sum(successful, "flat_dram")
    ssd = _flat_optional_sum(successful, "flat_ssd")
    mixed = _flat_optional_sum(successful, "flat_mixed")
    storage = _flat_optional_sum(successful, "server_cached_tokens_storage")
    total = dram + ssd if dram is not None and ssd is not None else None
    # FLAT_MEMORY: Mixed is a subset of SSD, never an additional cache tier.
    if total is not None and (storage != total or mixed is None or mixed > ssd):
        dram = ssd = mixed = total = None
    device = _flat_optional_sum(successful, "server_cached_tokens_device")

    def rate(tokens):
        return (
            100 * tokens / total_prompt_tokens
            if tokens is not None
            and total_prompt_tokens is not None
            and total_prompt_tokens > 0
            else None
        )

    result = {
        "flat_cached_tokens_dram": dram,
        "flat_cached_tokens_ssd": ssd,
        "flat_cached_tokens_mixed": mixed,
        "flat_cached_tokens_total": total,
        "device_hit_rate_pct": rate(device),
        "flat_dram_hit_rate_pct": rate(dram),
        "flat_ssd_hit_rate_pct": rate(ssd),
        "flat_storage_hit_rate_pct": rate(total),
    }
    sums, counts = [], []
    for medium in ("dram", "ssd"):
        elapsed = _flat_optional_sum(successful, f"flat_prefetch_{medium}_ms", False)
        ops = _flat_optional_sum(successful, f"flat_prefetch_{medium}_ops")
        sums.append(elapsed)
        counts.append(ops)
        result[f"flat_{medium}_prefetch_latency_ms"] = (
            elapsed / ops if elapsed is not None and ops else None
        )
    result["flat_prefetch_latency_ms"] = (
        sum(sums) / sum(counts)
        if all(value is not None for value in sums + counts) and sum(counts)
        else None
    )
    return result


def summarize_cache_usage(outputs):
    successful = [output for output in outputs if output.success]
    prompt = _flat_optional_sum(successful, "server_prompt_tokens")
    cached = _flat_optional_sum(successful, "server_cached_tokens")
    result = {
        "total_prompt_tokens": prompt,
        "total_cached_tokens": cached,
        "cache_hit_rate": cached / prompt if cached is not None and prompt else None,
    }
    for tier in ("device", "host", "storage"):
        result[f"total_cached_tokens_{tier}"] = _flat_optional_sum(
            successful, f"server_cached_tokens_{tier}"
        )
    for name in ("kvcache_page_size", "kvcache_bytes_per_page"):
        values = {getattr(output, name) for output in successful}
        result[name] = (
            next(iter(values)) if len(values) == 1 and None not in values else None
        )
    for name in ("storage_read_tokens", "d2h_tokens", "storage_write_tokens"):
        result[f"total_{name}"] = _flat_optional_sum(successful, name)
    latency = _flat_optional_sum(successful, "storage_read_latency_ms", False)
    reads = [output for output in successful if output.storage_read_tokens]
    result["avg_storage_read_latency_ms"] = (
        latency / len(reads) if latency is not None and reads else None
    )
    return result


def _flat_rank_map(snapshot, expected_tp_size=None):
    if not isinstance(snapshot, dict):
        raise ValueError("Missing Flat snapshot")
    tp_size = snapshot.get("tp_size", expected_tp_size)
    if tp_size not in (1, 4, 8) or type(tp_size) is not int:
        raise ValueError("Flat snapshot requires an explicit TP size of 1, 4 or 8")
    if expected_tp_size is not None and tp_size != expected_tp_size:
        raise ValueError("Flat snapshot TP size changed")
    ranks = snapshot["ranks"]
    if not isinstance(ranks, list) or len(ranks) != tp_size:
        raise ValueError(f"Flat I/O requires all {tp_size} TP rank snapshots")
    mapped = {}
    for rank in ranks:
        tp_rank = _flat_counter(rank["tp_rank"])
        if tp_rank in mapped or not _flat_counter(rank["pid"]):
            raise ValueError("Duplicate TP rank or invalid PID")
        _flat_counter(rank["gpu_id"])
        mapped[tp_rank] = rank
    if set(mapped) != set(range(tp_size)) or len({r["pid"] for r in ranks}) != tp_size:
        raise ValueError("Incomplete or duplicate Flat TP identities")
    return mapped


def _validate_flat_window(snapshot, window_id, active, expected_tp_size=None):
    ranks = _flat_rank_map(snapshot, expected_tp_size)
    boundaries = set()
    for rank in ranks.values():
        window = rank["io_window"]
        if (
            window["enabled"] is not True
            or window["window_id"] != window_id
            or window["active"] is not active
            or window["aborted"] is not False
            or window["overflowed"] is not False
            or _flat_counter(window["bucket_ns"]) != 100_000_000
        ):
            raise ValueError("Disabled, mismatched, aborted or overflowed Flat window")
        start, end = _flat_counter(window["start_ns"]), _flat_counter(window["end_ns"])
        if not start or (active and end != 0) or (not active and end <= start):
            raise ValueError("Invalid Flat window boundaries")
        boundaries.add((start, end))
        for name in _WINDOW_COUNTERS:
            value = _flat_counter(window[name])
            if (active or name == "io_errors") and value:
                raise ValueError("Nonempty begin window or Flat I/O errors")
        if not isinstance(window["buckets"], list) or (active and window["buckets"]):
            raise ValueError("Invalid Flat I/O buckets")
        for name in ("pending_backups", "pending_prefetches"):
            if _flat_counter(rank[name]):
                raise ValueError("Flat pipeline was not drained")
        _flat_counter(rank["flat_io_errors"])
    if len(boundaries) != 1:
        raise ValueError("TP ranks do not share Flat window boundaries")
    return ranks, next(iter(boundaries))


def summarize_flat_io(before, after, window_id, expected_tp_size=None):
    first, (start, _) = _validate_flat_window(before, window_id, True, expected_tp_size)
    last, (end_start, end) = _validate_flat_window(after, window_id, False, len(first))
    if start != end_start:
        raise ValueError("Flat begin/end window start mismatch")
    seconds = (end - start) / 1e9
    totals = dict.fromkeys(_WINDOW_COUNTERS, 0)
    deltas = dict.fromkeys(
        ("dram_read_total_bytes", "dram_write_total_bytes", "ssd_write_count"), 0
    )
    buckets = {}
    for tp_rank, rank in last.items():
        prior = first[tp_rank]
        if any(rank[name] != prior[name] for name in ("pid", "gpu_id", "gds_mode")):
            raise ValueError("Flat TP identity changed during collection")
        if _flat_counter(rank["flat_io_errors"] - prior["flat_io_errors"]):
            raise ValueError("Flat pipeline reported I/O errors")
        for name in deltas:
            delta = _flat_counter(rank["bandwidth"][name]) - _flat_counter(
                prior["bandwidth"][name]
            )
            deltas[name] += _flat_counter(delta)
        window = rank["io_window"]
        for name in totals:
            totals[name] += window[name]
        seen = set()
        read_bytes = write_bytes = 0
        for bucket in window["buckets"]:
            index = _flat_counter(bucket["index"])
            if index in seen or index * 100_000_000 >= end - start:
                raise ValueError("Duplicate or out-of-window Flat bucket index")
            seen.add(index)
            read, write = _flat_counter(bucket["read_bytes"]), _flat_counter(
                bucket["write_bytes"]
            )
            read_bytes += read
            write_bytes += write
            merged = buckets.setdefault(index, [0, 0])
            merged[0] += read
            merged[1] += write
        if (
            read_bytes != window["read_completed_bytes"]
            or write_bytes != window["durable_write_bytes"]
        ):
            raise ValueError("Flat bucket totals do not match completed/durable bytes")
    return {
        "flat_io_window_seconds": seconds,
        "flat_dram_read_bw_gbps": deltas["dram_read_total_bytes"] / seconds / 1e9,
        "flat_dram_write_bw_gbps": deltas["dram_write_total_bytes"] / seconds / 1e9,
        "flat_ssd_read_bw_gbps": totals["read_completed_bytes"] / seconds / 1e9,
        "flat_ssd_write_bw_gbps": totals["durable_write_bytes"] / seconds / 1e9,
        # FLAT_MEMORY: Merge rank buckets before peaks; partial buckets still divide by 100ms.
        "flat_ssd_read_peak_bw_gbps": max((b[0] for b in buckets.values()), default=0)
        / 0.1
        / 1e9,
        "flat_ssd_write_peak_bw_gbps": max((b[1] for b in buckets.values()), default=0)
        / 0.1
        / 1e9,
        "flat_ssd_read_io_count": totals["read_completed_ops"],
        "flat_ssd_write_io_count": totals["write_completed_ops"],
        "flat_ssd_durable_write_batches": totals["durable_write_batches"],
        "flat_ssd_new_kv_blocks": deltas["ssd_write_count"],
    }


class FlatMemoryIOWindow:
    """Own a UUID window without discarding completed requests on telemetry failure."""

    def __init__(self, base_url, enabled, *, headers=None, expected_tp_size=None):
        self.base_url, self.enabled = base_url.rstrip("/"), enabled
        self.headers = dict(headers or {})
        self.expected_tp_size = expected_tp_size
        self.window_id = str(uuid.uuid4())
        self.begin_attempted = self.ready = False
        self.result = dict.fromkeys(_IO_RESULT_FIELDS)
        self.result.update(
            flat_io_status="unavailable",
            flat_io_error="Flat I/O collection was not requested",
            flat_io_metadata={"before": None, "after": None},
        )

    async def _request(self, action=None):
        timeout = (
            float(os.environ.get("HICACHE_FLUSH_TIMEOUT", "150")) if action else 15
        )
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            kwargs = {"headers": self.headers}
            if action:
                kwargs["json"] = {"action": action, "window_id": self.window_id}
            async with session.request(
                "POST" if action else "GET",
                self.base_url
                + ("/flat_memory/io_window" if action else "/server_info"),
                **kwargs,
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def _wait_idle(self):
        timeout = float(
            os.environ.get(
                "DRAIN_TIMEOUT", os.environ.get("HICACHE_IO_DRAIN_TIMEOUT", "120")
            )
        )
        deadline = time.monotonic() + timeout
        while True:
            info = await self._request()
            tp_size = info.get("tp_size")
            if type(tp_size) is not int or tp_size < 1:
                raise ValueError("Flat server must report a positive integer TP size")
            if self.expected_tp_size is None:
                self.expected_tp_size = tp_size
            elif tp_size != self.expected_tp_size:
                raise ValueError("Flat server TP size changed during collection")
            snapshot = info["internal_states"][0]["flat_memory"]
            ranks = _flat_rank_map(snapshot, self.expected_tp_size)
            pending = sum(
                _flat_counter(rank[name])
                for rank in ranks.values()
                for name in ("pending_backups", "pending_prefetches")
            )
            if not pending:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Flat I/O pipeline drain timed out")
            await asyncio.sleep(min(1.0, remaining))

    async def _abort(self):
        if self.begin_attempted:
            try:
                await self._request("abort")
            except Exception as exc:
                warnings.warn(f"Flat I/O abort for {self.window_id} failed: {exc}")
            finally:
                self.begin_attempted = False

    def _unavailable(self, exc):
        self.result["flat_io_error"] = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"Flat I/O unavailable: {self.result['flat_io_error']}")

    async def __aenter__(self):
        if self.enabled:
            try:
                await self._wait_idle()
                # FLAT_MEMORY: A lost begin response still requires an owned abort.
                self.begin_attempted = True
                before = await self._request("begin")
                self.result["flat_io_metadata"]["before"] = before
                _validate_flat_window(
                    before, self.window_id, True, self.expected_tp_size
                )
                self.ready = True
            except Exception as exc:
                self._unavailable(exc)
                await self._abort()
                raise
            except BaseException:
                await self._abort()
                raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self.ready and exc_type is None:
                try:
                    await self._wait_idle()
                    after = await self._request("end")
                    self.result["flat_io_metadata"]["after"] = after
                    metrics = summarize_flat_io(
                        self.result["flat_io_metadata"]["before"],
                        after,
                        self.window_id,
                        self.expected_tp_size,
                    )
                    self.result.update(metrics, flat_io_status="ok", flat_io_error=None)
                    self.begin_attempted = False
                except Exception as error:
                    self._unavailable(error)
        finally:
            await self._abort()
        return False


def fetch_legacy_metrics(host, port, patterns, *, timeout=5.0):
    """Return explicitly legacy lifetime metrics, never native-window measurements."""
    try:
        response = requests.get(f"http://{host}:{port}/metrics", timeout=timeout)
        response.raise_for_status()
        result = {}
        for label, metric in patterns.items():
            matches = re.findall(
                rf"^{re.escape(metric)}(?:\{{[^\n]*\}})?\s+(\S+)(?:\s|$)",
                response.text,
                re.MULTILINE,
            )
            if matches:
                values = [float(value) for value in matches]
                if all(math.isfinite(value) and value >= 0 for value in values):
                    # FLAT_MEMORY: Percentages need capacity weights; do not sum rank gauges.
                    result[label] = (
                        (values[0] if len(values) == 1 else None)
                        if label.endswith("_pct")
                        else sum(values)
                    )
        return result
    except Exception as exc:
        warnings.warn(f"Legacy metrics unavailable from {host}:{port}: {exc}")
        return {}


def fetch_mooncake_eviction_metrics(
    master_host="localhost", metrics_port=9003, timeout=5.0
):
    return fetch_legacy_metrics(
        master_host,
        metrics_port,
        {
            "successful_evictions": "master_successful_evictions_total",
            "attempted_evictions": "master_attempted_evictions_total",
            "evicted_key_count": "master_evicted_key_count",
            "evicted_size_bytes": "master_evicted_size_bytes",
        },
        timeout=timeout,
    )


def fetch_sglang_bandwidth_metrics(
    prefill_host="localhost", prefill_port=30010, timeout=5.0
):
    patterns = {
        f"{kind}_bandwidth_{suffix}": f"sglang:{kind}_bandwidth_{suffix}"
        for kind in ("backup", "prefetch")
        for suffix in ("sum", "count")
    }
    values = fetch_legacy_metrics(prefill_host, prefill_port, patterns, timeout=timeout)
    result = {}
    for kind in ("backup", "prefetch"):
        total, count = values.get(f"{kind}_bandwidth_sum"), values.get(
            f"{kind}_bandwidth_count"
        )
        if total is not None and count is not None:
            result[f"{kind}_bandwidth_sum_gbs"] = total
            result[f"{kind}_bandwidth_count"] = count
            result[f"{kind}_bandwidth_avg_gbs"] = total / count if count else None
    return result


def fetch_flat_memory_metrics(
    prefill_host="localhost", prefill_port=30010, timeout=5.0
):
    names = [
        f"{medium}_{direction}_{field}"
        for medium in ("dram", "ssd")
        for direction in ("read", "write")
        for field in ("bw_gbps", "total_bytes", "ops")
    ]
    names += [
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
    ]
    return fetch_legacy_metrics(
        prefill_host,
        prefill_port,
        {name: f"sglang:flat_memory_{name}" for name in names},
        timeout=timeout,
    )
