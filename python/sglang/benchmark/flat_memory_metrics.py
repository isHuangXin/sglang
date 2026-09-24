"""Usage accounting and optional owned Flat I/O collection for benchmarks."""

import asyncio
import math
import os
import re
import time
import uuid
import warnings

import requests
from copy import deepcopy

from sglang.benchmark.flat_memory_report import (
    REPORT_FIELDS,
    _cache_values,
    _counter,
    _flat_rank_map,
    _number,
    _prefetch_values,
    _sum,
    _summarize_flat_io,
    _validate_flat_window,
    resolve_metrics_mode,
)

FLAT_DETAIL_FIELDS = (
    "flat_dram",
    "flat_ssd",
    "flat_mixed",
    "flat_prefetch_dram_ms",
    "flat_prefetch_dram_ops",
    "flat_prefetch_ssd_ms",
    "flat_prefetch_ssd_ops",
)
_IO_RESULT_FIELDS = tuple(
    field["name"] for field in REPORT_FIELDS if field["group"] == "io"
)


def consume_native_usage(output, meta):
    """Replace cumulative usage with the latest observation, including metadata-only SSE."""
    output.server_usage = {**output.server_usage, **deepcopy(meta)}
    for wire, field in (
        ("prompt_tokens", "server_prompt_tokens"),
        ("completion_tokens", "server_completion_tokens"),
        ("cached_tokens", "server_cached_tokens"),
    ):
        if wire in meta:
            setattr(output, field, _number(meta[wire], integer=True))
    if "cached_tokens" in meta:
        output.cached_tokens = output.server_cached_tokens
    if "cache_source_mode" in meta:
        output.cache_source_mode = deepcopy(meta["cache_source_mode"])
    if "cached_tokens_details" not in meta:
        return
    details = meta["cached_tokens_details"]
    output.cached_tokens_details = (
        deepcopy(details) if isinstance(details, dict) else None
    )
    details = details if isinstance(details, dict) else {}
    if "cache_source_mode" in details:
        output.cache_source_mode = deepcopy(details["cache_source_mode"])
    for tier in ("device", "host", "storage"):
        value = _number(details.get(tier), integer=True)
        setattr(output, f"server_cached_tokens_{tier}", value)
        setattr(output, f"cached_tokens_{tier}", value)
    for name in FLAT_DETAIL_FIELDS:
        setattr(
            output, name, _number(details.get(name), integer=not name.endswith("_ms"))
        )
    for field, wire in (
        ("kvcache_page_size", "page_size"),
        ("kvcache_bytes_per_page", "bytes_per_page"),
        ("storage_read_latency_ms", "storage_read_latency_ms"),
        ("storage_read_tokens", "storage_read_tokens"),
        ("d2h_tokens", "d2h_tokens"),
        ("storage_write_tokens", "storage_write_tokens"),
    ):
        setattr(
            output, field, _number(details.get(wire), integer=not field.endswith("_ms"))
        )


def _usage_result(outputs):
    successful = [output for output in outputs if output.success]
    return {
        "server_prompt_tokens": [output.server_prompt_tokens for output in successful],
        "server_cached_tokens": [output.server_cached_tokens for output in successful],
        "cached_tokens_details": [
            output.cached_tokens_details for output in successful
        ],
    }


def summarize_cache_usage(outputs):
    successful = [output for output in outputs if output.success]
    values = _cache_values(_usage_result(successful))
    result = {
        name: values[name]
        for name in (
            "total_prompt_tokens",
            "total_cached_tokens",
            "cache_hit_rate",
            "total_cached_tokens_device",
            "total_cached_tokens_host",
            "total_cached_tokens_storage",
        )
    }
    for name, values in (
        ("kvcache_page_size", [output.kvcache_page_size for output in successful]),
        (
            "kvcache_bytes_per_page",
            [output.kvcache_bytes_per_page for output in successful],
        ),
    ):
        result[name] = (
            values[0]
            if values
            and all(_number(value, integer=True) == values[0] for value in values)
            else None
        )
    for name, values in (
        ("storage_read_tokens", [output.storage_read_tokens for output in successful]),
        ("d2h_tokens", [output.d2h_tokens for output in successful]),
        (
            "storage_write_tokens",
            [output.storage_write_tokens for output in successful],
        ),
    ):
        result[f"total_{name}"] = _sum(values)
    latency = _sum(
        (output.storage_read_latency_ms for output in successful), integer=False
    )
    reads = [output for output in successful if output.storage_read_tokens]
    result["avg_storage_read_latency_ms"] = (
        latency / len(reads) if latency is not None and reads else None
    )
    return result


def summarize_flat_cache(outputs, total_prompt_tokens):
    result, _ = _prefetch_values(_usage_result(outputs))
    result.update(
        dict.fromkeys(
            (
                "flat_cached_tokens_dram",
                "flat_cached_tokens_ssd",
                "flat_cached_tokens_mixed",
                "flat_cached_tokens_total",
                "device_hit_rate_pct",
                "flat_dram_hit_rate_pct",
                "flat_ssd_hit_rate_pct",
                "flat_storage_hit_rate_pct",
            )
        )
    )
    return result


def summarize_flat_io(before, after, window_id, expected_tp_size=None):
    return _summarize_flat_io(before, after, window_id, expected_tp_size)


class FlatMemoryIOWindow:
    """Own only a UUID window; telemetry failure never discards completed requests."""

    def __init__(
        self, base_url, enabled, *, headers=None, expected_tp_size=None, mode=None
    ):
        self.base_url = base_url.rstrip("/")
        self.enabled = bool(
            enabled
            and mode
            and mode.get("metrics_mode") == "flat"
            and mode.get("metrics_mode_status") == "resolved"
        )
        self.headers = dict(headers or {})
        self.expected_tp_size = expected_tp_size
        self.window_id = str(uuid.uuid4())
        self.begin_attempted = self.ready = False
        self.server_info_before = self.server_info_after = None
        self.result = dict.fromkeys(_IO_RESULT_FIELDS)
        self.result.update(
            flat_io_status="unavailable",
            flat_io_error="Flat I/O collection not requested or Flat mode not established",
            flat_io_metadata=dict(
                before=None, after=None, server_info_before=None, server_info_after=None
            ),
        )

    async def _request(self, action=None):
        import aiohttp

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

    async def _wait_idle(self, *, before):
        timeout = float(os.environ.get("HICACHE_IO_DRAIN_TIMEOUT", "120"))
        deadline = time.monotonic() + timeout
        while True:
            info = await self._request()
            mode = resolve_metrics_mode(before=info)
            evidence = {
                "tp_size": info.get("tp_size"),
                "cache_source_mode": mode["metrics_mode"],
                "metrics_mode_evidence": mode["metrics_mode_evidence"],
            }
            states = info.get("internal_states")
            if isinstance(states, list):
                evidence["internal_states"] = [
                    (
                        {"avg_spec_accept_length": value}
                        if isinstance(state, dict)
                        and (value := _number(state.get("avg_spec_accept_length")))
                        is not None
                        else {}
                    )
                    for state in states
                ]
            key = "server_info_before" if before else "server_info_after"
            if before:
                self.server_info_before = evidence
            else:
                self.server_info_after = evidence
            self.result["flat_io_metadata"][key] = deepcopy(evidence)
            if (
                mode["metrics_mode"] != "flat"
                or mode["metrics_mode_status"] != "resolved"
            ):
                raise ValueError("Server is no longer in resolved Flat mode")
            tp_size = info.get("tp_size")
            if type(tp_size) is not int or tp_size not in (1, 4, 8):
                raise ValueError("Flat server must report TP1/4/8")
            if self.expected_tp_size is None:
                self.expected_tp_size = tp_size
            elif tp_size != self.expected_tp_size:
                raise ValueError("Flat server TP size changed")
            states = info.get("internal_states")
            if not isinstance(states, list) or len(states) != 1:
                raise ValueError("Flat window requires a single DP service group")
            ranks = _flat_rank_map(states[0].get("flat_memory"), tp_size)
            pending = sum(
                _counter(rank.get(name))
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
        if not self.begin_attempted:
            return
        try:
            # The server validates this UUID on all ranks before changing ownership.
            await self._request("abort")
        except Exception as error:
            warnings.warn(f"Flat I/O abort for {self.window_id} failed: {error}")
        finally:
            self.begin_attempted = False
            self.ready = False

    def _unavailable(self, error):
        self.result.update(dict.fromkeys(_IO_RESULT_FIELDS))
        self.result.update(
            flat_io_status="unavailable",
            flat_io_error=f"{type(error).__name__}: {error}",
        )
        warnings.warn(f"Flat I/O unavailable: {self.result['flat_io_error']}")

    async def __aenter__(self):
        if self.enabled:
            try:
                await self._wait_idle(before=True)
                # A lost begin response still requires an abort for this owned UUID.
                self.begin_attempted = True
                before = await self._request("begin")
                self.result["flat_io_metadata"]["before"] = deepcopy(before)
                _validate_flat_window(
                    before, self.window_id, True, self.expected_tp_size
                )
                self.ready = True
            except Exception as error:
                self._unavailable(error)
                await self._abort()
            except BaseException:
                await self._abort()
                raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self.ready and exc_type is None:
                try:
                    await self._wait_idle(before=False)
                    after = await self._request("end")
                    self.result["flat_io_metadata"]["after"] = deepcopy(after)
                    metrics = summarize_flat_io(
                        self.result["flat_io_metadata"]["before"],
                        after,
                        self.window_id,
                        self.expected_tp_size,
                    )
                    self.result.update(metrics, flat_io_status="ok", flat_io_error=None)
                    self.begin_attempted = self.ready = False
                except Exception as error:
                    self._unavailable(error)
        finally:
            await self._abort()
        return False


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
