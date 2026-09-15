"""Completed native HiCache copies and Mooncake storage measurement windows."""

from __future__ import annotations

import math
import re
import time

import requests

_HOST_IDENTITY = (
    "pid",
    "generation",
    "tp_rank",
    "tp_size",
    "host_capacity_bytes",
    "device_capacity_bytes",
    "bytes_per_token",
)
_PATHS = ("dram_read", "dram_write", "ssd_read", "ssd_write")


def _validate_host_rank(rank, tp_size):
    if not isinstance(rank, dict):
        raise ValueError("Invalid HiCache rank snapshot")
    for name in (*_HOST_IDENTITY, "pending"):
        value = rank.get(name)
        minimum = (
            0 if name in ("generation", "tp_rank", "pending", "bytes_per_token") else 1
        )
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid HiCache rank field: {name}")
    if (
        rank["tp_rank"] not in range(tp_size)
        or rank["tp_size"] != tp_size
        or rank.get("enabled") is not True
        or rank.get("io_backend") != "direct"
        or type(rank.get("idle")) is not bool
    ):
        raise ValueError(
            f"Incomplete or disabled HiCache telemetry: {rank.get('error')}"
        )
    for direction in ("read", "write"):
        counters = rank.get(direction)
        if not isinstance(counters, dict):
            raise ValueError(f"Missing HiCache {direction} counters")
        if any(
            type(counters.get(key)) is not int or counters[key] < 0
            for key in ("bytes", "batches")
        ):
            raise ValueError("Invalid completed Host copy counters")
        elapsed = counters.get("elapsed_ms")
        if (
            type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("Invalid completed Host copy duration")
        if (counters["batches"] == 0) != (counters["bytes"] == 0) or (
            counters["batches"] == 0
        ) != (elapsed == 0):
            raise ValueError("Inconsistent completed Host copy counters")
    pending = rank.get("storage_pending", 0)
    if type(pending) is not int or pending < 0:
        raise ValueError("Invalid storage pipeline pending count")


def fetch_hicache_io_snapshot(base_url, *, timeout=120.0, headers=None):
    deadline = time.perf_counter() + timeout
    last_pending = None
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"HiCache did not drain: {last_pending}")
        response = requests.get(
            base_url + "/server_info", headers=headers, timeout=min(5.0, remaining)
        )
        response.raise_for_status()
        info = response.json()
        if not isinstance(info, dict):
            raise ValueError("Invalid HiCache server-info response")
        if (
            info.get("enable_hierarchical_cache") is not True
            or info.get("disable_radix_cache") is not False
            or info.get("hicache_io_backend") != "direct"
            or (info.get("hicache_mem_layout"), info.get("hicache_storage_backend"))
            not in (("layer_first", None), ("page_first_direct", "mooncake"))
            or info.get("disaggregation_mode") != "null"
            or info.get("dp_size") != 1
            or info.get("pp_size") != 1
        ):
            raise ValueError(
                "Native I/O requires direct HiCache, PP1/DP1, no PD, and either L2-only layer_first or Mooncake page_first_direct"
            )
        tp_size = info.get("tp_size")
        states = info.get("internal_states")
        if (
            type(tp_size) is not int
            or tp_size < 1
            or not isinstance(states, list)
            or len(states) != 1
            or not isinstance(states[0], dict)
        ):
            raise ValueError("Invalid native HiCache TP configuration")
        telemetry = states[0].get("hicache_io")
        if not isinstance(telemetry, dict):
            raise ValueError("Missing native HiCache telemetry")
        ranks = telemetry.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != tp_size:
            raise ValueError("Missing native HiCache TP telemetry")
        for rank in ranks:
            _validate_host_rank(rank, tp_size)
        if len({rank["tp_rank"] for rank in ranks}) != tp_size:
            raise ValueError("Duplicate HiCache TP telemetry")
        last_pending = [
            {
                key: rank.get(key, 0)
                for key in ("tp_rank", "idle", "pending", "storage_pending")
            }
            for rank in ranks
        ]
        if all(
            rank["idle"]
            and rank["pending"] == 0
            and rank.get("storage_pending", 0) == 0
            for rank in ranks
        ):
            fields = (
                "model_path",
                "served_model_name",
                "dtype",
                "kv_cache_dtype",
                "tp_size",
                "pp_size",
                "dp_size",
                "disable_radix_cache",
                "enable_hierarchical_cache",
                "hicache_size",
                "hicache_ratio",
                "hicache_write_policy",
                "hicache_io_backend",
                "hicache_mem_layout",
                "hicache_storage_backend",
                "disaggregation_mode",
                "context_length",
                "max_total_num_tokens",
                "max_req_input_len",
                "mem_fraction_static",
                "chunked_prefill_size",
                "max_running_requests",
                "stream_interval",
                "version",
            )
            return {
                "sampled_at": time.perf_counter(),
                "server_config": {key: info[key] for key in fields if key in info},
                "ranks": sorted(ranks, key=lambda rank: rank["tp_rank"]),
            }
        time.sleep(min(0.05, max(0.0, deadline - time.perf_counter())))


def calculate_hicache_io_metrics(before, after):
    seconds = after["sampled_at"] - before["sampled_at"]
    if (
        not math.isfinite(seconds)
        or seconds <= 0
        or len(before["ranks"]) != len(after["ranks"])
    ):
        raise ValueError("Invalid Host I/O observation window or TP membership")
    totals = {
        direction: {"bytes": 0, "batches": 0, "elapsed_ms": 0.0}
        for direction in ("read", "write")
    }
    deltas = []
    for start, end in zip(before["ranks"], after["ranks"]):
        if any(start[key] != end[key] for key in _HOST_IDENTITY):
            raise ValueError("HiCache process or cache configuration changed")
        delta = {key: end[key] for key in _HOST_IDENTITY}
        for direction in totals:
            counters = {
                key: end[direction][key] - start[direction][key]
                for key in totals[direction]
            }
            if any(
                value < 0 or not math.isfinite(value) for value in counters.values()
            ):
                raise ValueError("HiCache completed counters went backwards")
            if (counters["batches"] == 0) != (counters["bytes"] == 0) or (
                counters["batches"] == 0
            ) != (counters["elapsed_ms"] == 0):
                raise ValueError("Incomplete Host copy accounting")
            delta[direction] = counters
            for key, value in counters.items():
                totals[direction][key] += value
        deltas.append(delta)
    result = {"hicache_io_status": "ok"}
    for direction, counters in totals.items():
        elapsed = counters["elapsed_ms"]
        result[f"dram_{direction}_bw_gbps"] = (
            counters["bytes"] / elapsed / 1e6 if elapsed else 0.0
        )
        result[f"hicache_io_{direction}_bytes"] = counters["bytes"]
        result[f"hicache_io_{direction}_batches"] = counters["batches"]
    reads = totals["read"]
    result["mean_l2_kv_readback_ms"] = (
        reads["elapsed_ms"] / reads["batches"] if reads["batches"] else None
    )
    result["hicache_io_metadata"] = {
        "bandwidth_scope": "sum completed rank-batch bytes / sum copy-stream GPU seconds; not aggregate TP bandwidth",
        "readback_scope": "mean all-layer H2D rank-batch GPU milliseconds; not request latency or additive TTFT",
        "window_scope": "all-TP bytes / snapshot observation seconds including copy drainage and control time",
        "window_duration_s": seconds,
        "window_read_bw_gbps": reads["bytes"] / seconds / 1e9,
        "window_write_bw_gbps": totals["write"]["bytes"] / seconds / 1e9,
        "totals": totals,
        "rank_deltas": deltas,
        "before": before,
        "after": after,
    }
    return result


def validate_mooncake_io_snapshot(snapshot, *, require_ssd=False):
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("schema_version") != 1
        or snapshot.get("enabled") is not True
    ):
        raise ValueError(f"Native Mooncake I/O missing or disabled: {snapshot}")
    for key in ("pid", "sampled_at_ns"):
        if type(snapshot.get(key)) is not int or snapshot[key] <= 0:
            raise ValueError(f"Invalid Mooncake identity: {key}")
    totals = snapshot.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("Missing native Mooncake completed counters")
    for path in _PATHS:
        counters = totals.get(path)
        if not isinstance(counters, dict) or any(
            type(counters.get(key)) is not int or counters[key] < 0
            for key in ("bytes", "ops", "errors")
        ):
            raise ValueError(f"Invalid Mooncake completed counters: {path}")
    if require_ssd:
        if snapshot.get("ssd_io_available") is not True:
            raise ValueError("SSD metrics require the native bucket/io_uring owner")
        window = snapshot.get("window")
        if (
            not isinstance(window, dict)
            or window.get("bucket_ns") != 100_000_000
            or type(window.get("active")) is not bool
        ):
            raise ValueError("Invalid Mooncake 100 ms window")
        for key in ("id", "start_ns", "end_ns"):
            if type(window.get(key)) is not int or window[key] < 0:
                raise ValueError(f"Invalid Mooncake window field: {key}")
        if not isinstance(window.get("counters"), dict) or not isinstance(
            window.get("peak_window_bytes"), dict
        ):
            raise ValueError("Missing native Mooncake window counters")
    return snapshot


def fetch_mooncake_io_snapshot(
    client_url, action="snapshot", window_id=None, owner_pid=None
):
    if action == "snapshot":
        response = requests.get(client_url + "/storage_io", timeout=10)
    else:
        params = {"window_id": window_id, "pid": owner_pid} if action == "end" else None
        response = requests.post(
            client_url + "/storage_io/" + action, params=params, timeout=10
        )
    response.raise_for_status()
    return validate_mooncake_io_snapshot(response.json(), require_ssd=True)


def fetch_mooncake_storage_capacity(master_host, metrics_port):
    response = requests.get(f"http://{master_host}:{metrics_port}/metrics", timeout=5)
    response.raise_for_status()
    result = {}
    for metric, key in (
        ("master_allocated_bytes", "mooncake_dram_used_bytes"),
        ("master_key_count", "mooncake_total_keys"),
    ):
        matches = re.findall(
            rf"^{metric}(?:\{{[^\n]*\}})?\s+([^\s]+)$", response.text, re.MULTILINE
        )
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous Mooncake capacity metric: {metric}")
        value = float(matches[0])
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError(f"Invalid Mooncake capacity metric: {metric}")
        result[key] = int(value)
    return result


def unavailable_mooncake_io_metrics(error):
    result = {"mooncake_io_status": "unavailable", "mooncake_io_error": str(error)}
    for medium in ("mooncake_dram", "ssd"):
        for direction in ("read", "write"):
            for field in ("bw_gbps", "total_bytes", "ops", "errors"):
                result[f"{medium}_{direction}_{field}"] = None
    for key in (
        "ssd_read_peak_bw_gbps",
        "ssd_write_peak_bw_gbps",
        "mooncake_dram_used_bytes",
        "mooncake_total_keys",
        "ssd_used_bytes",
        "ssd_keys",
    ):
        result[key] = None
    return result


def calculate_mooncake_io_metrics(before, after, ssd_before, ssd_after):
    validate_mooncake_io_snapshot(ssd_before, require_ssd=True)
    validate_mooncake_io_snapshot(ssd_after, require_ssd=True)
    start, end = ssd_before["window"], ssd_after["window"]
    if (
        ssd_before["pid"] != ssd_after["pid"]
        or start["id"] <= 0
        or start["id"] != end["id"]
        or start["start_ns"] != end["start_ns"]
        or start["active"] is not True
        or end["active"] is not False
        or end["end_ns"] <= start["start_ns"]
    ):
        raise ValueError("Mooncake process or measurement window changed")
    seconds = after["sampled_at"] - before["sampled_at"]
    if (
        len(before["ranks"]) != len(after["ranks"])
        or not math.isfinite(seconds)
        or seconds <= 0
    ):
        raise ValueError("Invalid Mooncake TP membership or DRAM observation window")
    totals = {
        direction: {key: 0 for key in ("bytes", "ops", "errors")}
        for direction in ("read", "write")
    }
    deltas = []
    for initial, final in zip(before["ranks"], after["ranks"]):
        first = validate_mooncake_io_snapshot(initial.get("mooncake_io"))
        last = validate_mooncake_io_snapshot(final.get("mooncake_io"))
        if (
            any(
                initial[key] != final[key]
                for key in ("tp_rank", "tp_size", "pid", "generation")
            )
            or first["pid"] != last["pid"]
            or last["pid"] != final["pid"]
            or last["sampled_at_ns"] <= first["sampled_at_ns"]
        ):
            raise ValueError("Mooncake TP counter owner changed")
        delta = {"tp_rank": final["tp_rank"], "pid": last["pid"]}
        for direction in totals:
            path = "dram_" + direction
            delta[direction] = {}
            for key in totals[direction]:
                value = last["totals"][path][key] - first["totals"][path][key]
                if value < 0:
                    raise ValueError("Mooncake completed counters went backwards")
                totals[direction][key] += value
                delta[direction][key] = value
        deltas.append(delta)
    result = {
        "mooncake_io_status": "ok",
        "mooncake_dram_used_bytes": None,
        "mooncake_total_keys": None,
    }
    for direction, counters in totals.items():
        result[f"mooncake_dram_{direction}_bw_gbps"] = counters["bytes"] / seconds / 1e9
        for key, value in counters.items():
            result[
                f"mooncake_dram_{direction}_{'total_bytes' if key == 'bytes' else key}"
            ] = value
    ssd_seconds = (end["end_ns"] - start["start_ns"]) / 1e9
    for direction in ("read", "write"):
        path = "ssd_" + direction
        counters = end["counters"].get(path)
        peak = end["peak_window_bytes"].get(path)
        if not isinstance(counters, dict) or any(
            type(counters.get(key)) is not int or counters[key] < 0
            for key in ("bytes", "ops", "errors")
        ):
            raise ValueError(f"Invalid SSD completion counters: {path}")
        if type(peak) is not int or not 0 <= peak <= counters["bytes"]:
            raise ValueError(f"Invalid SSD peak window: {path}")
        for key in ("bytes", "ops", "errors"):
            result[f"{path}_{'total_bytes' if key == 'bytes' else key}"] = counters[key]
        result[f"{path}_bw_gbps"] = counters["bytes"] / ssd_seconds / 1e9
        result[f"{path}_peak_bw_gbps"] = peak / (end["bucket_ns"] / 1e9) / 1e9
    for key in ("ssd_used_bytes", "ssd_keys"):
        value = ssd_after.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"Invalid SSD capacity: {key}")
        result[key] = value
    result["mooncake_io_metadata"] = {
        "dram_scope": "all-TP completed memory-replica bytes / observation seconds; excludes SSD staging",
        "dram_window_seconds": seconds,
        "ssd_scope": "native bucket O_DIRECT positive read CQEs and datasync-successful bucket data writes, before metadata persistence",
        "ssd_window_seconds": ssd_seconds,
        "peak_window_ms": 100,
        "peak_scope": "single storage-owner process, fixed 100 ms completed-byte buckets; not summed rank peaks",
        "measurement_scope": "after warmup/flush through post-request Host/backup drainage; not physical device bandwidth",
        "rank_deltas": deltas,
        "ssd_before": ssd_before,
        "ssd_after": ssd_after,
    }
    return result


def print_mooncake_io_metrics(metrics):
    print("Mooncake Storage I/O Statistics".center(62, "-"))
    print("Storage DRAM is separate from GPU/L2 Host copies; GB = 10^9 bytes.")
    if metrics["mooncake_io_status"] != "ok":
        print("Native storage I/O unavailable: " + metrics["mooncake_io_error"])
    rows = [
        ("DRAM used (GB):", "mooncake_dram_used_bytes", 1e9),
        ("SSD used (GB, metadata):", "ssd_used_bytes", 1e9),
        ("Mooncake keys:", "mooncake_total_keys", 1),
    ]
    for medium, prefix in (("DRAM", "mooncake_dram"), ("SSD", "ssd")):
        for direction in ("write", "read"):
            rows.append(
                (
                    f"{medium} {direction} bandwidth (GB/s):",
                    f"{prefix}_{direction}_bw_gbps",
                    1,
                )
            )
            if medium == "SSD":
                rows.append(
                    (
                        f"SSD {direction} peak (GB/s, 100 ms):",
                        f"ssd_{direction}_peak_bw_gbps",
                        1,
                    )
                )
            for key in ("ops", "errors"):
                rows.append(
                    (f"{medium} {direction} {key}:", f"{prefix}_{direction}_{key}", 1)
                )
            rows.append(
                (
                    f"{medium} {direction} total (GB):",
                    f"{prefix}_{direction}_total_bytes",
                    1e9,
                )
            )
    for label, key, divisor in rows:
        value = metrics.get(key)
        shown = "N/A (unavailable)" if value is None else f"{value / divisor:.4f}"
        print(f"{label:<48} {shown}")
    print(
        "SSD write completion is bucket datasync, not full metadata persistence or device physical peak."
    )


class NativeIOMeasurement:
    def __init__(
        self,
        *,
        base_url,
        collect_host,
        collect_storage,
        headers,
        master_host=None,
        master_port=9003,
        client_url=None,
        drain_timeout=120.0,
        flush_timeout=150.0,
    ):
        self.base_url = base_url
        self.enabled = bool(collect_host or collect_storage)
        self.storage_enabled = bool(collect_storage)
        self.headers = headers
        self.master_host = master_host
        self.master_port = master_port
        self.client_url = client_url
        self.drain_timeout = drain_timeout
        self.flush_timeout = flush_timeout
        self.before = None
        self.after = None
        self.ssd_before = None
        self.ssd_after = None
        self.host_metrics = {}
        self.storage_metrics = {}

    def validate(self, *, backend):
        if not self.enabled:
            return
        if backend != "sglang":
            raise ValueError("Native HiCache I/O metrics require --backend sglang")
        if not (
            math.isfinite(self.drain_timeout)
            and self.drain_timeout > 0
            and math.isfinite(self.flush_timeout)
            and self.flush_timeout > self.drain_timeout
        ):
            raise ValueError(
                "HICACHE_FLUSH_TIMEOUT must exceed positive HICACHE_IO_DRAIN_TIMEOUT"
            )
        if self.storage_enabled:
            if not self.master_host:
                raise ValueError(
                    "--collect-mooncake-io-metrics requires --mooncake-master-host"
                )
            fetch_mooncake_io_snapshot(self.client_url)

    def start(self):
        if not self.enabled:
            return
        self.before = fetch_hicache_io_snapshot(
            self.base_url, timeout=self.drain_timeout, headers=self.headers
        )
        if self.storage_enabled:
            if (
                self.before["server_config"].get("hicache_storage_backend")
                != "mooncake"
            ):
                raise ValueError("Native Mooncake metrics require the Mooncake backend")
            for rank in self.before["ranks"]:
                native = validate_mooncake_io_snapshot(rank.get("mooncake_io"))
                if native["pid"] != rank["pid"]:
                    raise ValueError(
                        "Mooncake counter owner does not match the TP worker"
                    )
            self.ssd_before = fetch_mooncake_io_snapshot(self.client_url, "begin")

    def _end_storage_window(self):
        self.ssd_after = fetch_mooncake_io_snapshot(
            self.client_url,
            "end",
            self.ssd_before["window"]["id"],
            self.ssd_before["pid"],
        )

    def abort(self):
        if self.ssd_before is not None and self.ssd_after is None:
            try:
                self._end_storage_window()
            except (requests.RequestException, ValueError) as exc:
                print(f"Could not close this benchmark's Mooncake I/O window: {exc}")

    def finish(self):
        if not self.enabled:
            return
        try:
            self.after = fetch_hicache_io_snapshot(
                self.base_url, timeout=self.drain_timeout, headers=self.headers
            )
            self.host_metrics = calculate_hicache_io_metrics(self.before, self.after)
        except (requests.RequestException, TimeoutError, ValueError) as exc:
            self.host_metrics = {
                "hicache_io_status": "unavailable",
                "hicache_io_error": str(exc),
                "dram_read_bw_gbps": None,
                "dram_write_bw_gbps": None,
                "mean_l2_kv_readback_ms": None,
                **{
                    f"hicache_io_{direction}_{field}": None
                    for direction in ("read", "write")
                    for field in ("bytes", "batches")
                },
                "hicache_io_metadata": {
                    "before": self.before,
                    "after": self.after,
                    "error": str(exc),
                },
            }
        if not self.storage_enabled:
            return
        try:
            # FLAT_MEMORY: EndWindow itself does not drain; the Host snapshot above must.
            self._end_storage_window()
            if self.after is None or self.host_metrics["hicache_io_status"] != "ok":
                raise ValueError("Completed TP I/O snapshots unavailable")
            self.storage_metrics = calculate_mooncake_io_metrics(
                self.before, self.after, self.ssd_before, self.ssd_after
            )
        except (requests.RequestException, ValueError) as exc:
            self.storage_metrics = unavailable_mooncake_io_metrics(exc)
            self.storage_metrics["mooncake_io_metadata"] = {
                "ssd_before": self.ssd_before,
                "ssd_after": self.ssd_after,
                "tp_before": self.before,
                "tp_after": self.after,
            }
        try:
            self.storage_metrics.update(
                fetch_mooncake_storage_capacity(self.master_host, self.master_port)
            )
        except (requests.RequestException, ValueError) as exc:
            self.storage_metrics["mooncake_io_metadata"]["capacity_error"] = str(exc)

    def report(self):
        if self.storage_enabled:
            print_mooncake_io_metrics(self.storage_metrics)
        if not self.enabled:
            return
        if self.host_metrics["hicache_io_status"] != "ok":
            print(
                "Native HiCache I/O unavailable: "
                + self.host_metrics["hicache_io_error"]
            )
            return
        print("HiCache L2 Host I/O Statistics".center(62, "-"))
        print("Rank-pooled copy-service bandwidth, not aggregate TP bandwidth.")
        for direction in ("read", "write"):
            bandwidth = self.host_metrics[f"dram_{direction}_bw_gbps"]
            byte_count = self.host_metrics[f"hicache_io_{direction}_bytes"]
            batches = self.host_metrics[f"hicache_io_{direction}_batches"]
            print(
                f"L2 Host {direction}: {bandwidth:.4f} GB/s, {byte_count} bytes, {batches} rank-batches"
            )
        print(
            f"Mean all-layer H2D rank-batch time: {self.host_metrics['mean_l2_kv_readback_ms']} ms; not an additive TTFT component."
        )
