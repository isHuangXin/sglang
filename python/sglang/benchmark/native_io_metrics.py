"""FLAT_MEMORY: Native Host, owner storage, and dedicated consumer GDS collectors."""

import math
import os
import time
import uuid
import warnings

import requests


def _auth_headers(headers):
    if headers is not None:
        return headers
    token = os.environ.get("OPENAI_API_KEY")
    if token:
        return {"Authorization": f"Bearer {token}"}
    token = os.environ.get("API_KEY")
    return {"Authorization": token} if token else {}


def sanitize_server_info(info: dict | None) -> dict | None:
    """Keep scalar benchmark configuration, not credentials or runtime state."""
    if not isinstance(info, dict):
        return None
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
        name: info[name]
        for name in fields
        if name in info
        and (info[name] is None or type(info[name]) in (str, bool, int, float))
    }


def fetch_hicache_io_snapshot(
    base_url: str, timeout: float = 120.0, *, headers=None
) -> dict:
    deadline = time.perf_counter() + timeout
    last_pending = None
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(
                f"HiCache workers or Host copies did not become idle: {last_pending}"
            )
        response = requests.get(
            base_url + "/server_info",
            headers=_auth_headers(headers),
            timeout=min(5.0, remaining),
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
                "Host I/O metrics require native direct HiCache: layer_first without storage, or page_first_direct with Mooncake; no PD"
            )
        tp_size, states = info.get("tp_size"), info.get("internal_states")
        if (
            type(tp_size) is not int
            or tp_size < 1
            or not isinstance(states, list)
            or len(states) != 1
            or not isinstance(states[0], dict)
        ):
            raise ValueError("Invalid HiCache worker configuration")
        telemetry = states[0].get("hicache_io")
        if not isinstance(telemetry, dict):
            raise ValueError("Missing native HiCache telemetry")
        ranks = telemetry.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != tp_size:
            raise ValueError("Missing HiCache telemetry from one or more TP ranks")
        seen = set()
        for rank in ranks:
            if not isinstance(rank, dict):
                raise ValueError("Invalid HiCache rank snapshot")
            for name in (
                "tp_rank",
                "tp_size",
                "pid",
                "generation",
                "pending",
                "host_capacity_bytes",
                "device_capacity_bytes",
                "bytes_per_token",
            ):
                value = rank.get(name)
                minimum = 0 if name in ("tp_rank", "generation", "pending") else 1
                if type(value) is not int or value < minimum:
                    raise ValueError(f"Invalid HiCache rank field: {name}")
            if (
                rank["tp_rank"] not in range(tp_size)
                or rank["tp_rank"] in seen
                or rank["tp_size"] != tp_size
                or rank.get("enabled") is not True
                or rank.get("io_backend") != "direct"
                or type(rank.get("idle")) is not bool
            ):
                raise ValueError("Incomplete or disabled HiCache TP telemetry")
            seen.add(rank["tp_rank"])
            for direction in ("read", "write"):
                counters = rank.get(direction)
                if not isinstance(counters, dict):
                    raise ValueError(f"Missing HiCache {direction} counters")
                for name in ("bytes", "batches"):
                    if type(counters.get(name)) is not int or counters[name] < 0:
                        raise ValueError(f"Invalid HiCache {direction} {name}")
                elapsed = counters.get("elapsed_ms")
                if (
                    type(elapsed) not in (int, float)
                    or not math.isfinite(elapsed)
                    or elapsed < 0
                ):
                    raise ValueError(f"Invalid HiCache {direction} timing")
                if (counters["batches"] == 0) != (counters["bytes"] == 0) or (
                    counters["batches"] == 0
                ) != (elapsed == 0):
                    raise ValueError(
                        f"Inconsistent completed HiCache {direction} counters"
                    )
        if info.get("hicache_storage_backend") == "mooncake" and any(
            "storage_pending" not in rank for rank in ranks
        ):
            raise ValueError("Missing Mooncake storage pipeline pending count")
        if any(
            type(rank.get("storage_pending", 0)) is not int
            or rank.get("storage_pending", 0) < 0
            for rank in ranks
        ):
            raise ValueError("Invalid storage pipeline pending count")
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
            return {
                "sampled_at": time.perf_counter(),
                "server_config": sanitize_server_info(info),
                "ranks": sorted(ranks, key=lambda rank: rank["tp_rank"]),
            }
        time.sleep(min(0.05, max(0.0, deadline - time.perf_counter())))


def calculate_hicache_io_metrics(before: dict, after: dict) -> dict:
    window_seconds = after["sampled_at"] - before["sampled_at"]
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("Invalid Host I/O observation window")
    if len(before["ranks"]) != len(after["ranks"]):
        raise ValueError("HiCache TP membership changed during benchmark")
    totals = {
        direction: {"bytes": 0, "batches": 0, "elapsed_ms": 0.0}
        for direction in ("read", "write")
    }
    deltas = []
    for start, end in zip(before["ranks"], after["ranks"]):
        identity = (
            "pid",
            "generation",
            "tp_rank",
            "tp_size",
            "host_capacity_bytes",
            "device_capacity_bytes",
            "bytes_per_token",
        )
        if any(start[name] != end[name] for name in identity):
            raise ValueError(
                "HiCache worker or cache configuration changed during benchmark"
            )
        delta = {name: end[name] for name in identity}
        for direction in ("read", "write"):
            delta[direction] = {
                name: end[direction][name] - start[direction][name]
                for name in ("bytes", "batches", "elapsed_ms")
            }
            counters = delta[direction]
            if any(
                value < 0 or not math.isfinite(value) for value in counters.values()
            ):
                raise ValueError("HiCache completed counters went backwards")
            if (counters["batches"] == 0) != (counters["bytes"] == 0) or (
                counters["batches"] == 0
            ) != (counters["elapsed_ms"] == 0):
                raise ValueError("HiCache copy accounting is incomplete")
            for name, value in counters.items():
                totals[direction][name] += value
        deltas.append(delta)
    result = {"hicache_io_status": "ok"}
    for direction, counters in totals.items():
        elapsed = counters["elapsed_ms"]
        result[f"dram_{direction}_bw_gbps"] = (
            counters["bytes"] / elapsed / 1e6 if elapsed > 0 else 0.0
        )
        result[f"hicache_io_{direction}_bytes"] = counters["bytes"]
        result[f"hicache_io_{direction}_batches"] = counters["batches"]
    reads = totals["read"]
    result["mean_l2_kv_readback_ms"] = (
        reads["elapsed_ms"] / reads["batches"] if reads["batches"] else None
    )
    result["hicache_io_metadata"] = {
        "bandwidth_scope": "sum completed rank-batch bytes / sum copy-stream GPU seconds / 1e9; not aggregate TP bandwidth",
        "readback_scope": "mean completed all-layer H2D rank-batch GPU milliseconds; not request latency or exposed TTFT penalty",
        "window_scope": "all-TP bytes / snapshot observation seconds, including post-request copy drainage and control-query time",
        "window_duration_s": window_seconds,
        "window_read_bw_gbps": reads["bytes"] / window_seconds / 1e9,
        "window_write_bw_gbps": totals["write"]["bytes"] / window_seconds / 1e9,
        "totals": totals,
        "rank_deltas": deltas,
        "before": before,
        "after": after,
    }
    return result


def validate_mooncake_io_snapshot(snapshot: dict, require_ssd: bool = False) -> dict:
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("schema_version") != 1
        or snapshot.get("enabled") is not True
    ):
        raise ValueError("Native Mooncake I/O metrics are missing or disabled")
    for field in ("pid", "sampled_at_ns"):
        if type(snapshot.get(field)) is not int or snapshot[field] <= 0:
            raise ValueError(f"Invalid native Mooncake identity: {field}")
    totals = snapshot.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("Missing Mooncake completed counters")
    for path in ("dram_read", "dram_write", "ssd_read", "ssd_write"):
        counters = totals.get(path)
        if not isinstance(counters, dict) or any(
            type(counters.get(key)) is not int or counters[key] < 0
            for key in ("bytes", "ops", "errors")
        ):
            raise ValueError(f"Invalid Mooncake completed counters: {path}")
    if require_ssd:
        if snapshot.get("ssd_io_available") is not True:
            raise ValueError(
                "SSD metrics require the native bucket/io_uring storage owner"
            )
        window = snapshot.get("window")
        if (
            not isinstance(window, dict)
            or window.get("bucket_ns") != 100_000_000
            or type(window.get("active")) is not bool
        ):
            raise ValueError("Invalid Mooncake 100 ms I/O window")
        for field in ("id", "start_ns", "end_ns"):
            if type(window.get(field)) is not int or window[field] < 0:
                raise ValueError(f"Invalid Mooncake window field: {field}")
        if not isinstance(window.get("counters"), dict) or not isinstance(
            window.get("peak_window_bytes"), dict
        ):
            raise ValueError("Missing Mooncake window counters or peaks")
    return snapshot


def fetch_mooncake_io_snapshot(
    client_url: str,
    action: str = "snapshot",
    window_id: int = None,
    owner_pid: int = None,
) -> dict:
    if action == "snapshot":
        response = requests.get(client_url + "/storage_io", timeout=10)
    else:
        params = {"window_id": window_id, "pid": owner_pid} if action == "end" else None
        response = requests.post(
            client_url + "/storage_io/" + action, params=params, timeout=10
        )
    response.raise_for_status()
    return validate_mooncake_io_snapshot(response.json(), require_ssd=True)


def fetch_mooncake_storage_capacity(master_host: str, metrics_port: int) -> dict:
    import re

    response = requests.get(f"http://{master_host}:{metrics_port}/metrics", timeout=5)
    response.raise_for_status()
    result = {}
    for metric, field in (
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
        result[field] = int(value)
    return result


def unavailable_mooncake_io_metrics(error: Exception) -> dict:
    result = {"mooncake_io_status": "unavailable", "mooncake_io_error": str(error)}
    for medium in ("mooncake_dram", "ssd"):
        for direction in ("read", "write"):
            for field in ("bw_gbps", "total_bytes", "ops", "errors"):
                result[f"{medium}_{direction}_{field}"] = None
    for field in (
        "ssd_read_peak_bw_gbps",
        "ssd_write_peak_bw_gbps",
        "mooncake_dram_used_bytes",
        "mooncake_total_keys",
        "ssd_used_bytes",
        "ssd_keys",
    ):
        result[field] = None
    return result


def calculate_mooncake_io_metrics(
    before: dict, after: dict, ssd_before: dict, ssd_after: dict
) -> dict:
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
    if len(before["ranks"]) != len(after["ranks"]):
        raise ValueError("Mooncake TP membership changed")
    seconds = after["sampled_at"] - before["sampled_at"]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid Mooncake DRAM observation window")
    totals = {
        direction: {field: 0 for field in ("bytes", "ops", "errors")}
        for direction in ("read", "write")
    }
    rank_deltas = []
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
            raise ValueError("Mooncake TP worker changed during measurement")
        delta = {"tp_rank": final["tp_rank"], "pid": last["pid"]}
        for direction in ("read", "write"):
            path = "dram_" + direction
            delta[direction] = {}
            for field in ("bytes", "ops", "errors"):
                value = last["totals"][path][field] - first["totals"][path][field]
                if value < 0:
                    raise ValueError("Mooncake completed counters went backwards")
                totals[direction][field] += value
                delta[direction][field] = value
        rank_deltas.append(delta)
    result = {
        "mooncake_io_status": "ok",
        "mooncake_dram_used_bytes": None,
        "mooncake_total_keys": None,
    }
    for direction, counters in totals.items():
        result[f"mooncake_dram_{direction}_bw_gbps"] = counters["bytes"] / seconds / 1e9
        for field, value in counters.items():
            name = "total_bytes" if field == "bytes" else field
            result[f"mooncake_dram_{direction}_{name}"] = value
    ssd_seconds = (end["end_ns"] - start["start_ns"]) / 1e9
    for direction in ("read", "write"):
        path = "ssd_" + direction
        counters = end.get("counters", {}).get(path)
        peak = end.get("peak_window_bytes", {}).get(path)
        if not isinstance(counters, dict) or any(
            type(counters.get(key)) is not int or counters[key] < 0
            for key in ("bytes", "ops", "errors")
        ):
            raise ValueError(f"Invalid SSD window counters: {path}")
        if type(peak) is not int or peak < 0 or peak > counters["bytes"]:
            raise ValueError(f"Invalid SSD peak window: {path}")
        for field, value in counters.items():
            if field in ("bytes", "ops", "errors"):
                name = "total_bytes" if field == "bytes" else field
                result[f"{path}_{name}"] = value
        result[f"{path}_bw_gbps"] = counters["bytes"] / ssd_seconds / 1e9
        result[f"{path}_peak_bw_gbps"] = peak / (end["bucket_ns"] / 1e9) / 1e9
    for field in ("ssd_used_bytes", "ssd_keys"):
        value = ssd_after.get(field)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"Invalid SSD capacity: {field}")
        result[field] = value
    result["mooncake_io_metadata"] = {
        "dram_scope": "all-TP completed memory-replica bytes / observation seconds; excludes SSD staging transfers",
        "dram_window_seconds": seconds,
        "ssd_scope": "native bucket O_DIRECT read completions and datasync-successful bucket data writes; excludes consumer cuFile reads; not device physical bandwidth",
        "ssd_window_seconds": ssd_seconds,
        "peak_window_ms": 100,
        "peak_scope": "all-thread completed bytes per fixed 100 ms bucket; final partial bucket divided by 100 ms",
        "ops_scope": "DRAM memory-replica objects; SSD reads are completed read CQEs; SSD writes are successfully persisted buckets",
        "measurement_scope": "after warmup/flush through post-request Host/backup drainage; asynchronous offload counted by completion time",
        "rank_deltas": rank_deltas,
        "ssd_before": ssd_before,
        "ssd_after": ssd_after,
    }
    return result


def print_mooncake_io_metrics(metrics: dict) -> None:
    print("Mooncake Storage I/O Statistics".center(62, "-"))
    print("Storage DRAM is separate from GPU<->L2 Host copies; GB = 10^9 bytes.")
    print("Owner SSD read counters exclude GDS (compat) reads in TP consumers.")
    if metrics["mooncake_io_status"] != "ok":
        print("Native storage I/O unavailable: " + metrics["mooncake_io_error"])
    rows = [
        ("DRAM used (GB):", "mooncake_dram_used_bytes", 1e9),
        ("SSD used (GB, metadata):", "ssd_used_bytes", 1e9),
        ("Total blocks stored (Mooncake keys):", "mooncake_total_keys", 1),
    ]
    for medium, prefix in (("DRAM", "mooncake_dram"), ("Owner SSD", "ssd")):
        for direction in ("write", "read"):
            rows.append(
                (
                    f"{medium} {direction} bandwidth (GB/s):",
                    f"{prefix}_{direction}_bw_gbps",
                    1,
                )
            )
            if prefix == "ssd":
                rows.append(
                    (
                        f"Owner SSD {direction} bandwidth peak (GB/s, 100 ms):",
                        f"ssd_{direction}_peak_bw_gbps",
                        1,
                    )
                )
            rows.extend(
                [
                    (f"{medium} {direction} ops:", f"{prefix}_{direction}_ops", 1),
                    (
                        f"{medium} {direction} total (GB):",
                        f"{prefix}_{direction}_total_bytes",
                        1e9,
                    ),
                    (
                        f"{medium} {direction} errors:",
                        f"{prefix}_{direction}_errors",
                        1,
                    ),
                ]
            )
    for label, key, divisor in rows:
        value = metrics.get(key)
        shown = (
            "N/A (unavailable)"
            if value is None
            else (
                str(value)
                if type(value) is int and divisor == 1
                else f"{value / divisor:.4f}"
            )
        )
        print(f"{label:<48} {shown}")
    print(
        "SSD writes/peak: datasync-successful bucket data, not buffered-write or device physical peak."
    )


def validate_gds_io_window(snapshot, window_id, active):
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("ranks"), list):
        raise ValueError("Missing GDS TP snapshots")
    ranks = snapshot["ranks"]
    if len(ranks) not in (1, 4, 8):
        raise ValueError("GDS I/O requires all TP1/TP4/TP8 ranks")
    mapped, pids, boundaries = {}, set(), set()
    for rank in ranks:
        validate_mooncake_io_snapshot(rank)
        for key in ("tp_rank", "tp_size", "gpu_id", "generation"):
            if type(rank.get(key)) is not int or rank[key] < 0:
                raise ValueError(f"Invalid GDS rank field: {key}")
        if (
            rank["tp_rank"] not in range(len(ranks))
            or rank["tp_rank"] in mapped
            or rank["pid"] in pids
            or rank["tp_size"] != len(ranks)
            or rank.get("gds_mode") != "compat"
        ):
            raise ValueError("Incomplete or duplicate GDS TP identities")
        mapped[rank["tp_rank"]] = rank
        pids.add(rank["pid"])
        window = rank.get("window")
        if (
            not isinstance(window, dict)
            or type(window.get("id")) is not int
            or window["id"] != int(window_id)
            or window.get("active") is not active
            or window.get("capture_buckets") is not True
            or window.get("aborted") is not False
            or window.get("overflowed") is not False
            or type(window.get("bucket_ns")) is not int
            or window["bucket_ns"] != 100_000_000
        ):
            raise ValueError(
                "Mismatched, aborted, overflowing or unsupported GDS window"
            )
        start, end = window.get("start_ns"), window.get("end_ns")
        if (
            type(start) is not int
            or start <= 0
            or start > rank["sampled_at_ns"]
            or type(end) is not int
            or (active and end != 0)
            or (not active and (end <= start or end > rank["sampled_at_ns"]))
        ):
            raise ValueError("Invalid GDS window boundaries")
        boundaries.add((start, end))
        window_counters = window.get("counters")
        if not isinstance(window_counters, dict):
            raise ValueError("Missing GDS window counters")
        counters = window_counters.get("ssd_read")
        if not isinstance(counters, dict) or any(
            type(counters.get(key)) is not int or counters[key] < 0
            for key in ("bytes", "ops", "errors")
        ):
            raise ValueError("Missing or invalid GDS read counters")
        buckets = window.get("buckets")
        if not isinstance(buckets, list):
            raise ValueError("Missing GDS completion buckets")
        seen, total = set(), 0
        for bucket in buckets:
            if not isinstance(bucket, dict):
                raise ValueError("Invalid GDS bucket")
            index, byte_count = bucket.get("index"), bucket.get("ssd_read")
            if (
                type(index) is not int
                or not 0 <= index < 72000
                or index in seen
                or type(byte_count) is not int
                or byte_count < 0
                or (not active and index > (end - start) // 100_000_000)
            ):
                raise ValueError("Invalid, duplicate or out-of-window GDS bucket")
            seen.add(index)
            total += byte_count
        if total != counters["bytes"]:
            raise ValueError("GDS bucket bytes do not match window bytes")
        if active and (any(counters.values()) or buckets):
            raise ValueError("GDS begin window is not empty")
    if len(boundaries) != 1:
        raise ValueError("GDS TP ranks do not share common boundaries")
    return mapped, next(iter(boundaries))


def unavailable_gds_io_metrics(error):
    result = dict.fromkeys(
        (
            "gds_io_window_seconds",
            "gds_ssd_read_bw_gbps",
            "gds_ssd_read_peak_bw_gbps",
            "gds_ssd_read_ops",
            "gds_ssd_read_total_bytes",
            "gds_ssd_read_errors",
        )
    )
    result.update(gds_io_status="unavailable", gds_io_error=str(error))
    return result


def calculate_gds_io_metrics(before, after, window_id):
    first, start = validate_gds_io_window(before, window_id, True)
    last, end = validate_gds_io_window(after, window_id, False)
    if first.keys() != last.keys() or start[0] != end[0]:
        raise ValueError("GDS TP membership or start boundary changed")
    totals = dict.fromkeys(("bytes", "ops", "errors"), 0)
    buckets = {}
    for rank_id, rank in last.items():
        initial = first[rank_id]
        if (
            any(
                initial[key] != rank[key]
                for key in ("pid", "tp_size", "gpu_id", "generation")
            )
            or rank["sampled_at_ns"] <= initial["sampled_at_ns"]
        ):
            raise ValueError("GDS worker, clock or cache generation changed")
        counters = rank["window"]["counters"]["ssd_read"]
        for key in totals:
            if (
                rank["totals"]["ssd_read"][key] - initial["totals"]["ssd_read"][key]
                < counters[key]
            ):
                raise ValueError(
                    "GDS lifetime counters reset or disagree with the window"
                )
            totals[key] += counters[key]
        for bucket in rank["window"]["buckets"]:
            index = bucket["index"]
            buckets[index] = buckets.get(index, 0) + bucket["ssd_read"]
    seconds = (end[1] - end[0]) / 1e9
    return {
        "gds_io_status": "ok",
        "gds_io_error": None,
        "gds_io_window_seconds": seconds,
        "gds_ssd_read_bw_gbps": totals["bytes"] / seconds / 1e9,
        "gds_ssd_read_peak_bw_gbps": max(buckets.values(), default=0) / 0.1 / 1e9,
        "gds_ssd_read_ops": totals["ops"],
        "gds_ssd_read_total_bytes": totals["bytes"],
        "gds_ssd_read_errors": totals["errors"],
        "gds_io_metadata": {
            "window_id": str(window_id),
            "start_ns": end[0],
            "end_ns": end[1],
            "bucket_ns": 100_000_000,
            "bandwidth_scope": "all-TP cuFile returned bytes / common observation seconds; includes alignment, not device physical bandwidth",
            "peak_scope": "max(sum same-index TP completion bytes) / 100ms; partial final bucket uses 100ms",
            "ops_scope": "cuFileRead calls including failures; errors also include range/file/copy failures",
            "buckets": [
                {"index": index, "ssd_read_bytes": value}
                for index, value in sorted(buckets.items())
            ],
            "before": before,
            "after": after,
        },
    }


def fetch_gds_io_window(base_url, action, window_id, timeout=150, *, headers=None):
    response = requests.post(
        base_url + "/mooncake/gds_io_window",
        headers=_auth_headers(headers),
        json={"action": action, "window_id": str(window_id)},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def print_gds_io_metrics(metrics):
    print("Consumer GDS (compat) SSD Read I/O Statistics".center(62, "-"))
    for label, key, divisor in (
        ("SSD read bandwidth (GB/s):", "gds_ssd_read_bw_gbps", 1),
        ("SSD read bandwidth peak (GB/s, 100 ms):", "gds_ssd_read_peak_bw_gbps", 1),
        ("SSD read ops (cuFile calls):", "gds_ssd_read_ops", 1),
        ("SSD read total (GB):", "gds_ssd_read_total_bytes", 1e9),
        ("SSD read errors:", "gds_ssd_read_errors", 1),
    ):
        value = metrics[key]
        shown = (
            "N/A (unavailable)"
            if value is None
            else (
                str(value)
                if type(value) is int and divisor == 1
                else f"{value / divisor:.4f}"
            )
        )
        print(f"{label:<48} {shown}")
    print("GDS I/O status: " + metrics["gds_io_status"])
    if metrics["gds_io_error"]:
        print("GDS I/O detail: " + metrics["gds_io_error"])


class NativeIOWindow:
    def __init__(
        self,
        base_url,
        backend,
        *,
        collect_mooncake_io_metrics=False,
        collect_mooncake_gds_io=False,
        collect_hicache_io_metrics=False,
        collect_flat_memory_io=False,
        mooncake_master_host=None,
        mooncake_metrics_port=9003,
        mooncake_client_host="127.0.0.1",
        mooncake_client_metrics_port=9301,
        headers=None,
    ):
        self.base_url = base_url
        self.headers = _auth_headers(headers)
        self.master_host = mooncake_master_host
        self.metrics_port = mooncake_metrics_port
        self.collect_mooncake = collect_mooncake_io_metrics
        self.collect_gds = collect_mooncake_gds_io
        self.collect_host = (
            self.collect_mooncake or self.collect_gds or collect_hicache_io_metrics
        )
        self.before = self.after = self.ssd_before = self.ssd_after = None
        self.gds_before = self.gds_after = None
        self.gds_window_id = (
            str(uuid.uuid4().int % (2**63 - 1) + 1) if self.collect_gds else None
        )
        self.gds_begin_attempted = False
        self.host_result = {}
        self.mooncake_result = {}
        self.gds_result = {}
        self.timeout = 120.0
        self.flush_timeout = 150.0
        if not self.collect_host:
            return
        if backend != "sglang":
            raise ValueError(
                "Native HiCache/Mooncake I/O metrics require --backend sglang"
            )
        if collect_flat_memory_io:
            raise ValueError(
                "Native Mooncake/HiCache I/O and Flat I/O windows cannot be combined"
            )
        self.timeout = float(os.environ.get("HICACHE_IO_DRAIN_TIMEOUT", "120"))
        self.flush_timeout = float(os.environ.get("HICACHE_FLUSH_TIMEOUT", "150"))
        if not (
            math.isfinite(self.timeout)
            and self.timeout > 0
            and math.isfinite(self.flush_timeout)
            and self.flush_timeout > self.timeout
        ):
            raise ValueError(
                "HICACHE_FLUSH_TIMEOUT must exceed a finite positive HICACHE_IO_DRAIN_TIMEOUT"
            )
        if self.collect_mooncake:
            if not mooncake_master_host:
                raise ValueError(
                    "--collect-mooncake-io-metrics requires --mooncake-master-host"
                )
            self.client_url = (
                f"http://{mooncake_client_host}:{mooncake_client_metrics_port}"
            )
            fetch_mooncake_io_snapshot(self.client_url)

    def _abort_gds(self):
        if self.gds_begin_attempted:
            try:
                fetch_gds_io_window(
                    self.base_url,
                    "abort",
                    self.gds_window_id,
                    self.flush_timeout,
                    headers=self.headers,
                )
            except (requests.RequestException, ValueError, TimeoutError) as error:
                warnings.warn(
                    f"Could not abort GDS I/O window {self.gds_window_id}: {error}"
                )
            finally:
                self.gds_begin_attempted = False

    async def __aenter__(self):
        try:
            if self.collect_host:
                self.before = fetch_hicache_io_snapshot(
                    self.base_url, timeout=self.timeout, headers=self.headers
                )
            if self.collect_mooncake or self.collect_gds:
                if (
                    self.before["server_config"].get("hicache_storage_backend")
                    != "mooncake"
                ):
                    raise ValueError(
                        "Mooncake I/O metrics require the Mooncake storage backend"
                    )
            if self.collect_mooncake:
                for rank in self.before["ranks"]:
                    snapshot = validate_mooncake_io_snapshot(rank.get("mooncake_io"))
                    if snapshot["pid"] != rank["pid"]:
                        raise ValueError(
                            "Mooncake counter owner does not match TP worker"
                        )
                self.ssd_before = fetch_mooncake_io_snapshot(self.client_url, "begin")
            if self.collect_gds:
                self.gds_begin_attempted = True
                self.gds_before = fetch_gds_io_window(
                    self.base_url,
                    "begin",
                    self.gds_window_id,
                    self.flush_timeout,
                    headers=self.headers,
                )
                validate_gds_io_window(self.gds_before, self.gds_window_id, True)
        except BaseException as error:
            await self.__aexit__(type(error), error, error.__traceback__)
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if not self.collect_host:
            return False
        if exc_type is None:
            try:
                self.after = fetch_hicache_io_snapshot(
                    self.base_url, timeout=self.timeout, headers=self.headers
                )
                self.host_result = calculate_hicache_io_metrics(self.before, self.after)
            except (requests.RequestException, TimeoutError, ValueError) as error:
                self.host_result = {
                    "hicache_io_status": "unavailable",
                    "hicache_io_error": str(error),
                    "dram_read_bw_gbps": None,
                    "dram_write_bw_gbps": None,
                    "mean_l2_kv_readback_ms": None,
                    "hicache_io_read_bytes": None,
                    "hicache_io_write_bytes": None,
                    "hicache_io_read_batches": None,
                    "hicache_io_write_batches": None,
                    "hicache_io_metadata": {
                        "before": self.before,
                        "after": self.after,
                        "error": str(error),
                    },
                }
        if self.collect_gds and exc_type is None:
            try:
                self.gds_after = fetch_gds_io_window(
                    self.base_url,
                    "end",
                    self.gds_window_id,
                    self.flush_timeout,
                    headers=self.headers,
                )
                self.gds_result = calculate_gds_io_metrics(
                    self.gds_before, self.gds_after, self.gds_window_id
                )
                self.gds_begin_attempted = False
            except (requests.RequestException, ValueError, TimeoutError) as error:
                self.gds_result = unavailable_gds_io_metrics(error)
                self.gds_result["gds_io_metadata"] = {
                    "before": self.gds_before,
                    "after": self.gds_after,
                }
        self._abort_gds()
        if self.ssd_before is not None:
            try:
                self.ssd_after = fetch_mooncake_io_snapshot(
                    self.client_url,
                    "end",
                    self.ssd_before["window"]["id"],
                    self.ssd_before["pid"],
                )
                if exc_type is None:
                    if self.after is None:
                        raise ValueError("Completed TP I/O snapshots unavailable")
                    self.mooncake_result = calculate_mooncake_io_metrics(
                        self.before, self.after, self.ssd_before, self.ssd_after
                    )
            except (requests.RequestException, TimeoutError, ValueError) as error:
                self.mooncake_result = unavailable_mooncake_io_metrics(error)
                self.mooncake_result["mooncake_io_metadata"] = {
                    "ssd_before": self.ssd_before,
                    "ssd_after": self.ssd_after,
                    "tp_before": self.before,
                    "tp_after": self.after,
                }
                if exc_type is not None:
                    warnings.warn(
                        f"Could not close this benchmark's Mooncake I/O window: {error}"
                    )
        if self.collect_mooncake and exc_type is None:
            try:
                self.mooncake_result.update(
                    fetch_mooncake_storage_capacity(self.master_host, self.metrics_port)
                )
            except (requests.RequestException, ValueError) as error:
                self.mooncake_result.setdefault("mooncake_io_metadata", {})[
                    "capacity_error"
                ] = str(error)
        return False

    def print_metrics(self):
        if self.collect_host:
            print("HiCache L2 Host I/O Statistics".center(62, "-"))
            if self.host_result["hicache_io_status"] == "ok":
                print("Rank-pooled copy-service bandwidth, not aggregate TP bandwidth.")
                for direction in ("read", "write"):
                    value = self.host_result[f"dram_{direction}_bw_gbps"]
                    batches = self.host_result[f"hicache_io_{direction}_batches"]
                    byte_count = self.host_result[f"hicache_io_{direction}_bytes"]
                    print(
                        f"L2 Host DRAM {direction} bandwidth (GB/s): {value:.4f}"
                        + (" (no I/O)" if not batches else "")
                    )
                    print(
                        f"L2 Host {direction} rank-batches: {batches}; total (GB): {byte_count / 1e9:.4f}"
                    )
                value = self.host_result["mean_l2_kv_readback_ms"]
                print(
                    f"Mean all-layer H2D rank-batch time (ms): {value if value is not None else 'N/A (no readbacks)'}"
                )
            else:
                print(
                    f"Native HiCache I/O metrics unavailable: {self.host_result['hicache_io_error']}"
                )
        if self.collect_mooncake:
            print_mooncake_io_metrics(self.mooncake_result)
        if self.collect_gds:
            print_gds_io_metrics(self.gds_result)
