"""Pure cache-report projection, also loadable by file path without SGLang."""

import math
from copy import deepcopy

REPORT_SCHEMA_VERSION = 1


def _field(name, group, label, unit, aliases=()):
    return dict(name=name, group=group, label=label, unit=unit, aliases=aliases)


REPORT_FIELDS = (
    _field("total_prompt_tokens", "cache", "Total prompt tokens", "tokens"),
    _field("total_cached_tokens", "cache", "Total cached tokens", "tokens"),
    _field("cache_hit_rate", "cache", "Cache hit ratio", "ratio"),
    _field("cache_hit_rate_pct", "cache", "Cache hit rate", "%"),
    *(
        _field(
            f"total_cached_tokens_{tier}",
            "cache",
            f"{label} cached tokens",
            "tokens",
            (f"{tier}_cached_tokens",),
        )
        for tier, label in (
            ("device", "Device HBM"),
            ("host", "L2 Host DRAM"),
            ("storage", "L3 Storage backend"),
        )
    ),
    *(
        _field(f"{tier}_hit_rate_pct", "cache", f"{label} hit rate", "%")
        for tier, label in (
            ("device", "Device HBM"),
            ("host", "L2 Host DRAM"),
            ("storage", "L3 Storage backend"),
        )
    ),
    *(
        _field(
            f"flat_cached_tokens_{tier}",
            "cache",
            f"Legacy Flat {tier} tokens",
            "tokens",
        )
        for tier in ("dram", "ssd", "mixed", "total")
    ),
    *(
        _field(
            f"flat_{tier}_hit_rate_pct", "cache", f"Legacy Flat {tier} hit rate", "%"
        )
        for tier in ("dram", "ssd", "storage")
    ),
    _field(
        "offdevice_cached_tokens", "cache", "Host + Storage cached tokens", "tokens"
    ),
    _field("offdevice_hit_rate_pct", "cache", "Host + Storage hit rate", "%"),
    _field("flat_prefetch_latency_ms", "prefetch", "Overall mean prefetch", "ms"),
    _field(
        "flat_dram_prefetch_latency_ms", "prefetch", "DRAM-only mean prefetch", "ms"
    ),
    _field(
        "flat_ssd_prefetch_latency_ms", "prefetch", "SSD-dependent mean prefetch", "ms"
    ),
    _field("flat_io_window_seconds", "io", "I/O window duration", "s"),
    *(
        _field(
            f"flat_{medium}_{direction}_{kind}bw_gbps",
            "io",
            f"{medium.upper()} {'durable write' if medium == 'ssd' and direction == 'write' else direction} {label}",
            "GB/s",
        )
        for medium in ("dram", "ssd")
        for direction in ("read", "write")
        for kind, label in (
            ("", "average bandwidth"),
            ("peak_", "100ms peak bandwidth"),
        )
    ),
    _field("flat_ssd_read_io_count", "io", "SSD completed reads", "ops"),
    _field("flat_ssd_write_io_count", "io", "SSD completed writes", "ops"),
    _field(
        "flat_ssd_durable_write_batches", "io", "SSD durable write batches", "batches"
    ),
    _field("flat_ssd_new_kv_blocks", "io", "SSD new KV blocks", "blocks"),
    _field(
        "flat_capacity_pressure_offloads", "io", "Capacity-pressure offloads", "ops"
    ),
)
REPORT_FIELD_NAMES = tuple(
    name for field in REPORT_FIELDS for name in (field["name"], *field["aliases"])
)
_MODE_KEYS = (
    "metrics_mode",
    "metrics_mode_status",
    "metrics_mode_source",
    "metrics_mode_evidence",
)
REPORT_METADATA_FIELDS = (
    "cache_report_schema_version",
    *_MODE_KEYS,
    "cache_report_metadata",
    "cache_report",
    "flat_io_status",
    "flat_io_error",
)


def _number(value, *, integer=False):
    if type(value) is int:
        return value if value >= 0 else None
    if not integer and type(value) is float and math.isfinite(value) and value >= 0:
        return value
    return None


def _mode_name(value):
    if not isinstance(value, str):
        return None
    value = value.lower().replace("-", "_")
    if value in ("flat", "flat_memory", "flatcake"):
        return "flat"
    if value in (
        "native",
        "tiered",
        "hicache",
        "hbm",
        "hbm_only",
        "recompute",
        "disabled",
        "radix",
        "unified",
        "unified_radix",
        "none",
        "mooncake_tiered_gds",
    ):
        return "native"
    return None


def _runtime_evidence(snapshot, path):
    evidence = []
    if not isinstance(snapshot, dict):
        return evidence
    for name in ("cache_source_mode", "cache_mode"):
        if snapshot.get(name) is not None:
            evidence.append(
                dict(
                    source=f"{path}.{name}",
                    mode=_mode_name(snapshot[name]),
                    value=snapshot[name] if isinstance(snapshot[name], str) else None,
                )
            )
    for name in ("flat_memory_attached", "flat_memory_enabled"):
        if type(snapshot.get(name)) is bool:
            evidence.append(
                dict(
                    source=f"{path}.{name}",
                    mode="flat" if snapshot[name] else "native",
                    value=snapshot[name],
                )
            )
    if "flat_memory" in snapshot:
        flat = snapshot["flat_memory"]
        if flat is False:
            mode = "native"
        elif (
            isinstance(flat, dict)
            and flat.get("attached", flat.get("enabled")) is False
        ):
            mode = "native"
        elif isinstance(flat, dict) and (
            flat.get("attached") is True
            or flat.get("enabled") is True
            or flat.get("ranks")
        ):
            mode = "flat"
        else:
            mode = None
        if mode:
            evidence.append(dict(source=f"{path}.flat_memory", mode=mode, value=mode))
    for key in (
        "internal_states",
        "ranks",
        "flat_memory",
        "runtime",
        "runtime_state",
        "cached_tokens_details",
    ):
        value = snapshot.get(key)
        children = value if isinstance(value, list) else [value]
        for index, child in enumerate(children):
            evidence.extend(_runtime_evidence(child, f"{path}.{key}[{index}]"))
    return evidence


def _snapshot_evidence(snapshot, path):
    runtime = _runtime_evidence(snapshot, path)
    if runtime or not isinstance(snapshot, dict):
        return runtime
    evidence = []
    for key in ("server_config", "server_args", "memory", "cache"):
        if isinstance(snapshot.get(key), dict):
            evidence.extend(_snapshot_evidence(snapshot[key], f"{path}.{key}"))
    backend = snapshot.get("radix_cache_backend")
    storage = snapshot.get("hicache_storage_backend")
    if backend is not None:
        evidence.append(
            dict(
                source=f"{path}.radix_cache_backend",
                mode=_mode_name(backend),
                value=backend if isinstance(backend, str) else None,
            )
        )
    elif storage == "flat_memory":
        evidence.append(
            dict(source=f"{path}.hicache_storage_backend", mode="flat", value=storage)
        )
    elif type(snapshot.get("enable_hierarchical_cache")) is bool:
        evidence.append(
            dict(
                source=f"{path}.enable_hierarchical_cache",
                mode="native",
                value=snapshot["enable_hierarchical_cache"],
            )
        )
    elif type(snapshot.get("disable_radix_cache")) is bool:
        evidence.append(
            dict(
                source=f"{path}.disable_radix_cache",
                mode="native",
                value=snapshot["disable_radix_cache"],
            )
        )
    return evidence


def resolve_metrics_mode(*, before=None, after=None, details=()) -> dict:
    """Resolve architecture only from runtime, effective configuration and requests."""
    first = _snapshot_evidence(before, "before")
    last = _snapshot_evidence(after, "after")
    requests = []
    for index, detail in enumerate(details or ()):
        if isinstance(detail, dict):
            requests.extend(_runtime_evidence(detail, f"requests[{index}]"))
        elif isinstance(detail, str):
            requests.append(
                dict(
                    source=f"requests[{index}].cache_source_mode",
                    mode=_mode_name(detail),
                    value=detail,
                )
            )
    evidence = first + last + requests
    modes = {item["mode"] for item in evidence}
    status, mode = "unavailable", "unknown"
    if modes and None not in modes and len(modes) == 1:
        mode, status = next(iter(modes)), "resolved"
    elif evidence:
        status = "conflicting"
        first_modes = {item["mode"] for item in first}
        last_modes = {item["mode"] for item in last}
        if first_modes and last_modes and first_modes != last_modes:
            status = "mixed"
    return dict(
        metrics_mode=mode,
        metrics_mode_status=status,
        metrics_mode_source=", ".join(item["source"] for item in evidence) or "none",
        metrics_mode_evidence=evidence,
    )


def _result_mode(result):
    flat_meta = result.get("flat_io_metadata")
    flat_meta = flat_meta if isinstance(flat_meta, dict) else {}
    before = result.get("server_info_before", flat_meta.get("server_info_before"))
    after = result.get("server_info_after", flat_meta.get("server_info_after"))
    if before is None:
        before = result.get("server_info", result.get("server_config"))
    details = []
    for key in ("cached_tokens_details", "server_usage", "cache_source_modes"):
        value = result.get(key)
        if isinstance(value, (list, tuple)):
            details.extend(value)
        elif isinstance(value, dict):
            details.append(value)
    if "cache_source_mode" in result:
        details.append(dict(cache_source_mode=result["cache_source_mode"]))
    return resolve_metrics_mode(before=before, after=after, details=details)


def _usage_rows(result):
    arrays = {
        key: result[key]
        for key in (
            "server_usage",
            "server_prompt_tokens",
            "server_cached_tokens",
            "cached_tokens_details",
        )
        if isinstance(result.get(key), (list, tuple))
    }
    if not arrays:
        return None
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        return []
    count = lengths.pop()
    errors = result.get("errors")
    successes = result.get("successes")
    if errors is not None and (
        not isinstance(errors, (list, tuple)) or len(errors) != count
    ):
        return []
    if successes is not None and (
        not isinstance(successes, (list, tuple)) or len(successes) != count
    ):
        return []
    rows = []
    for index in range(count):
        if (errors is not None and errors[index]) or (
            successes is not None and not successes[index]
        ):
            continue
        usage = arrays["server_usage"][index] if "server_usage" in arrays else None
        row = dict(usage) if isinstance(usage, dict) else {}
        for source, target in (
            ("server_prompt_tokens", "prompt_tokens"),
            ("server_cached_tokens", "cached_tokens"),
            ("cached_tokens_details", "cached_tokens_details"),
        ):
            if source in arrays:
                row[target] = arrays[source][index]
        rows.append(row)
    if "completed" in result and result["completed"] != len(rows):
        return []
    return rows


def _sum(values, *, integer=True):
    values = list(values)
    if not values or any(_number(value, integer=integer) is None for value in values):
        return None
    return _number(sum(values), integer=integer)


def _row_cached(row):
    cached = _number(row.get("cached_tokens"), integer=True)
    prompt = _number(row.get("prompt_tokens"), integer=True)
    if cached is not None and prompt is not None and cached > prompt:
        return None
    return cached


def _row_tier(row, tier):
    details = row.get("cached_tokens_details")
    details = details if isinstance(details, dict) else {}
    raw = details.get(tier)
    value = _number(raw, integer=True)
    cached = _row_cached(row)
    if row.get("cached_tokens") is not None and cached is None:
        return None
    tiers = [
        _number(details.get(name), integer=True)
        for name in ("device", "host", "storage")
    ]
    known = sum(item for item in tiers if item is not None)
    if cached is not None and (
        known > cached or (all(item is not None for item in tiers) and known != cached)
    ):
        return None
    if cached == 0:
        return 0 if raw is None or value == 0 else None
    return value if value is not None and (cached is None or value <= cached) else None


def _cache_values(result):
    rows = _usage_rows(result)
    values = {}
    legacy = result.get("cache_report")
    legacy = legacy if isinstance(legacy, dict) else {}
    for name in ("total_prompt_tokens", "total_cached_tokens"):
        wire = "prompt_tokens" if name == "total_prompt_tokens" else "cached_tokens"
        values[name] = (
            _sum(
                _row_cached(row) if wire == "cached_tokens" else row.get(wire)
                for row in rows
            )
            if rows is not None
            else _number(result.get(name, legacy.get(name)), integer=True)
        )
    for tier in ("device", "host", "storage"):
        name = f"total_cached_tokens_{tier}"
        value = result.get(
            name,
            result.get(f"{tier}_cached_tokens", legacy.get(f"{tier}_cached_tokens")),
        )
        values[name] = (
            _sum(_row_tier(row, tier) for row in rows)
            if rows is not None
            else _number(value, integer=True)
        )
        if rows is None and values["total_cached_tokens"] == 0 and value is None:
            values[name] = 0
    prompt, cached = values["total_prompt_tokens"], values["total_cached_tokens"]
    if cached is not None and prompt is not None and cached > prompt:
        values["total_cached_tokens"] = cached = None
    tiers = [
        values[f"total_cached_tokens_{tier}"] for tier in ("device", "host", "storage")
    ]
    inconsistent = (
        cached is not None
        and all(value is not None for value in tiers)
        and sum(tiers) != cached
    )
    for tier in ("device", "host", "storage"):
        name = f"total_cached_tokens_{tier}"
        value = values[name]
        if inconsistent or (
            value is not None and cached is not None and value > cached
        ):
            values[name] = None
        values[f"{tier}_hit_rate_pct"] = _rate(values[name], prompt, scale=100)
    values["cache_hit_rate"] = _rate(cached, prompt)
    values["cache_hit_rate_pct"] = _rate(cached, prompt, scale=100)
    values["offdevice_cached_tokens"] = _sum(
        values[f"total_cached_tokens_{tier}"] for tier in ("host", "storage")
    )
    values["offdevice_hit_rate_pct"] = _rate(
        values["offdevice_cached_tokens"], prompt, scale=100
    )
    return values


def _rate(tokens, prompt, *, scale=1):
    if tokens is None or prompt is None or not prompt or tokens > prompt:
        return None
    return scale * tokens / prompt


def _prefetch_values(result):
    rows = _usage_rows(result)
    values, reasons = {}, {}
    pairs = []
    for medium in ("dram", "ssd"):
        if rows is None:
            elapsed = _number(result.get(f"flat_prefetch_{medium}_ms"))
            ops = _number(result.get(f"flat_prefetch_{medium}_ops"), integer=True)
        else:
            details = [
                (
                    row["cached_tokens_details"]
                    if isinstance(row.get("cached_tokens_details"), dict)
                    else {}
                )
                for row in rows
            ]
            elapsed = _sum(
                (detail.get(f"flat_prefetch_{medium}_ms") for detail in details),
                integer=False,
            )
            ops = _sum(detail.get(f"flat_prefetch_{medium}_ops") for detail in details)
        if ops == 0 and elapsed not in (0, None):
            elapsed = ops = None
        pairs.append((elapsed, ops))
        name = f"flat_{medium}_prefetch_latency_ms"
        values[name] = elapsed / ops if elapsed is not None and ops else None
        reasons[name] = (
            "no_samples" if ops == 0 else "missing_or_invalid_prefetch_counters"
        )
    name = "flat_prefetch_latency_ms"
    if all(value is not None for pair in pairs for value in pair):
        elapsed, ops = (sum(pair[index] for pair in pairs) for index in (0, 1))
        values[name] = elapsed / ops if ops else None
        reasons[name] = (
            "no_samples" if ops == 0 else "missing_or_invalid_prefetch_counters"
        )
    else:
        values[name], reasons[name] = None, "missing_or_invalid_prefetch_counters"
    return values, reasons


_WINDOW_COUNTERS = (
    "read_completed_ops",
    "read_completed_bytes",
    "write_completed_ops",
    "write_completed_bytes",
    "durable_write_batches",
    "durable_write_bytes",
    "io_errors",
)
_DRAM_COUNTERS = ("dram_read_completed_bytes", "dram_write_completed_bytes")


def _counter(value):
    if _number(value, integer=True) is None:
        raise ValueError(f"Missing or invalid Flat counter: {value!r}")
    return value


def _flat_rank_map(snapshot, expected_tp_size=None):
    if not isinstance(snapshot, dict):
        raise ValueError("Missing Flat snapshot")
    tp_size = snapshot.get("tp_size", expected_tp_size)
    if type(tp_size) is not int or tp_size not in (1, 4, 8):
        raise ValueError("Flat snapshot requires explicit TP1/4/8")
    if expected_tp_size is not None and tp_size != expected_tp_size:
        raise ValueError("Flat TP size changed")
    ranks = snapshot.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != tp_size:
        raise ValueError("Flat snapshot requires every TP rank")
    mapped = {}
    for rank in ranks:
        if not isinstance(rank, dict) or rank.get("error"):
            raise ValueError("Invalid Flat rank snapshot")
        index = _counter(rank.get("tp_rank"))
        if index in mapped or not _counter(rank.get("pid")):
            raise ValueError("Duplicate TP rank or invalid process identity")
        _counter(rank.get("gpu_id"))
        if rank.get("tp_size", tp_size) != tp_size:
            raise ValueError("Rank TP topology mismatch")
        mapped[index] = rank
    if (
        set(mapped) != set(range(tp_size))
        or len({rank["pid"] for rank in ranks}) != tp_size
    ):
        raise ValueError("Incomplete or duplicate Flat TP identities")
    return mapped


def _validate_flat_window(snapshot, window_id, active, expected_tp_size=None):
    ranks = _flat_rank_map(snapshot, expected_tp_size)
    boundaries, versions = set(), set()
    for rank in ranks.values():
        window = rank.get("io_window")
        if not isinstance(window, dict):
            raise ValueError("Missing Flat I/O window")
        version = window.get("schema_version", 1)
        if type(version) is not int or version not in (1, 2):
            raise ValueError("Unsupported Flat window schema")
        versions.add(version)
        if (
            window.get("enabled") is not True
            or window.get("window_id") != window_id
            or window.get("active") is not active
            or window.get("aborted") is not False
            or window.get("overflowed") is not False
            or _counter(window.get("bucket_ns")) != 100_000_000
        ):
            raise ValueError("Disabled, mismatched, aborted or overflowed Flat window")
        start, end = _counter(window.get("start_ns")), _counter(window.get("end_ns"))
        if not start or (active and end != 0) or (not active and end <= start):
            raise ValueError("Invalid Flat window boundaries")
        boundaries.add((start, end))
        for name in _WINDOW_COUNTERS + (_DRAM_COUNTERS if version == 2 else ()):
            value = _counter(window.get(name))
            if (active or name == "io_errors") and value:
                raise ValueError("Nonempty begin window or Flat I/O errors")
        if not isinstance(window.get("buckets"), list) or (
            active and window["buckets"]
        ):
            raise ValueError("Invalid Flat completion buckets")
        for name in ("pending_backups", "pending_prefetches"):
            if _counter(rank.get(name)):
                raise ValueError("Flat pipeline was not drained")
        _counter(rank.get("flat_io_errors"))
    if len(boundaries) != 1 or len(versions) != 1:
        raise ValueError("Flat TP window boundaries or schemas differ")
    return ranks, next(iter(boundaries)), next(iter(versions))


def _flat_bucket_peaks(ranks, *, duration_ns, schema_version):
    fields = dict(read_bytes="read_completed_bytes", write_bytes="durable_write_bytes")
    if schema_version == 2:
        fields.update(
            dram_read_bytes="dram_read_completed_bytes",
            dram_write_bytes="dram_write_completed_bytes",
        )
    buckets = {}
    for rank in ranks.values():
        window = rank["io_window"]
        totals, seen = dict.fromkeys(fields, 0), set()
        for bucket in window["buckets"]:
            if not isinstance(bucket, dict):
                raise ValueError("Invalid Flat completion bucket")
            index = _counter(bucket.get("index"))
            if index in seen or index * 100_000_000 >= duration_ns:
                raise ValueError("Duplicate or out-of-window Flat bucket")
            seen.add(index)
            merged = buckets.setdefault(index, dict.fromkeys(fields, 0))
            for name in fields:
                value = _counter(bucket.get(name))
                totals[name] += value
                merged[name] += value
        if any(totals[name] != window[total] for name, total in fields.items()):
            raise ValueError("Flat bucket totals do not match completed/durable bytes")
    # TP peaks use aligned completion buckets, not the sum of per-rank peaks.
    return {
        name: max((bucket[name] for bucket in buckets.values()), default=0) / 0.1 / 1e9
        for name in fields
    }


def _flat_optional_delta(first, last, name, *, bandwidth=False, replicated=False):
    values = []
    for index, rank in last.items():
        prior = first[index]
        if bandwidth:
            prior, rank = prior.get("bandwidth") or {}, rank.get("bandwidth") or {}
        if name not in prior or name not in rank:
            return None
        values.append(_counter(_counter(rank[name]) - _counter(prior[name])))
    if replicated and len(set(values)) != 1:
        raise ValueError(f"Flat ranks disagree on {name}")
    return values[0] if replicated else sum(values)


def _validate_flat_counts(window):
    for bytes_name, ops_name in (
        ("read_completed_bytes", "read_completed_ops"),
        ("write_completed_bytes", "write_completed_ops"),
        ("durable_write_bytes", "durable_write_batches"),
    ):
        if bool(window[bytes_name]) != bool(window[ops_name]):
            raise ValueError("Flat bytes and operation counts disagree")
    if window["durable_write_bytes"] > window["write_completed_bytes"]:
        raise ValueError("Durable bytes exceed completed SSD writes")


def _summarize_flat_io(before, after, window_id, expected_tp_size=None):
    first, (start, _), version = _validate_flat_window(
        before, window_id, True, expected_tp_size
    )
    last, (end_start, end), end_version = _validate_flat_window(
        after, window_id, False, len(first)
    )
    if start != end_start or version != end_version:
        raise ValueError("Flat begin/end boundary or schema mismatch")
    totals = dict.fromkeys(
        _WINDOW_COUNTERS + (_DRAM_COUNTERS if version == 2 else ()), 0
    )
    for index, rank in last.items():
        prior = first[index]
        if any(
            name not in rank or name not in prior or rank[name] != prior[name]
            for name in ("pid", "gpu_id", "gds_mode")
        ):
            raise ValueError("Flat process or device identity changed")
        if _counter(rank["flat_io_errors"] - prior["flat_io_errors"]):
            raise ValueError("Flat pipeline reported I/O errors")
        _validate_flat_counts(rank["io_window"])
        for name in totals:
            totals[name] += rank["io_window"][name]
    seconds = (end - start) / 1e9
    peaks = _flat_bucket_peaks(last, duration_ns=end - start, schema_version=version)
    result = dict(flat_io_window_seconds=seconds)
    for medium, prefix, bucket in (("dram", "dram_", "dram_"), ("ssd", "", "")):
        for direction in ("read", "write"):
            name = f"{prefix}{direction}_completed_bytes"
            if medium == "ssd" and direction == "write":
                name = "durable_write_bytes"
            result[f"flat_{medium}_{direction}_bw_gbps"] = (
                totals[name] / seconds / 1e9 if name in totals else None
            )
            result[f"flat_{medium}_{direction}_peak_bw_gbps"] = peaks.get(
                f"{bucket}{direction}_bytes"
            )
    result.update(
        flat_ssd_read_io_count=totals["read_completed_ops"],
        flat_ssd_write_io_count=totals["write_completed_ops"],
        flat_ssd_durable_write_batches=totals["durable_write_batches"],
        # The manager counts fresh inserted blocks here, not SSD CQEs or batches.
        flat_ssd_new_kv_blocks=_flat_optional_delta(
            first, last, "ssd_write_count", bandwidth=True
        ),
        flat_capacity_pressure_offloads=_flat_optional_delta(
            first, last, "flat_capacity_pressure_offloads", replicated=True
        ),
    )
    return result


def _io_values(result):
    metadata = result.get("flat_io_metadata")
    if not isinstance(metadata, dict) or result.get("flat_io_error"):
        return {}, result.get("flat_io_error") or "missing_window_evidence"
    try:
        before, after = metadata.get("before"), metadata.get("after")
        ranks = _flat_rank_map(before)
        window_id = ranks[0]["io_window"]["window_id"]
        return _summarize_flat_io(before, after, window_id), None
    except (ValueError, KeyError, TypeError) as error:
        return {}, str(error)


def _metadata(*, status, reason, source, unit):
    return dict(status=status, reason=reason, source=source, unit=unit)


def _versioned_projection(result):
    report = {name: deepcopy(result.get(name)) for name in REPORT_FIELD_NAMES}
    mode = {key: deepcopy(result.get(key)) for key in _MODE_KEYS}
    mode["metrics_mode"] = mode["metrics_mode"] or "unknown"
    mode["metrics_mode_status"] = mode["metrics_mode_status"] or (
        "resolved" if mode["metrics_mode"] in ("native", "flat") else "unavailable"
    )
    mode["metrics_mode_source"] = mode["metrics_mode_source"] or "versioned_report"
    mode["metrics_mode_evidence"] = mode["metrics_mode_evidence"] or []
    version = result["cache_report_schema_version"]
    if type(version) is not int or version != REPORT_SCHEMA_VERSION:
        mode.update(
            metrics_mode="unknown",
            metrics_mode_status="unavailable",
            metrics_mode_source="unsupported_report_schema",
        )
    elif mode["metrics_mode"] not in ("native", "flat", "unknown") or mode[
        "metrics_mode_status"
    ] not in ("resolved", "unavailable", "conflicting", "mixed"):
        mode.update(
            metrics_mode="unknown",
            metrics_mode_status="unavailable",
            metrics_mode_source="unsupported_report_mode",
        )
    metadata = result.get("cache_report_metadata") or {}
    projected_meta = {}
    for field in REPORT_FIELDS:
        name = field["name"]
        report[name] = _number(
            report[name],
            integer=field["unit"] in ("tokens", "ops", "batches", "blocks"),
        )
        selected = (
            mode["metrics_mode"]
            if mode["metrics_mode_status"] == "resolved"
            else "unknown"
        )
        inactive = selected in ("native", "flat") and selected != (
            "native" if field["group"] == "cache" else "flat"
        )
        legacy_flat = field["group"] == "cache" and name.startswith("flat_")
        gated = inactive or selected == "unknown" or legacy_flat
        if gated:
            report[name] = None
        projected_meta[name] = (
            deepcopy(metadata.get(name)) if not gated else None
        ) or _metadata(
            status=(
                "not_applicable"
                if inactive
                else ("measured" if report[name] is not None else "unavailable")
            ),
            reason=(
                f"not_applicable_in_{selected}_mode"
                if inactive
                else (None if report[name] is not None else "authoritative_null")
            ),
            source="versioned_report",
            unit=field["unit"],
        )
        for alias in field["aliases"]:
            report[alias] = report[name]
            projected_meta[alias] = deepcopy(projected_meta[name])
    report.update(
        mode,
        cache_report_schema_version=result["cache_report_schema_version"],
        cache_report_metadata=projected_meta,
    )
    report["flat_io_status"] = (
        result.get("flat_io_status", "unavailable")
        if selected == "flat"
        else ("not_applicable" if selected == "native" else "unavailable")
    )
    report["flat_io_error"] = (
        deepcopy(result.get("flat_io_error")) if selected == "flat" else None
    )
    report["cache_report"] = {
        name: report[name]
        for field in REPORT_FIELDS
        if field["group"] == "cache"
        for name in (field["name"], *field["aliases"])
    }
    return report


def project_cache_reports(result: dict, *, mode: dict | None = None) -> dict:
    """Return a full public projection without changing raw inputs or prior nulls."""
    if result.get("cache_report_schema_version") is not None:
        return _versioned_projection(result)
    mode = deepcopy(mode) if mode is not None else _result_mode(result)
    resolved = mode.get("metrics_mode_status") == "resolved"
    selected = mode.get("metrics_mode") if resolved else "unknown"
    cache = _cache_values(result) if selected == "native" else {}
    prefetch, prefetch_reasons = (
        _prefetch_values(result) if selected == "flat" else ({}, {})
    )
    report, metadata = {}, {}
    io, io_reason = _io_values(result) if selected == "flat" else ({}, None)
    for field in REPORT_FIELDS:
        name, group = field["name"], field["group"]
        applicable = selected == ("native" if group == "cache" else "flat")
        value, status, reason, source = None, "unavailable", "mode_unresolved", "none"
        if selected != "unknown" and not applicable:
            status, reason = "not_applicable", f"not_applicable_in_{selected}_mode"
        elif applicable:
            if group == "cache":
                value, source, reason = (
                    cache.get(name),
                    "server_usage",
                    "missing_or_invalid_server_usage",
                )
                if name.startswith("flat_"):
                    reason = "physical_source_semantics_unavailable"
            elif group == "prefetch":
                value, source, reason = (
                    prefetch.get(name),
                    "cached_tokens_details",
                    prefetch_reasons.get(name),
                )
            else:
                value = io.get(name)
                source, reason = (
                    "flat_io_window",
                    io_reason or "unsupported_window_metric",
                )
            if value is not None:
                status, reason = "measured", None
        report[name] = value
        metadata[name] = _metadata(
            status=status, reason=reason, source=source, unit=field["unit"]
        )
        for alias in field["aliases"]:
            report[alias], metadata[alias] = value, deepcopy(metadata[name])
    report.update({key: deepcopy(mode.get(key)) for key in _MODE_KEYS})
    report.update(
        cache_report_schema_version=REPORT_SCHEMA_VERSION,
        cache_report_metadata=metadata,
    )
    report["flat_io_status"] = (
        "not_applicable" if selected == "native" else ("ok" if io else "unavailable")
    )
    report["flat_io_error"] = (
        io_reason
        if selected == "flat"
        else (None if selected == "native" else "mode_unresolved")
    )
    report["cache_report"] = {
        name: report[name]
        for field in REPORT_FIELDS
        if field["group"] == "cache"
        for name in (field["name"], *field["aliases"])
    }
    return report


def format_cache_reports(report: dict) -> str:
    """Render all three blocks; absent numbers are blank, never synthetic zero."""
    report = project_cache_reports(report)
    lines = [
        f"Metrics mode: {report['metrics_mode']} ({report['metrics_mode_status']})"
    ]
    for group, title in (
        ("cache", "Cache Hit Statistics"),
        ("prefetch", "Flat Prefetch Arrival-to-All-TP-GPU-Ready"),
        ("io", "Flat Completed/Durable I/O Window"),
    ):
        lines.append(f"\n{title}")
        if group == "io":
            lines.append(f"  Flat I/O status: {report['flat_io_status']}")
            if report["flat_io_error"]:
                lines.append(f"  Flat I/O reason: {report['flat_io_error']}")
        for field in REPORT_FIELDS:
            if (
                field["group"] != group
                or field["name"].startswith("flat_cached_")
                or field["name"]
                in (
                    "flat_dram_hit_rate_pct",
                    "flat_ssd_hit_rate_pct",
                    "flat_storage_hit_rate_pct",
                )
            ):
                continue
            value = report[field["name"]]
            text = (
                ""
                if value is None
                else (f"{value:,}" if type(value) is int else f"{value:.6g}")
            )
            lines.append(f"  {field['label']} ({field['unit']}): {text}")
    return "\n".join(lines)


def format_flat_memory_report(result: dict, *, mode: dict | None = None) -> str:
    return format_cache_reports(project_cache_reports(result, mode=mode))
