"""FLAT_MEMORY: Pure report projection; loadable by path without SGLang."""

import math
from collections.abc import Mapping
from copy import deepcopy

REPORT_SCHEMA_VERSION = 1


def _field(name, group, label, unit, applicability="flat", aliases=()):
    return dict(
        name=name, group=group, label=label, unit=unit,
        applicability=applicability, aliases=aliases,
    )


REPORT_FIELDS = (
    _field("total_prompt_tokens", "cache", "Total prompt tokens", "tokens", "common"),
    _field("total_cached_tokens", "cache", "Total cached tokens", "tokens", "common"),
    _field("cache_hit_rate", "cache", "Cache hit ratio", "ratio", "common"),
    _field("cache_hit_rate_pct", "cache", "Cache hit rate", "%", "common"),
    *(
        _field(
            f"total_cached_tokens_{tier}", "cache", f"{label} cached tokens",
            "tokens", applicability, (f"{tier}_cached_tokens",),
        )
        for tier, label, applicability in (
            ("device", "Device HBM", "common"),
            ("host", "L2 Host DRAM", "native"),
            ("storage", "L3 Storage backend", "native"),
        )
    ),
    *(
        _field(f"{tier}_hit_rate_pct", "cache", f"{label} hit rate", "%", applicability)
        for tier, label, applicability in (
            ("device", "Device HBM", "common"),
            ("host", "L2 Host DRAM", "native"),
            ("storage", "L3 Storage backend", "native"),
        )
    ),
    _field("offdevice_cached_tokens", "cache", "Host + Storage cached tokens", "tokens", "native"),
    _field("offdevice_hit_rate_pct", "cache", "Host + Storage hit rate", "%", "native"),
    *(
        _field(f"flat_cached_tokens_{tier}", "cache", f"Flat {label} cached tokens", "tokens")
        for tier, label in (("dram", "DRAM"), ("ssd", "SSD"), ("mixed", "mixed (SSD subset)"), ("total", "total"))
    ),
    *(
        _field(f"flat_{tier}_hit_rate_pct", "cache", f"Flat {label} hit rate", "%")
        for tier, label in (("dram", "DRAM"), ("ssd", "SSD"), ("storage", "total"))
    ),
    _field("flat_prefetch_latency_ms", "prefetch", "Overall mean prefetch", "ms"),
    _field("flat_dram_prefetch_latency_ms", "prefetch", "DRAM-only mean prefetch", "ms"),
    _field("flat_ssd_prefetch_latency_ms", "prefetch", "SSD-dependent mean prefetch", "ms"),
    _field("flat_io_window_seconds", "io", "I/O window duration", "s"),
    *(
        _field(
            f"flat_{medium}_{direction}_{kind}bw_gbps", "io",
            f"{medium.upper()} {'durable write' if medium == 'ssd' and direction == 'write' else direction} {label}",
            "GB/s",
        )
        for medium in ("dram", "ssd")
        for direction in ("read", "write")
        for kind, label in (("", "average bandwidth"), ("peak_", "100ms peak bandwidth"))
    ),
    _field("flat_ssd_read_io_count", "io", "SSD completed reads", "ops"),
    _field("flat_ssd_write_io_count", "io", "SSD completed writes", "ops"),
    _field("flat_ssd_durable_write_batches", "io", "SSD durable write batches", "batches"),
    _field("flat_ssd_new_kv_blocks", "io", "SSD new KV blocks", "blocks"),
    _field("flat_capacity_pressure_offloads", "io", "Capacity-pressure offloads", "ops"),
)
REPORT_FIELD_NAMES = tuple(sorted(
    name for field in REPORT_FIELDS for name in (field["name"], *field["aliases"])
))
_MODE_KEYS = (
    "metrics_mode", "metrics_mode_status", "metrics_mode_source", "metrics_mode_evidence",
)
REPORT_METADATA_FIELDS = (
    "cache_report_schema_version", *_MODE_KEYS, "cache_report_metadata",
    "cache_report", "flat_io_status", "flat_io_error",
)
_RAW_FIELDS = (
    "server_prompt_tokens", "server_cached_tokens", "cached_tokens_details",
    "server_usage", "errors", "successes", "cache_source_modes", "cache_source_mode",
    "server_info_before", "server_info_after", "server_info", "server_config",
    "flat_io_metadata", "flat_io", "cache_report_raw",
    "cache_metadata_status", "cache_metadata_errors",
)


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


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
        "native", "tiered", "hicache", "hbm", "hbm_only", "recompute", "disabled",
        "radix", "unified", "unified_radix", "none", "mooncake", "mooncake_tiered_gds",
    ):
        return "native"
    return None


def _evidence(snapshot, path, *, request=False):
    if not isinstance(snapshot, Mapping):
        return []
    evidence = []

    def add(name, mode, value):
        evidence.append(dict(source=f"{path}.{name}", mode=mode, value=value))

    for name in ("cache_source_mode", "cache_mode", "radix_cache_backend"):
        if snapshot.get(name) is not None:
            add(name, _mode_name(snapshot[name]), snapshot[name])
    for name in ("flat_memory_attached", "flat_memory_enabled"):
        if type(snapshot.get(name)) is bool:
            add(name, "flat" if snapshot[name] else "native", snapshot[name])
    flat = _mapping(snapshot.get("flat_memory"))
    for name in ("attached", "enabled"):
        if type(flat.get(name)) is bool:
            add(f"flat_memory.{name}", "flat" if flat[name] else "native", flat[name])
    storage = snapshot.get("hicache_storage_backend")
    if storage == "flat_memory" or (storage == "mooncake" and not evidence):
        add("hicache_storage_backend", _mode_name(storage), storage)
    if not evidence:
        for name in ("enable_hicache", "enable_hierarchical_cache", "disable_radix_cache"):
            if type(snapshot.get(name)) is bool:
                add(name, "native", snapshot[name])
                break
    # FLAT_MEMORY: Request schema presence is evidence even when every hit is zero.
    if request and all(
        _number(snapshot.get(name), integer=True) is not None
        for name in ("flat_dram", "flat_ssd", "flat_mixed")
    ):
        add("flat_source_schema", "flat", "flat_dram/flat_ssd/flat_mixed")
    for name in (
        "server_args", "server_config", "internal_states", "runtime", "runtime_state",
        "memory", "cache", "cached_tokens_details",
    ):
        children = snapshot.get(name)
        children = children if isinstance(children, (list, tuple)) else [children]
        for index, child in enumerate(children):
            evidence.extend(_evidence(child, f"{path}.{name}[{index}]", request=request))
    return evidence


def resolve_metrics_mode(*, before=None, after=None, details=()) -> dict:
    """Resolve mode from runtime configuration and request provenance, not counters."""
    evidence = _evidence(before, "before") + _evidence(after, "after")
    details = [details] if isinstance(details, (Mapping, str)) else details or ()
    for index, detail in enumerate(details):
        if isinstance(detail, str):
            detail = {"cache_source_mode": detail}
        evidence.extend(_evidence(detail, f"requests[{index}]", request=True))
    modes = {item["mode"] for item in evidence}
    resolved = len(modes) == 1 and None not in modes
    return dict(
        metrics_mode=next(iter(modes)) if resolved else "unknown",
        metrics_mode_status="resolved" if resolved else ("conflicting" if len(modes) > 1 else "unavailable"),
        metrics_mode_source=", ".join(item["source"] for item in evidence) or "none",
        metrics_mode_evidence=evidence,
    )


def _result_mode(result, mode, versioned):
    if mode is None and (versioned or "metrics_mode" in result):
        mode = {key: result[key] for key in _MODE_KEYS if key in result}
    if mode is not None:
        if isinstance(mode, str):
            mode = dict(metrics_mode=_mode_name(mode) or "unknown", metrics_mode_source="explicit")
        mode = deepcopy(dict(_mapping(mode)))
        selected = _mode_name(mode.get("metrics_mode")) or "unknown"
        status = mode.get("metrics_mode_status", "unavailable" if selected == "unknown" else "resolved")
        if status not in ("resolved", "unavailable", "conflicting", "mixed") or selected == "unknown":
            status = status if status in ("conflicting", "mixed") else "unavailable"
        return dict(
            metrics_mode=selected if status == "resolved" else "unknown",
            metrics_mode_status=status,
            metrics_mode_source=mode.get("metrics_mode_source", "report"),
            metrics_mode_evidence=mode.get("metrics_mode_evidence", []),
        )
    metadata = _mapping(result.get("flat_io_metadata"))
    details = []
    for name in ("cached_tokens_details", "server_usage", "cache_source_modes"):
        value = result.get(name)
        details.extend(value if isinstance(value, (list, tuple)) else [value])
    if "cache_source_mode" in result:
        details.append({"cache_source_mode": result["cache_source_mode"]})
    return resolve_metrics_mode(
        before=result.get("server_info_before", metadata.get("server_info_before", result.get("server_info", result.get("server_config")))),
        after=result.get("server_info_after", metadata.get("server_info_after")),
        details=details,
    )


def _usage_rows(result):
    names = ("server_usage", "server_prompt_tokens", "server_cached_tokens", "cached_tokens_details")
    arrays = {name: result[name] for name in names if name in result}
    if not arrays or any(not isinstance(value, (list, tuple)) for value in arrays.values()):
        return []
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        return []
    count = lengths.pop()
    for name in ("errors", "successes"):
        if name in result and (not isinstance(result[name], (list, tuple)) or len(result[name]) != count):
            return []
    rows = []
    errors, successes = result.get("errors"), result.get("successes")
    for index in range(count):
        if (errors is not None and errors[index]) or (
            successes is not None and successes[index] is not True
        ):
            continue
        usage = arrays["server_usage"][index] if "server_usage" in arrays else None
        row = dict(_mapping(usage))
        for source, target in zip(
            names[1:], ("prompt_tokens", "cached_tokens", "cached_tokens_details")
        ):
            if source in arrays:
                row[target] = arrays[source][index]
        rows.append(row)
    return rows if result.get("completed", len(rows)) == len(rows) else []


def _sum(values, *, integer=True):
    values = [_number(value, integer=integer) for value in values]
    return sum(values) if values and None not in values else None


def _legacy_cache(result):
    rows = _usage_rows(result)
    counters = []
    for row in rows:
        prompt = _number(row.get("prompt_tokens"), integer=True)
        cached = _number(row.get("cached_tokens"), integer=True)
        details = _mapping(row.get("cached_tokens_details"))
        tiers = {tier: _number(details.get(tier), integer=True) for tier in ("device", "host", "storage")}
        if cached is not None and prompt is not None and cached > prompt:
            cached = None
        known = [value for value in tiers.values() if value is not None]
        if cached is None or sum(known) > cached or (len(known) == 3 and sum(known) != cached):
            tiers = dict.fromkeys(tiers)
        flat = [_number(details.get(f"flat_{tier}"), integer=True) for tier in ("dram", "ssd", "mixed")]
        dram, ssd, mixed = flat
        total = dram + ssd if dram is not None and ssd is not None else None
        if total != tiers["storage"] or mixed is None or ssd is None or mixed > ssd:
            flat, total = [None] * 3, None
        counters.append({
            "total_prompt_tokens": prompt, "total_cached_tokens": cached,
            **{f"total_cached_tokens_{tier}": value for tier, value in tiers.items()},
            **dict(zip((f"flat_cached_tokens_{tier}" for tier in ("dram", "ssd", "mixed", "total")), [*flat, total])),
        })
    return {name: _sum(row[name] for row in counters) for name in counters[0]} if counters else {}


def _prefetch(result, rows):
    values, reasons, pairs = {}, {}, []
    details = [_mapping(row.get("cached_tokens_details")) for row in rows]
    for medium in ("dram", "ssd"):
        elapsed_key, ops_key = f"flat_prefetch_{medium}_ms", f"flat_prefetch_{medium}_ops"
        elapsed = result.get(elapsed_key, _sum((row.get(elapsed_key) for row in details), integer=False))
        ops = result.get(ops_key, _sum(row.get(ops_key) for row in details))
        elapsed, ops = _number(elapsed), _number(ops, integer=True)
        pairs.append((elapsed, ops))
        name = f"flat_{medium}_prefetch_latency_ms"
        values[name] = elapsed / ops if elapsed is not None and ops else None
        reasons[name] = "no_samples" if ops == 0 else "missing_or_invalid_prefetch_counters"
    elapsed, ops = (_sum((pair[index] for pair in pairs), integer=index == 1) for index in (0, 1))
    values["flat_prefetch_latency_ms"] = elapsed / ops if elapsed is not None and ops else None
    reasons["flat_prefetch_latency_ms"] = "no_samples" if ops == 0 else "missing_or_invalid_prefetch_counters"
    return values, reasons


def _input_values(result):
    cache = _mapping(result.get("cache_report"))
    io = _mapping(result.get("flat_io"))
    io = _mapping(io.get("result", io))
    values = {}
    for field in REPORT_FIELDS:
        for name in (field["name"], *field["aliases"]):
            sources = (result, cache, io) if field["group"] == "io" else (result, cache)
            found = next((source for source in sources if name in source), None)
            if found is not None:
                values[field["name"]] = found[name]
                break
    return values, io


def _derived_cache(values):
    values = dict(values)
    values.setdefault("offdevice_cached_tokens", _sum(values.get(f"total_cached_tokens_{tier}") for tier in ("host", "storage")))
    rates = {
        "cache_hit_rate": "total_cached_tokens", "cache_hit_rate_pct": "total_cached_tokens",
        **{f"{tier}_hit_rate_pct": f"total_cached_tokens_{tier}" for tier in ("device", "host", "storage")},
        "offdevice_hit_rate_pct": "offdevice_cached_tokens",
        **{f"flat_{tier}_hit_rate_pct": f"flat_cached_tokens_{tier}" for tier in ("dram", "ssd")},
        "flat_storage_hit_rate_pct": "flat_cached_tokens_total",
    }
    prompt = _number(values.get("total_prompt_tokens"), integer=True)
    for name, count in rates.items():
        tokens = _number(values.get(count), integer=True)
        scale = 1 if name == "cache_hit_rate" else 100
        values.setdefault(name, scale * tokens / prompt if prompt and tokens is not None and tokens <= prompt else None)
    return values


def _project_field(field, *, values, metadata, selected, versioned, cache_error, io_error, reasons):
    name = field["name"]
    prior = deepcopy(dict(_mapping(metadata.get(name))))
    value = _number(values.get(name), integer=field["unit"] in ("tokens", "ops", "batches", "blocks"))
    status, reason = "measured", None
    if selected == "unknown":
        status, reason = "unavailable", "mode_unresolved"
    elif field["applicability"] not in ("common", selected):
        status, reason = "not_applicable", f"not_applicable_in_{selected}_mode"
    elif field["group"] == "io" and io_error:
        status, reason = "unavailable", io_error
    elif field["group"] in ("cache", "prefetch") and cache_error:
        status, reason = "unavailable", cache_error
    elif prior.get("status") in ("unavailable", "not_applicable"):
        status, reason = "unavailable", prior.get("reason") or "authoritative_null"
    elif field["unit"] in ("%", "ratio") and not _number(values.get("total_prompt_tokens"), integer=True):
        status, reason = "unavailable", "missing_or_zero_server_prompt_tokens"
    elif field["unit"] in ("%", "ratio") and value is not None and value > (100 if field["unit"] == "%" else 1):
        status, reason = "unavailable", "invalid_hit_rate"
    elif value is None:
        status, reason = "unavailable", prior.get("reason") or reasons.get(name) or ("authoritative_null" if name in values else "missing_or_invalid_telemetry")
    if status != "measured":
        value = None
    prior.update(status=status, reason=reason, source=prior.get("source", "versioned_report" if versioned else "validated_aggregate"), unit=field["unit"])
    return value, prior


def project_cache_reports(result: Mapping, *, mode: Mapping | str | None = None) -> dict:
    """Project validated aggregates; only unversioned raw-only cache data is reduced.

    Canonical nulls are authoritative. I/O values require an ok collector result;
    this module never recomputes bandwidth from snapshots or imports collectors.
    """
    versioned = "cache_report_schema_version" in result
    version = result.get("cache_report_schema_version", REPORT_SCHEMA_VERSION)
    mode = _result_mode(result, mode, versioned)
    if type(version) is not int or version != REPORT_SCHEMA_VERSION:
        mode.update(metrics_mode="unknown", metrics_mode_status="unavailable", metrics_mode_source="unsupported_report_schema")
    selected = mode["metrics_mode"] if mode["metrics_mode_status"] == "resolved" else "unknown"
    values, io = _input_values(result)
    reasons = {}
    if not versioned:
        if not any(field["name"] in values for field in REPORT_FIELDS if field["group"] == "cache"):
            values.update(_legacy_cache(result))
        values = _derived_cache(values)
        prefetch, reasons = _prefetch(result, _usage_rows(result))
        for name, value in prefetch.items():
            if name not in values:
                values[name] = value
        for medium in ("dram", "ssd"):
            ops = result.get(f"flat_prefetch_{medium}_ops")
            if type(ops) is int and ops == 0:
                name = f"flat_{medium}_prefetch_latency_ms"
                values[name], reasons[name] = None, "no_samples"
        if all(type(result.get(f"flat_prefetch_{medium}_ops")) is int and result[f"flat_prefetch_{medium}_ops"] == 0 for medium in ("dram", "ssd")):
            values["flat_prefetch_latency_ms"] = None
            reasons["flat_prefetch_latency_ms"] = "no_samples"
    io_status = result.get("flat_io_status", io.get("flat_io_status"))
    io_error = result.get("flat_io_error", io.get("flat_io_error"))
    io_error = (io_error or "flat_io_collection_unavailable") if io_status != "ok" or io_error not in (None, "") else None
    cache_error = None
    if result.get("cache_metadata_status") in ("unavailable", "invalid", "error"):
        cache_error = "; ".join(map(str, result.get("cache_metadata_errors") or ())) or "cache_metadata_unavailable"
    report = {name: deepcopy(result[name]) for name in _RAW_FIELDS if name in result}
    metadata = deepcopy(dict(_mapping(result.get("cache_report_metadata"))))
    for field in REPORT_FIELDS:
        value, meta = _project_field(
            field, values=values, metadata=metadata, selected=selected, versioned=versioned,
            cache_error=cache_error, io_error=io_error, reasons=reasons,
        )
        for name in (field["name"], *field["aliases"]):
            report[name], metadata[name] = value, deepcopy(meta)
    report.update(mode, cache_report_schema_version=version, cache_report_metadata=metadata)
    report["flat_io_status"] = "not_applicable" if selected == "native" else ("ok" if selected == "flat" and io_error is None else "unavailable")
    report["flat_io_error"] = io_error if selected == "flat" else (None if selected == "native" else "mode_unresolved")
    report["cache_report"] = deepcopy(dict(_mapping(result.get("cache_report"))))
    report["cache_report"].update({
        name: report[name] for field in REPORT_FIELDS if field["group"] == "cache"
        for name in (field["name"], *field["aliases"])
    })
    return report


def format_cache_reports(report: Mapping) -> str:
    """Render one mode line, legend and the three shared report sections."""
    report = project_cache_reports(report)
    lines = [
        f"Metrics mode: {report['metrics_mode']} ({report['metrics_mode_status']})",
        "Legend".center(50, "-"),
        "  --  : not tested here",
        "  N/A : unavailable (see reason)",
        "  0   : measured zero",
    ]
    for group, title in (
        ("cache", "Cache Hit Statistics"),
        ("prefetch", "Flat Prefetch Arrival-to-All-TP-GPU-Ready"),
        ("io", "Flat Completed/Durable I/O Window"),
    ):
        lines.append(f"{title:-^50}")
        if group == "cache":
            lines.append("  Hit-rate denominator: all successful server prompt tokens.")
        if group == "io":
            lines.append(f"  Flat I/O status: {report['flat_io_status']}")
        for field in REPORT_FIELDS:
            if field["group"] != group:
                continue
            if group == "cache" and report["metrics_mode"] == "native" and field["applicability"] == "flat":
                continue
            value, meta = report[field["name"]], report["cache_report_metadata"][field["name"]]
            if meta["status"] == "not_applicable":
                text = "--"
            elif value is None:
                text = f"N/A ({meta['reason']})"
            else:
                text = f"{value:,}" if type(value) is int else f"{value:.6g}"
            lines.append(f"  {field['label']} ({field['unit']}): {text}")
        if group == "cache" and report["metrics_mode"] == "flat":
            lines.append("  Mixed is an SSD subset, already included; not a third tier.")
        if group == "io":
            lines.extend((
                "  Bandwidth: node aggregates in decimal GB/s (1 GB = 1e9 bytes).",
                "  Averages: bytes / I/O window; DRAM counter deltas, SSD completed reads / durable writes.",
                "  Peaks: max of summed-rank completion buckets / 100 ms; not physical device maxima.",
            ))
    return "\n".join(lines)


def format_flat_memory_report(result: Mapping, *, mode: Mapping | str | None = None) -> str:
    """Compatibility entry point for the shared report."""
    return format_cache_reports(project_cache_reports(result, mode=mode))
