"""FLAT_MEMORY: Shared ASCII report, loadable without importing SGLang."""

import math
from typing import Mapping


def _counter(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _number(value: object) -> int | float | None:
    if type(value) is int:
        return value if value >= 0 else None
    if type(value) is float and math.isfinite(value) and value >= 0:
        return value
    return None


def _format_counter(value: object) -> str:
    value = _counter(value)
    return f"{value:,}" if value is not None else "N/A"


def _format_number(value: object) -> str:
    value = _number(value)
    return f"{value:.4g}" if value is not None else "N/A"


def _hit_rate(tokens: int | None, prompt: int | None) -> str:
    if tokens is None or prompt is None or prompt == 0 or tokens > prompt:
        return "N/A"
    return f"{100 * tokens / prompt:.2f}"


def format_flat_memory_report(metrics: Mapping) -> str:
    """Format merged native cache/I/O metrics or a benchmark case summary.

    Missing or invalid telemetry stays N/A; cache counts are independent of the
    I/O collection status. Rates use all server-reported prompt tokens.
    """
    prompt = _counter(metrics.get("total_prompt_tokens"))
    io_ok = metrics.get("flat_io_status") == "ok" and metrics.get("flat_io_error") in (
        None,
        "",
    )
    headers = (
        "Tier",
        "KV hit tokens",
        "Hit rate %",
        "Avg read",
        "Avg write",
        "Peak read",
        "Peak write",
    )
    rows = [headers]
    for medium in ("dram", "ssd"):
        hits = _counter(metrics.get(f"flat_cached_tokens_{medium}"))
        bandwidths = tuple(
            _format_number(metrics.get(f"flat_{medium}_{direction}_{kind}bw_gbps"))
            if io_ok
            else "N/A"
            for kind in ("", "peak_")
            for direction in ("read", "write")
        )
        rows.append(
            (medium.upper(), _format_counter(hits), _hit_rate(hits, prompt), *bandwidths)
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    separator = "-+-".join("-" * width for width in widths)
    table = [
        " | ".join(
            cell.ljust(width) if index == 0 else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(row, widths))
        )
        for row in rows
    ]
    lines = [
        "Flat Memory cache and I/O",
        f"Hit-rate denominator: {_format_counter(prompt)} input tokens "
        "(all server prompt tokens).",
    ]
    if "cache_hit_rate" in metrics:
        overall = _number(metrics["cache_hit_rate"])
        rate = f"{100 * overall:.2f}%" if overall is not None and overall <= 1 else "N/A"
        lines.append(f"Overall cached rate (all cache tiers): {rate}")
    lines.extend([table[0], separator, *table[1:]])
    lines.extend(
        [
            "Mixed KV hit tokens: "
            f"{_format_counter(metrics.get('flat_cached_tokens_mixed'))} "
            "(SSD subset, already included; not a third tier).",
            "Bandwidth: node aggregates in decimal GB/s (1 GB = 1e9 bytes).",
            "Averages: bytes / I/O window; DRAM lifetime-counter deltas, "
            "SSD completed reads / durable writes.",
            "Peaks: max of summed-rank completion buckets / 100 ms "
            "(including partial buckets), not device ceilings.",
            "N/A means unavailable, not zero.",
        ]
    )
    if not io_ok:
        lines.append("I/O collection unavailable; cache-hit metrics remain independent.")
    elif any(
        _number(metrics.get(f"flat_dram_{direction}_peak_bw_gbps")) is None
        for direction in ("read", "write")
    ):
        lines.append(
            "DRAM peaks require native I/O schema v2: update/rebuild the native "
            "extension and restart the server."
        )
    return "\n".join(lines)
