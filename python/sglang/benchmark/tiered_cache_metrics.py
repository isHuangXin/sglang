"""FLAT_MEMORY: Tiered cache response accounting; missing sources stay unavailable."""

from sglang.benchmark.flat_memory_metrics import _flat_optional_sum


def consume_tiered_cache_metadata(output, meta):
    # FLAT_MEMORY: Streaming details replace prior cumulative values, never add to them.
    if "cached_tokens_details" not in meta:
        return
    details = meta["cached_tokens_details"]
    details = details if isinstance(details, dict) else {}
    mode = details.get("cache_source_mode")
    output.cache_source_mode = mode if isinstance(mode, str) else None
    tiered = details.get("tiered_cache")
    output.tiered_cache = tiered if isinstance(tiered, dict) else None


def summarize_tiered_cache(outputs):
    successful = [output for output in outputs if output.success]
    fields = ("device", "host", "l2_host", "mooncake_dram", "ssd", "mixed")
    result = {
        "tiered_cache_status": "unavailable",
        "tiered_cache_error": None,
        "total_cached_tokens": _flat_optional_sum(successful, "server_cached_tokens"),
        "total_prompt_tokens": _flat_optional_sum(successful, "server_prompt_tokens"),
        "cache_hit_rate": None,
        **{f"tiered_cached_tokens_{name}": None for name in fields},
        **{f"{name}_hit_rate_pct": None for name in ("device", "host", "storage")},
    }
    if not successful:
        result.update(
            total_cached_tokens=0,
            total_prompt_tokens=0,
            tiered_cache_error="No successful requests",
        )
        return result

    def counter(value, name):
        if type(value) is not int or value < 0:
            raise ValueError(f"Missing or invalid {name}")
        return value

    try:
        for index, output in enumerate(successful):
            prompt = counter(
                output.server_prompt_tokens,
                f"prompt tokens for successful request {index}",
            )
            cached = counter(
                output.server_cached_tokens,
                f"cached tokens for successful request {index}",
            )
            if cached > prompt:
                raise ValueError(
                    f"Cached tokens exceed prompt tokens for successful request {index}"
                )
        prompt = result["total_prompt_tokens"]
        if prompt:
            result["cache_hit_rate"] = result["total_cached_tokens"] / prompt
        totals = dict.fromkeys(fields, 0)
        for index, output in enumerate(successful):
            if output.cache_source_mode != "mooncake_tiered_gds" or not isinstance(
                output.tiered_cache, dict
            ):
                raise ValueError(
                    f"Missing Mooncake GDS source metadata for successful request {index}; restart with matching SGLang code"
                )
            counts = {
                name: counter(
                    output.tiered_cache.get(name),
                    f"{name} count for successful request {index}",
                )
                for name in fields
            }
            if (
                counts["host"] != counts["l2_host"] + counts["mooncake_dram"]
                or counts["mixed"] > counts["ssd"]
                or counts["device"] + counts["host"] + counts["ssd"]
                != output.server_cached_tokens
                or counts["device"] != output.cached_tokens_device
                or counts["l2_host"] != output.cached_tokens_host
                or counts["mooncake_dram"] + counts["ssd"]
                != output.cached_tokens_storage
            ):
                raise ValueError(
                    f"Cache source totals are inconsistent for successful request {index}"
                )
            for name in fields:
                totals[name] += counts[name]
        result.update(
            {f"tiered_cached_tokens_{name}": value for name, value in totals.items()}
        )
        if not prompt:
            result["tiered_cache_error"] = "No prompt tokens to define hit rates"
            return result
        result.update(
            tiered_cache_status="ok",
            device_hit_rate_pct=100 * totals["device"] / prompt,
            host_hit_rate_pct=100 * totals["host"] / prompt,
            storage_hit_rate_pct=100 * totals["ssd"] / prompt,
        )
    except ValueError as error:
        result["tiered_cache_error"] = str(error)
    return result


def print_tiered_cache(summary):
    print("----------------Cache Hit Statistics----------------")
    for label, key in (
        ("Total Cached tokens", "total_cached_tokens"),
        ("Total prompt tokens", "total_prompt_tokens"),
        ("Device Hit Rate (GPU HBM)", "device_hit_rate_pct"),
        ("Host Hit Rate (CPU DRAM)", "host_hit_rate_pct"),
        ("Storage Hit Rate (NVMe SSD)", "storage_hit_rate_pct"),
        ("Host detail: L2 Host tokens", "tiered_cached_tokens_l2_host"),
        (
            "Host detail: Mooncake DRAM-only tokens",
            "tiered_cached_tokens_mooncake_dram",
        ),
        ("SSD-dependent tokens", "tiered_cached_tokens_ssd"),
        ("Mixed tokens (subset of SSD)", "tiered_cached_tokens_mixed"),
    ):
        value = summary[key]
        rendered = (
            "N/A"
            if value is None
            else (f"{value:.2f}%" if key.endswith("_pct") else str(value))
        )
        print(f"{label + ':':<44} {rendered}")
    print(f"Cache statistics status: {summary['tiered_cache_status']}")
    if summary["tiered_cache_error"]:
        print(f"Cache statistics detail: {summary['tiered_cache_error']}")
