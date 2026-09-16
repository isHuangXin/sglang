"""Completed, accounted HiCache and Mooncake I/O; snapshots do not drain storage."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from prometheus_client.parser import text_string_to_metric_families

_SCOPE = "completed_accounted_window"
_WRITE_SEMANTICS = "data_synced_bucket_completions_v1"
_FETCH_CAPABILITY = "ssd_to_host_fetch_v1"
_SSD_PEAK_SEMANTICS = "bucket_data_cqe_observed_100ms_v1"
_SSD_WINDOW_PREFIX = "mooncake_ssd_kv_io_"
_SSD_BUCKET_NS = 100_000_000
_SSD_MAX_BUCKETS = 16_384
_SSD_WINDOW_SCALARS = (
    "bucket_width_ns",
    "capacity_buckets",
    "snapshot_ns",
    "oldest_bucket_id",
    "newest_bucket_id",
)
_METRICS = (
    "avg_ssd_to_host_latency_ms",
    "dram_read_bw_gbps",
    "dram_write_bw_gbps",
    "ssd_write_ops",
    "ssd_read_peak_bw_gbps",
    "ssd_write_peak_bw_gbps",
)
_L2_FIELDS = ("bytes", "duration_ns", "batches", "untimed_batches")
_FETCH_FIELDS = ("bytes", "latency_ns_sum", "batches", "errors", "inflight")
_PENDING_FIELDS = ("h2d", "d2h", "storage_prefetch", "storage_backup")


class _SnapshotError(ValueError):
    """Contract errors contain only our messages, not untrusted response text."""


def _mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise _SnapshotError(f"{name} is unavailable")
    return value


def _counter(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise _SnapshotError(f"{name} must be a nonnegative integer counter")
    return value


def _identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _SnapshotError(f"{name} is unavailable")
    return value


def _project_counters(value: Any, fields: tuple[str, ...]) -> dict:
    source = value if isinstance(value, Mapping) else {}
    return {
        key: number if type(number := source.get(key)) is int and number >= 0 else None
        for key in fields
    }


def _project_native(value: Any) -> dict:
    source = value if isinstance(value, Mapping) else {}
    return {
        "schema_version": source.get("schema_version"),
        "instance_id": source.get("instance_id"),
        "capabilities": source.get("capabilities"),
        "status": source.get("status"),
        "ssd_to_host_fetch": _project_counters(
            source.get("ssd_to_host_fetch"), _FETCH_FIELDS
        ),
    }


def project_hicache_snapshot(info: Any) -> dict:
    """Keep only the I/O contract, never the surrounding server configuration."""
    states = _mapping(info, "server info").get("internal_states")
    if not isinstance(states, list) or len(states) != 1:
        raise _SnapshotError("HiCache I/O requires one DP server state")
    source = _mapping(
        _mapping(states[0], "worker state").get("hicache_io"), "hicache_io"
    )
    if (
        source.get("schema_version") != 1
        or type(source.get("schema_version")) is not int
    ):
        raise _SnapshotError("Unsupported HiCache I/O schema")
    if source.get("pp_size") != 1 or source.get("dp_size") != 1:
        raise _SnapshotError("HiCache I/O requires PP1/DP1")
    ranks = source.get("ranks")
    if not isinstance(ranks, list):
        raise _SnapshotError("HiCache I/O rank snapshots are unavailable")
    projected = []
    for rank in ranks:
        rank = _mapping(rank, "rank snapshot")
        l2 = rank.get("l2") or {}
        projected.append(
            {
                **{
                    key: rank.get(key)
                    for key in (
                        "tp_rank",
                        "pp_rank",
                        "dp_rank",
                        "pid",
                        "epoch",
                        "status",
                    )
                },
                "l2": {
                    direction: _project_counters(
                        l2.get(direction) if isinstance(l2, Mapping) else None,
                        _L2_FIELDS,
                    )
                    for direction in ("read", "write")
                },
                "pending": _project_counters(rank.get("pending"), _PENDING_FIELDS),
                "mooncake": _project_native(rank.get("mooncake")),
            }
        )
    return {
        "schema_version": 1,
        "tp_size": source.get("tp_size"),
        "pp_size": 1,
        "dp_size": 1,
        "scope": _SCOPE,
        "drain": False,
        "ranks": projected,
    }


def _parse_data_synced_snapshot(text: str) -> dict:
    samples: dict[str, list] = {}
    names = ("mooncake_ssd_io_info", "mooncake_ssd_data_synced_buckets_completed_total")
    # FLAT_MEMORY: malformed new telemetry must not invalidate the older counters.
    text = "\n".join(
        line
        for line in text.splitlines()
        if not line.lstrip().startswith(
            (
                _SSD_WINDOW_PREFIX,
                f"# HELP {_SSD_WINDOW_PREFIX}",
                f"# TYPE {_SSD_WINDOW_PREFIX}",
            )
        )
    )
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in names:
                samples.setdefault(sample.name, []).append(sample)
    if any(len(samples.get(name, [])) != 1 for name in names):
        raise _SnapshotError(
            "Owner data_synced-bucket metrics are missing or ambiguous"
        )
    info, counter = (samples[name][0] for name in names)
    if (
        info.value != 1
        or info.labels.get("schema_version") != "1"
        or info.labels.get("semantics") != _WRITE_SEMANTICS
    ):
        raise _SnapshotError("Owner data_synced-bucket capability is unsupported")
    instance_id = _identity(info.labels.get("instance_id"), "owner instance")
    if counter.labels and counter.labels != info.labels:
        raise _SnapshotError("Owner counter labels do not match its capability")
    value = counter.value
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        raise _SnapshotError("Owner data_synced-bucket counter is invalid")
    return {
        "schema_version": 1,
        "instance_id": instance_id,
        "semantics": _WRITE_SEMANTICS,
        "data_synced_buckets": int(value),
    }


def _wire_integer(value: str) -> int:
    # FLAT_MEMORY: Prometheus float samples lose native uint64 precision above 2**53.
    if not value or not value.isascii() or not value.isdecimal():
        raise _SnapshotError("SSD byte-window samples require decimal integers")
    number = int(value)
    if number > 2**64 - 1:
        raise _SnapshotError("SSD byte-window sample exceeds uint64")
    return number


def _ssd_window_samples(text: str) -> dict:
    samples = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(_SSD_WINDOW_PREFIX):
            continue
        fields = line.rsplit(None, 1)
        if len(fields) != 2:
            raise _SnapshotError("Malformed SSD byte-window sample")
        value = _wire_integer(fields[1])
        parsed = [
            sample
            for family in text_string_to_metric_families(line)
            for sample in family.samples
        ]
        if len(parsed) != 1 or parsed[0].timestamp is not None:
            raise _SnapshotError("Malformed SSD byte-window sample")
        sample = parsed[0]
        key = (
            sample.name.removeprefix(_SSD_WINDOW_PREFIX),
            frozenset(sample.labels.items()),
        )
        if key in samples:
            raise _SnapshotError("Duplicate SSD byte-window sample")
        samples[key] = value
        if len(samples) > 2 * _SSD_MAX_BUCKETS + 10:
            raise _SnapshotError("SSD byte-window snapshot exceeds bounded capacity")
    return samples


def _take_window_sample(samples: dict, name: str, **labels: str) -> int:
    try:
        return samples.pop((name, frozenset(labels.items())))
    except KeyError:
        raise _SnapshotError(
            "SSD byte-window samples are missing or ambiguous"
        ) from None


def _parse_ssd_window_snapshot(text: str) -> dict:
    samples = _ssd_window_samples(text)
    markers = [key for key in samples if key[0] == "window_info"]
    if len(markers) != 1:
        raise _SnapshotError("Owner SSD byte-window capability is missing or ambiguous")
    marker = markers[0]
    labels = dict(marker[1])
    expected = {
        "schema_version": "1",
        "semantics": _SSD_PEAK_SEMANTICS,
        "clock": "steady_relative_ns",
        "backend": "io_uring",
    }
    if (
        samples.pop(marker) != 1
        or set(labels) != {*expected, "instance_id"}
        or any(labels.get(key) != value for key, value in expected.items())
    ):
        raise _SnapshotError("Owner SSD byte-window capability is unsupported")
    result = {
        **labels,
        "schema_version": 1,
        "instance_id": _identity(labels.get("instance_id"), "SSD byte-window instance"),
        **{field: _take_window_sample(samples, field) for field in _SSD_WINDOW_SCALARS},
    }
    for field in ("completed_bytes_total", "observation_losses_total"):
        result[field] = {
            direction: _take_window_sample(samples, field, direction=direction)
            for direction in ("read", "write")
        }
    buckets: dict[int, dict] = {}
    for (name, raw_labels), value in samples.items():
        labels = dict(raw_labels)
        if name != "bucket_bytes" or set(labels) != {"bucket_id", "direction"}:
            raise _SnapshotError("Unexpected SSD byte-window sample")
        index = _wire_integer(labels["bucket_id"])
        direction = labels["direction"]
        bucket = buckets.setdefault(index, {})
        if direction not in ("read", "write") or direction in bucket:
            raise _SnapshotError("Invalid or duplicate SSD byte-window direction")
        bucket[direction] = value
    result["buckets"] = [
        {"bucket_id": index, **values} for index, values in sorted(buckets.items())
    ]
    _validate_ssd_window(result)
    return result


def parse_owner_snapshot(text: str) -> dict:
    result = {}
    try:
        result.update(_parse_data_synced_snapshot(text))
    except ValueError as error:
        result["data_synced_error"] = (
            str(error)
            if isinstance(error, _SnapshotError)
            else "Owner counters are malformed"
        )
    try:
        result["ssd_kv_io"] = _parse_ssd_window_snapshot(text)
    except ValueError as error:
        result["ssd_kv_io"] = {
            "error": (
                str(error)
                if isinstance(error, _SnapshotError)
                else "Owner SSD byte-window metrics are malformed"
            )
        }
    if result.get("data_synced_error") and result["ssd_kv_io"].get("error"):
        raise _SnapshotError(result["data_synced_error"])
    return result


def _validate_ssd_window(snapshot: Mapping) -> dict[int, Mapping]:
    if snapshot.get("error"):
        raise _SnapshotError(str(snapshot["error"]))
    if (
        type(snapshot.get("schema_version")) is not int
        or snapshot["schema_version"] != 1
        or snapshot.get("semantics") != _SSD_PEAK_SEMANTICS
        or snapshot.get("clock") != "steady_relative_ns"
        or snapshot.get("backend") != "io_uring"
    ):
        raise _SnapshotError("Unsupported SSD byte-window schema")
    _identity(snapshot.get("instance_id"), "SSD byte-window instance")
    values = {
        field: _counter(snapshot.get(field), field) for field in _SSD_WINDOW_SCALARS
    }
    capacity = values["capacity_buckets"]
    if (
        values["bucket_width_ns"] != _SSD_BUCKET_NS
        or not 1 <= capacity <= _SSD_MAX_BUCKETS
    ):
        raise _SnapshotError("Unsupported SSD byte-window width or capacity")
    newest = values["newest_bucket_id"]
    oldest = values["oldest_bucket_id"]
    if newest != values["snapshot_ns"] // _SSD_BUCKET_NS or oldest != max(
        0, newest - capacity + 1
    ):
        raise _SnapshotError("Inconsistent SSD byte-window clock or coverage")
    buckets = snapshot.get("buckets")
    if not isinstance(buckets, list) or len(buckets) > capacity:
        raise _SnapshotError("Missing or oversized SSD byte-window history")
    indexed = {}
    for bucket in buckets:
        bucket = _mapping(bucket, "SSD byte-window bucket")
        index = _counter(bucket.get("bucket_id"), "SSD bucket index")
        if not oldest <= index <= newest or index in indexed:
            raise _SnapshotError("Out-of-range or duplicate SSD byte-window bucket")
        indexed[index] = {
            direction: _counter(bucket.get(direction), f"SSD {direction} bucket bytes")
            for direction in ("read", "write")
        }
    for field in ("completed_bytes_total", "observation_losses_total"):
        counters = _mapping(snapshot.get(field), f"SSD {field}")
        for direction in ("read", "write"):
            value = _counter(counters.get(direction), f"SSD {direction} {field}")
            if (
                field == "completed_bytes_total"
                and sum(bucket[direction] for bucket in indexed.values()) > value
            ):
                raise _SnapshotError(
                    "SSD retained bytes exceed cumulative completed bytes"
                )
    return indexed


def _source_snapshot(sample: Mapping, name: str) -> Mapping:
    source = _mapping(sample.get(name), f"{name} sample")
    if source.get("error"):
        raise _SnapshotError(f"{name} snapshot failed: {source['error']}")
    return _mapping(source.get("snapshot"), f"{name} snapshot")


def _rank_map(snapshot: Mapping) -> dict[int, Mapping]:
    if any(
        type(snapshot.get(field)) is not int or snapshot[field] != 1
        for field in ("schema_version", "pp_size", "dp_size")
    ):
        raise _SnapshotError("Unsupported HiCache I/O schema or PP/DP topology")
    tp_size = _counter(snapshot.get("tp_size"), "TP size")
    if tp_size < 1:
        raise _SnapshotError("TP size must be positive")
    ranks = snapshot.get("ranks")
    if not isinstance(ranks, list):
        raise _SnapshotError("Missing HiCache I/O ranks")
    indexed = {}
    for rank in ranks:
        rank = _mapping(rank, "rank")
        index = _counter(rank.get("tp_rank"), "TP rank")
        if index in indexed or rank.get("pp_rank") != 0 or rank.get("dp_rank") != 0:
            raise _SnapshotError("Duplicate or unsupported rank identity")
        if _counter(rank.get("pid"), "rank pid") < 1:
            raise _SnapshotError("Rank pid must be positive")
        _identity(rank.get("epoch"), "rank epoch")
        indexed[index] = rank
    if set(indexed) != set(range(tp_size)):
        raise _SnapshotError("Incomplete HiCache TP coverage")
    return indexed


def _rank_pairs(before: Mapping, after: Mapping) -> list[tuple[Mapping, Mapping]]:
    begin = _rank_map(_source_snapshot(before, "server"))
    end = _rank_map(_source_snapshot(after, "server"))
    if begin.keys() != end.keys():
        raise _SnapshotError("HiCache rank set changed")
    pairs = []
    for index, left in begin.items():
        right = end[index]
        if (left["epoch"], left["pid"]) != (right["epoch"], right["pid"]):
            raise _SnapshotError("HiCache rank restarted or its cache was replaced")
        pairs.append((left, right))
    return pairs


def _deltas(before: Mapping, after: Mapping, fields: tuple[str, ...]) -> dict[str, int]:
    result = {}
    for field in fields:
        value = _counter(after.get(field), field) - _counter(before.get(field), field)
        if value < 0:
            raise _SnapshotError(f"Counter decreased: {field}")
        result[field] = value
    return result


def _l2_delta(
    pairs: list[tuple[Mapping, Mapping]], direction: str
) -> tuple[dict, bool]:
    total = dict.fromkeys(_L2_FIELDS, 0)
    pending_key = "h2d" if direction == "read" else "d2h"
    pending = False
    for before, after in pairs:
        for rank in (before, after):
            if rank.get("status") != "ok":
                raise _SnapshotError("A rank cannot report measured HiCache bytes")
            pending |= _counter(rank["pending"].get(pending_key), pending_key) > 0
        values = _deltas(before["l2"][direction], after["l2"][direction], _L2_FIELDS)
        if values["untimed_batches"] > values["batches"]:
            raise _SnapshotError("Untimed batch count exceeds completed batch count")
        if not values["batches"] and (values["bytes"] or values["duration_ns"]):
            raise _SnapshotError(
                "HiCache bytes/time have no matching completed batches"
            )
        for field, value in values.items():
            total[field] += value
    return total, pending


def _fetch_delta(pairs: list[tuple[Mapping, Mapping]]) -> tuple[dict, bool]:
    fields = ("bytes", "latency_ns_sum", "batches", "errors")
    total = dict.fromkeys(fields, 0)
    pending = False
    for before, after in pairs:
        for rank in (before, after):
            pending |= (
                _counter(rank["pending"].get("storage_prefetch"), "storage_prefetch")
                > 0
            )
        left, right = before["mooncake"], after["mooncake"]
        for snapshot in (left, right):
            if (
                snapshot.get("schema_version") != 1
                or type(snapshot.get("schema_version")) is not int
                or not isinstance(snapshot.get("capabilities"), list)
                or _FETCH_CAPABILITY not in snapshot["capabilities"]
            ):
                raise _SnapshotError("Native SSD-to-Host fetch metrics are unsupported")
            _identity(snapshot.get("instance_id"), "Mooncake client instance")
            pending |= (
                _counter(snapshot["ssd_to_host_fetch"].get("inflight"), "inflight") > 0
            )
        if left["instance_id"] != right["instance_id"]:
            raise _SnapshotError("Mooncake client restarted")
        values = _deltas(left["ssd_to_host_fetch"], right["ssd_to_host_fetch"], fields)
        if not values["batches"] and (values["bytes"] or values["latency_ns_sum"]):
            raise _SnapshotError(
                "SSD fetch bytes/time have no matching successful batches"
            )
        for field, value in values.items():
            total[field] += value
    return total, pending


def _group_status(statuses: list[str]) -> str:
    if all(status in ("ok", "no_samples") for status in statuses):
        return "ok"
    if all(status == "unavailable" for status in statuses):
        return "unavailable"
    return "partial"


def _empty_result() -> dict:
    result = dict.fromkeys(_METRICS)
    for direction in ("read", "write"):
        result.update(
            {
                f"hicache_io_{direction}_{field}": None
                for field in ("bytes", "batches", "duration_ns")
            }
        )
    return result | {
        "mean_l2_kv_readback_ms": None,
        "hicache_io_status": "unavailable",
        "mooncake_io_status": "unavailable",
        "ssd_read_evidence_status": "unavailable",
        "ssd_write_ops_semantics": _WRITE_SEMANTICS,
        "ssd_to_host_latency_semantics": "successful_host_destination_owner_fetch_batches_v1",
        "dram_read_bw_semantics": "l2_host_to_gpu_active_time_weighted_v1",
        "dram_write_bw_semantics": "l2_gpu_to_host_active_time_weighted_v1",
        "ssd_read_peak_bw_semantics": _SSD_PEAK_SEMANTICS,
        "ssd_write_peak_bw_semantics": _SSD_PEAK_SEMANTICS,
        "ssd_peak_bucket_width_ms": _SSD_BUCKET_NS // 1_000_000,
        "hicache_io_units": {
            "avg_ssd_to_host_latency_ms": "ms",
            "dram_read_bw_gbps": "GB/s",
            "dram_write_bw_gbps": "GB/s",
            "ssd_write_ops": "data_synced_buckets",
            "ssd_read_peak_bw_gbps": "GB/s",
            "ssd_write_peak_bw_gbps": "GB/s",
        },
    }


def summarize_hicache_io(before: Mapping, after: Mapping) -> dict:
    result = _empty_result()
    statuses = {
        field: {"status": "unavailable", "reason": "not_collected"}
        for field in _METRICS
    }
    metadata = {
        "schema_version": 1,
        "scope": _SCOPE,
        "drain": False,
        "before": before,
        "after": after,
        "metric_status": statuses,
        "deltas": {},
    }
    result["hicache_io_metadata"] = metadata
    try:
        pairs = _rank_pairs(before, after)
    except (KeyError, TypeError, ValueError) as error:
        pairs = None
        for name in _METRICS[:3]:
            statuses[name]["reason"] = str(error)
    if pairs is not None:
        _summarize_l2(result=result, pairs=pairs, metadata=metadata)
        _summarize_fetch(result=result, pairs=pairs, metadata=metadata)
    try:
        begin, end = (_source_snapshot(sample, "owner") for sample in (before, after))
        for snapshot in (begin, end):
            if snapshot.get("data_synced_error"):
                raise _SnapshotError(snapshot["data_synced_error"])
            if (
                type(snapshot.get("schema_version")) is not int
                or snapshot["schema_version"] != 1
                or snapshot.get("semantics") != _WRITE_SEMANTICS
            ):
                raise _SnapshotError("Unsupported owner data_synced-bucket schema")
            _identity(snapshot.get("instance_id"), "owner instance")
        if begin["instance_id"] != end["instance_id"]:
            raise _SnapshotError("Mooncake owner restarted")
        value = _deltas(begin, end, ("data_synced_buckets",))["data_synced_buckets"]
        result["ssd_write_ops"] = value
        metadata["deltas"]["ssd_data_synced_buckets"] = value
        statuses["ssd_write_ops"] = {"status": "ok", "reason": None}
    except (KeyError, TypeError, ValueError) as error:
        statuses["ssd_write_ops"]["reason"] = str(error)
    _summarize_ssd_peaks(result=result, before=before, after=after, metadata=metadata)
    result["hicache_io_status"] = _group_status(
        [
            statuses[f"dram_{direction}_bw_gbps"]["status"]
            for direction in ("read", "write")
        ]
    )
    result["mooncake_io_status"] = _group_status(
        [
            statuses[field]["status"]
            for field in (
                "avg_ssd_to_host_latency_ms",
                "ssd_write_ops",
                "ssd_read_peak_bw_gbps",
                "ssd_write_peak_bw_gbps",
            )
        ]
    )
    return result


def _ssd_peak_delta(*, before: Mapping, after: Mapping, direction: str) -> dict:
    begin, end = (
        _mapping(_source_snapshot(sample, "owner").get("ssd_kv_io"), "SSD byte-window")
        for sample in (before, after)
    )
    left, right = (_validate_ssd_window(snapshot) for snapshot in (begin, end))
    if begin["instance_id"] != end["instance_id"]:
        raise _SnapshotError("Mooncake SSD byte-window source restarted")
    if begin["capacity_buckets"] != end["capacity_buckets"]:
        raise _SnapshotError("SSD byte-window capacity changed")
    if end["snapshot_ns"] <= begin["snapshot_ns"]:
        raise _SnapshotError("SSD byte-window duration is not positive")
    first, last = begin["newest_bucket_id"], end["newest_bucket_id"]
    if end["oldest_bucket_id"] > first:
        raise _SnapshotError("SSD byte-window history does not cover the entire case")
    losses = _deltas(
        begin["observation_losses_total"], end["observation_losses_total"], (direction,)
    )[direction]
    if losses:
        raise _SnapshotError("SSD I/O completions were lost or unclassifiable")
    total = _deltas(
        begin["completed_bytes_total"], end["completed_bytes_total"], (direction,)
    )[direction]
    counts = []
    for index in range(first, last + 1):
        value = right.get(index, {}).get(direction, 0)
        if index == first:
            value -= left.get(index, {}).get(direction, 0)
        if value < 0:
            raise _SnapshotError("SSD byte-window bucket counter decreased")
        counts.append(value)
    if sum(counts) != total:
        raise _SnapshotError(
            "SSD byte-window buckets do not match the completed-byte delta"
        )
    return {
        "bytes": total,
        "peak_bucket_bytes": max(counts, default=0),
        "first_bucket_id": first,
        "last_bucket_id": last,
        "bucket_width_ns": _SSD_BUCKET_NS,
        "observation_losses": losses,
    }


def _summarize_ssd_peaks(
    *, result: dict, before: Mapping, after: Mapping, metadata: dict
) -> None:
    for direction in ("read", "write"):
        name = f"ssd_{direction}_peak_bw_gbps"
        try:
            raw = _ssd_peak_delta(before=before, after=after, direction=direction)
            metadata["deltas"][f"ssd_kv_io_{direction}"] = raw
            # FLAT_MEMORY: partial boundary buckets retain the full 100 ms divisor.
            result[name] = raw["peak_bucket_bytes"] / _SSD_BUCKET_NS
            metadata["metric_status"][name] = {"status": "ok", "reason": None}
        except (KeyError, TypeError, ValueError) as error:
            metadata["metric_status"][name]["reason"] = str(error)


def _summarize_l2(*, result: dict, pairs: list, metadata: dict) -> None:
    for direction in ("read", "write"):
        name = f"dram_{direction}_bw_gbps"
        try:
            raw, pending = _l2_delta(pairs, direction)
            metadata["deltas"][f"l2_{direction}"] = raw
            for field in ("bytes", "duration_ns", "batches"):
                result[f"hicache_io_{direction}_{field}"] = raw[field]
            if raw["untimed_batches"]:
                raise _SnapshotError(
                    "CUDA timing is unavailable for completed transfers"
                )
            if not raw["batches"] or not raw["duration_ns"]:
                status = "pending_at_boundary" if pending else "no_samples"
                reason = "no_timed_completed_samples"
            else:
                # Nanoseconds and decimal GB cancel in bytes/ns == GB/s.
                result[name] = raw["bytes"] / raw["duration_ns"]
                if direction == "read":
                    result["mean_l2_kv_readback_ms"] = (
                        raw["duration_ns"] / raw["batches"] / 1e6
                    )
                status = "pending_at_boundary" if pending else "ok"
                reason = "only_completed_accounted_transfers" if pending else None
            metadata["metric_status"][name] = {"status": status, "reason": reason}
        except (KeyError, TypeError, ValueError) as error:
            metadata["metric_status"][name]["reason"] = str(error)


def _summarize_fetch(*, result: dict, pairs: list, metadata: dict) -> None:
    name = "avg_ssd_to_host_latency_ms"
    try:
        raw, pending = _fetch_delta(pairs)
        metadata["deltas"]["ssd_to_host_fetch"] = raw
        if raw["batches"]:
            result[name] = raw["latency_ns_sum"] / raw["batches"] / 1e6
            result["ssd_read_evidence_status"] = (
                "observed" if raw["bytes"] else "unavailable"
            )
            status = "pending_at_boundary" if pending else "ok"
            reason = "successful_completed_fetch_batches_only" if pending else None
        else:
            result["ssd_read_evidence_status"] = "not_observed"
            status = "pending_at_boundary" if pending else "no_samples"
            reason = "no_successful_ssd_fetch_batches"
        metadata["metric_status"][name] = {"status": status, "reason": reason}
    except (KeyError, TypeError, ValueError) as error:
        metadata["metric_status"][name]["reason"] = str(error)


def format_hicache_io_report(result: Mapping) -> str:
    lines = ["----------------HiCache / Mooncake I/O----------------"]
    labels = (
        ("Average SSD --> Host Latency (ms):", "avg_ssd_to_host_latency_ms"),
        ("DRAM Read bandwidth (GB/s):", "dram_read_bw_gbps"),
        ("DRAM Write bandwidth (GB/s):", "dram_write_bw_gbps"),
        ("SSD write operations (data-synced buckets):", "ssd_write_ops"),
        ("SSD Read peak bandwidth, 100 ms (GB/s):", "ssd_read_peak_bw_gbps"),
        ("SSD Write peak bandwidth, 100 ms (GB/s):", "ssd_write_peak_bw_gbps"),
    )
    for label, field in labels:
        value = result.get(field)
        text = (
            "N/A"
            if value is None
            else str(value) if field == "ssd_write_ops" else f"{value:.3f}"
        )
        lines.append(f"{label:<44} {text}")
    lines.extend(
        (
            "DRAM: Host->GPU / GPU->Host, active-time weighted across TP ranks.",
            "SSD latency: successful owner fetch batches to Host, not per-request latency.",
            "SSD writes: data-synced buckets with metadata write success, not keys or NVMe commands.",
            "Metadata is not additionally synced; crash durability is not implied.",
            "SSD peaks: owner KV data-file bytes per fixed 100 ms, counted once across I/O threads.",
            "Completion time is userspace CQE observation, not device time; alignment bytes are included.",
            "Scope: completed/accounted snapshot interval; no drain or request ownership.",
        )
    )
    for name, state in result["hicache_io_metadata"]["metric_status"].items():
        if state["status"] != "ok":
            lines.append(f"{name}: {state['status']} ({state['reason']})")
    return "\n".join(lines)


class HiCacheIOCollector:
    def __init__(
        self,
        *,
        base_url: str,
        enabled: bool,
        owner_metrics_url: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self.owner_metrics_url = owner_metrics_url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.before: dict = {}
        self.result = summarize_hicache_io({}, {})

    async def _fetch(self, *, source: str) -> Any:
        url = (
            self.base_url + "/server_info"
            if source == "server"
            else self.owner_metrics_url
        )
        if not url:
            raise _SnapshotError("Owner metrics URL was not configured")
        parts = urlsplit(url)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
        ):
            raise _SnapshotError(
                "Telemetry requires an HTTP(S) URL without credentials or query parameters"
            )
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            # Credentials for SGLang must never be forwarded to a different owner.
            headers = self.headers if source == "server" else {}
            async with session.get(
                url, headers=headers, allow_redirects=False
            ) as response:
                response.raise_for_status()
                if response.status != 200:
                    raise _SnapshotError(
                        f"Unexpected telemetry HTTP status: {response.status}"
                    )
                return (
                    await response.json()
                    if source == "server"
                    else await response.text()
                )

    async def _sample(self, *, source: str) -> dict:
        start = time.perf_counter()
        snapshot, error = None, None
        try:
            response = await self._fetch(source=source)
            snapshot = (
                project_hicache_snapshot(response)
                if source == "server"
                else parse_owner_snapshot(response)
            )
        except Exception as exc:
            # External parser/HTTP errors can contain private labels or URLs.
            error = (
                str(exc)
                if isinstance(exc, _SnapshotError)
                else f"{type(exc).__name__}: telemetry collection failed"
            )
        return {
            "sample_start_monotonic_s": start,
            "sample_end_monotonic_s": time.perf_counter(),
            "snapshot": snapshot,
            "error": error,
        }

    async def _capture(self) -> dict:
        server, owner = await asyncio.gather(
            self._sample(source="server"), self._sample(source="owner")
        )
        return {"server": server, "owner": owner}

    async def __aenter__(self):
        if self.enabled:
            self.before = {}
            self.result = summarize_hicache_io({}, {})
            self.before = await self._capture()
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        if self.enabled:
            after = await self._capture() if exc_type is None else {}
            self.result = summarize_hicache_io(self.before, after)
        return False
