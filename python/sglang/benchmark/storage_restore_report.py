"""Versioned request-restore timing validation and CPU-only aggregation."""

import math
from collections import Counter


SCHEMA_VERSION = 1
SCOPE = "scheduler_prefetch_start_to_all_tp_gpu_ready_observed"
SAMPLE_UNIT = "first_successful_adopted_restore_per_successful_request"
CLOCK = "monotonic_ns"
DETAIL_KEY = "storage_restore_timing"
REPORT_KEY = "storage_restore_timing_report"
RECORDS_KEY = "storage_restore_timing_records"
MEMORY = 1
LOCAL_DISK = 2
ENDPOINTS = frozenset(("flat_all_tp_ready", "host_all_layer_ack_all_tp_ready"))
STATUSES = frozenset(("pending", "completed", "miss", "cancelled", "failed", "discarded", "not_applicable"))
FIELDS = {
    "all": "storage_restore_ready_ms",
    "ssd": "ssd_storage_restore_ready_ms",
    "dram": "dram_storage_restore_ready_ms",
}


def _integer(value, *, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid_{name}")
    return value


def _milliseconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid_elapsed_ms")
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid_elapsed_ms")
    return float(value)


def validate_timing(payload, *, expected_tp_size=None):
    """Validate producer evidence without interpreting legacy timing aliases."""
    if not isinstance(payload, dict):
        raise ValueError("timing_not_collected")
    if type(payload.get("schema_version")) is not int:
        raise ValueError("unsupported_schema_version")
    if expected_tp_size is not None:
        _integer(expected_tp_size, name="expected_tp_size", minimum=1)
    for key, expected in (("schema_version", SCHEMA_VERSION), ("scope", SCOPE),
                          ("sample_unit", SAMPLE_UNIT), ("clock", CLOCK)):
        if payload.get(key) != expected:
            raise ValueError(f"unsupported_{key}")
    tp_size = _integer(payload.get("tp_size"), name="tp_size", minimum=1)
    if expected_tp_size is not None and tp_size != expected_tp_size:
        raise ValueError("tp_size_mismatch")
    if payload.get("status") != "available":
        raise ValueError(payload.get("reason") or "producer_unavailable")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("invalid_episodes")
    seen, valid, outcomes = set(), [], Counter()
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("invalid_episode")
        episode_id = _integer(episode.get("episode_id"), name="episode_id")
        if episode_id in seen:
            raise ValueError("duplicate_episode_id")
        seen.add(episode_id)
        status = episode.get("status")
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError("invalid_episode_status")
        outcomes[status] += 1
        if status != "completed":
            continue
        start = _integer(episode.get("start_ns"), name="start_ns")
        ready = _integer(episode.get("ready_ns"), name="ready_ns")
        elapsed = _milliseconds(episode.get("elapsed_ms"))
        if ready < start or not math.isclose(elapsed, (ready - start) / 1e6, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("timing_interval_mismatch")
        restored = _integer(episode.get("restored_tokens"), name="restored_tokens", minimum=1)
        adopted = _integer(episode.get("adopted_tokens"), name="adopted_tokens", minimum=1)
        if adopted > restored:
            raise ValueError("adopted_tokens_exceed_restore")
        endpoint = episode.get("endpoint")
        if (episode.get("all_tp_ready") is not True
                or not isinstance(endpoint, str) or endpoint not in ENDPOINTS):
            raise ValueError("missing_all_tp_gpu_ready_evidence")
        mask = _integer(episode.get("source_mask"), name="source_mask")
        complete = episode.get("source_complete")
        if mask > (MEMORY | LOCAL_DISK) or type(complete) is not bool:
            raise ValueError("invalid_source_evidence")
        if complete and mask == 0:
            raise ValueError("complete_source_is_unknown")
        valid.append({"episode_id": episode_id, "elapsed_ms": elapsed,
                      "restored_tokens": restored, "adopted_tokens": adopted,
                      "source_mask": mask, "source_complete": complete})
    if seen != set(range(len(episodes))):
        raise ValueError("incomplete_episode_history")
    if not valid and outcomes["pending"]:
        raise ValueError("incomplete_restore_episode")
    valid.sort(key=lambda episode: episode["episode_id"])
    return valid, dict(outcomes)


def _field(values, *, reason=None):
    count = len(values)
    total = sum(values)
    if reason is not None:
        return {"value": None, "unit": "ms", "status": "unavailable", "reason": reason,
                "sample_count": count, "elapsed_sum_ms": total, "source": RECORDS_KEY}
    if not count:
        return {"value": None, "unit": "ms", "status": "unavailable", "reason": "no_samples",
                "sample_count": 0, "elapsed_sum_ms": 0.0, "source": RECORDS_KEY}
    return {"value": total / count, "unit": "ms", "status": "measured", "reason": None,
            "sample_count": count, "elapsed_sum_ms": total, "source": RECORDS_KEY}


def summarize_restore_records(records, *, expected_tp_size=None, expected_requests=None):
    """Reduce one final metadata snapshot per request, selecting at most one episode."""
    if expected_tp_size is not None:
        _integer(expected_tp_size, name="expected_tp_size", minimum=1)
    if expected_requests is not None:
        _integer(expected_requests, name="expected_requests")
    resolved_tp_size = expected_tp_size
    populations = {name: [] for name in FIELDS}
    counts, reasons = Counter(), Counter()
    seen = set()
    if not isinstance(records, list):
        records = []
        reasons["request_records_not_collected"] += 1
    if expected_requests is not None and len(records) != expected_requests:
        reasons["request_count_mismatch"] += 1
    for record in records:
        counts["requests"] += 1
        if not isinstance(record, dict) or type(record.get("success")) is not bool:
            reasons["invalid_request_record"] += 1
            continue
        if not record["success"]:
            counts["unsuccessful_requests"] += 1
            continue
        counts["successful_requests"] += 1
        rid = record.get("request_id")
        if not isinstance(rid, str) or not rid:
            reasons["missing_request_id"] += 1
            continue
        if rid in seen:
            reasons["duplicate_request_id"] += 1
            continue
        seen.add(rid)
        try:
            episodes, outcomes = validate_timing(record.get("timing"), expected_tp_size=resolved_tp_size)
            if resolved_tp_size is None:
                resolved_tp_size = record["timing"]["tp_size"]
        except ValueError as exc:
            counts["invalid_or_missing_timing_requests"] += 1
            reasons[str(exc)] += 1
            continue
        counts["requests_with_valid_timing_metadata"] += 1
        for status, count in outcomes.items():
            counts[f"episodes_{status}"] += count
        if not episodes:
            counts["requests_without_eligible_restore"] += 1
            continue
        sample = episodes[0]
        counts["selected_requests"] += 1
        counts["additional_completed_episodes"] += len(episodes) - 1
        populations["all"].append(sample["elapsed_ms"])
        if not sample["source_complete"]:
            counts["source_unknown_requests"] += 1
        elif sample["source_mask"] & LOCAL_DISK:
            counts["ssd_dependent_requests"] += 1
            populations["ssd"].append(sample["elapsed_ms"])
        else:
            counts["dram_only_requests"] += 1
            populations["dram"].append(sample["elapsed_ms"])
    invalid_reason = "invalid_or_missing_request_evidence" if reasons else None
    source_reason = invalid_reason or ("source_evidence_incomplete" if counts["source_unknown_requests"] else None)
    fields = {field: _field(populations[name], reason=invalid_reason if name == "all" else source_reason)
              for name, field in FIELDS.items()}
    return {"schema_version": SCHEMA_VERSION, "scope": SCOPE, "sample_unit": SAMPLE_UNIT,
            "clock": CLOCK, "tp_size": resolved_tp_size, "fields": fields,
            "counts": dict(counts), "validation_reasons": dict(reasons),
            "source": RECORDS_KEY}


def validate_restore_report(report, records, *, expected_tp_size, expected_requests):
    """Recompute a report from request evidence and reject summary disagreement."""
    expected = summarize_restore_records(records, expected_tp_size=expected_tp_size,
                                         expected_requests=expected_requests)
    if not isinstance(report, dict):
        raise ValueError("restore_report_not_collected")
    for key in ("schema_version", "scope", "sample_unit", "clock", "tp_size", "counts", "validation_reasons", "source"):
        if type(report.get(key)) is not type(expected[key]) or report.get(key) != expected[key]:
            raise ValueError(f"restore_report_{key}_mismatch")
    if report.get("fields") != expected["fields"]:
        raise ValueError("restore_report_fields_mismatch")
    return expected
