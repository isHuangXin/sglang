"""HiCache I/O observations must not substitute lifetime proxies or drain storage."""

import ast
import asyncio
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.benchmark.hicache_io_metrics import (
    HiCacheIOCollector,
    format_hicache_io_report,
    parse_owner_snapshot,
    project_hicache_snapshot,
    summarize_hicache_io,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _server_info(*, completed=False):
    ranks = []
    for index in range(2):
        read = {
            "bytes": 1_000_000_000,
            "duration_ns": 1_000_000_000,
            "batches": 10,
            "untimed_batches": 0,
        }
        write = dict(read)
        fetch = {
            "bytes": 100,
            "latency_ns_sum": 1_000_000,
            "batches": 1,
            "errors": 0,
            "inflight": 0,
        }
        if completed:
            read["bytes"] += (2, 6)[index] * 10**9
            read["duration_ns"] += (1, 2)[index] * 10**9
            read["batches"] += (2, 3)[index]
            write["bytes"] += (9, 1)[index] * 10**9
            write["duration_ns"] += (3, 1)[index] * 10**9
            write["batches"] += 1
            fetch["bytes"] += (6000, 2000)[index]
            fetch["latency_ns_sum"] += (90, 10)[index] * 10**6
            fetch["batches"] += (3, 1)[index]
        ranks.append(
            {
                "tp_rank": index,
                "pp_rank": 0,
                "dp_rank": 0,
                "pid": 1000 + index,
                "epoch": f"cache-{index}",
                "status": "ok",
                "l2": {"read": read, "write": write},
                "pending": {
                    "h2d": 0,
                    "d2h": 0,
                    "storage_prefetch": 0,
                    "storage_backup": 0,
                },
                "mooncake": {
                    "schema_version": 1,
                    "instance_id": f"client-{index}",
                    "capabilities": ["ssd_to_host_fetch_v1"],
                    "ssd_to_host_fetch": fetch,
                },
            }
        )
    return {
        "api_key": "not-an-io-field",
        "internal_states": [
            {
                "hicache_io": {
                    "schema_version": 1,
                    "tp_size": 2,
                    "pp_size": 1,
                    "dp_size": 1,
                    "ranks": ranks,
                }
            }
        ],
    }


def _owner_text(*, completed=False, instance="owner-one"):
    return (
        "# TYPE mooncake_ssd_io_info gauge\n"
        f'mooncake_ssd_io_info{{schema_version="1",instance_id="{instance}",semantics="data_synced_bucket_completions_v1"}} 1\n'
        "# TYPE mooncake_ssd_data_synced_buckets_completed_total counter\n"
        f"mooncake_ssd_data_synced_buckets_completed_total {17 if completed else 10}\n"
        "mooncake_ssd_write_ops_total 9000\n"
    )


def _sample(*, completed=False):
    return {
        "server": {
            "snapshot": project_hicache_snapshot(_server_info(completed=completed)),
            "error": None,
        },
        "owner": {
            "snapshot": parse_owner_snapshot(_owner_text(completed=completed)),
            "error": None,
        },
    }


def _ssd_window_text(
    *,
    snapshot_ns,
    buckets=(),
    totals=None,
    losses=(0, 0),
    instance="ssd-window-one",
    capacity=16_384,
):
    prefix = "mooncake_ssd_kv_io_"
    current = snapshot_ns // 100_000_000
    if totals is None:
        totals = tuple(sum(bucket[index] for bucket in buckets) for index in (1, 2))
    lines = [
        (
            f'{prefix}window_info{{schema_version="1",instance_id="{instance}",'
            'semantics="bucket_data_cqe_observed_100ms_v1",clock="steady_relative_ns",'
            'backend="io_uring"} 1'
        ),
        f"{prefix}bucket_width_ns 100000000",
        f"{prefix}capacity_buckets {capacity}",
        f"{prefix}snapshot_ns {snapshot_ns}",
        f"{prefix}oldest_bucket_id {max(0, current - capacity + 1)}",
        f"{prefix}newest_bucket_id {current}",
    ]
    for index, direction in enumerate(("read", "write")):
        lines.extend(
            (
                f'{prefix}completed_bytes_total{{direction="{direction}"}} {totals[index]}',
                f'{prefix}observation_losses_total{{direction="{direction}"}} {losses[index]}',
            )
        )
    for index, read, write in buckets:
        for direction, value in (("read", read), ("write", write)):
            lines.append(
                f'{prefix}bucket_bytes{{bucket_id="{index}",direction="{direction}"}} {value}'
            )
    return "\n".join(lines) + "\n"


def _ssd_sample(*, completed=False, **window):
    sample = _sample(completed=completed)
    sample["owner"]["snapshot"] = parse_owner_snapshot(
        _owner_text(completed=completed) + _ssd_window_text(**window)
    )
    return sample


def _ssd_samples():
    before = _ssd_sample(
        snapshot_ns=150_000_000,
        buckets=((0, 900_000_000, 10_000_000), (1, 50_000_000, 20_000_000)),
    )
    after = _ssd_sample(
        completed=True,
        snapshot_ns=350_000_000,
        buckets=(
            (0, 900_000_000, 10_000_000),
            (1, 100_000_000, 30_000_000),
            (2, 300_000_000, 600_000_000),
            (3, 25_000_000, 40_000_000),
        ),
    )
    return before, after


class TestHiCacheIODeltas(CustomTestCase):
    def test_weighted_rank_bandwidth_and_fetch_batch_denominator(self):
        """Unequal TP workloads require ratios of sums, not averaging rank ratios."""
        result = summarize_hicache_io(_sample(), _sample(completed=True))
        self.assertAlmostEqual(result["dram_read_bw_gbps"], 8 / 3)
        self.assertAlmostEqual(result["dram_write_bw_gbps"], 2.5)
        self.assertEqual(result["avg_ssd_to_host_latency_ms"], 25)
        self.assertEqual(result["ssd_write_ops"], 7)
        self.assertEqual(result["hicache_io_read_bytes"], 8 * 10**9)
        self.assertEqual(result["hicache_io_read_batches"], 5)
        self.assertEqual(result["mean_l2_kv_readback_ms"], 600)
        self.assertEqual(result["ssd_read_evidence_status"], "observed")
        self.assertEqual(result["hicache_io_status"], "ok")
        self.assertEqual(result["mooncake_io_status"], "partial")
        self.assertIsNone(result["ssd_read_peak_bw_gbps"])
        self.assertNotIn("not-an-io-field", json.dumps(result))

    def test_actual_runtime_counter_snapshot_matches_collector_contract(self):
        from sglang.srt.observability.hicache_io import (
            HiCacheIOCounters,
            gather_hicache_io,
        )

        counters = HiCacheIOCounters()

        def snapshot():
            return counters.snapshot() | {
                "pending": {
                    "h2d": 0,
                    "d2h": 0,
                    "storage_prefetch": 0,
                    "storage_backup": 0,
                },
                "mooncake": {"status": "unavailable"},
            }

        def sample():
            state = gather_hicache_io(
                snapshot=snapshot,
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                dp_rank=0,
                dp_size=1,
                gather=lambda rank: [rank],
            )
            return {
                "server": {
                    "snapshot": project_hicache_snapshot(
                        {"internal_states": [{"hicache_io": state}]}
                    ),
                    "error": None,
                },
                "owner": _sample()["owner"],
            }

        before = sample()
        counters.account_completed(
            "read",
            SimpleNamespace(
                num_bytes=4_000_000,
                timing_enabled=True,
                start_event=SimpleNamespace(elapsed_time=lambda _: 2.0),
                finish_event=object(),
            ),
        )
        result = summarize_hicache_io(before, sample())
        self.assertEqual(result["dram_read_bw_gbps"], 2.0)
        self.assertEqual(result["hicache_io_read_duration_ns"], 2_000_000)
        self.assertEqual(result["mean_l2_kv_readback_ms"], 2.0)
        self.assertIsNone(result["avg_ssd_to_host_latency_ms"])

    def test_zero_activity_is_not_missing_telemetry(self):
        result = summarize_hicache_io(_sample(), _sample())
        self.assertEqual(result["ssd_write_ops"], 0)
        self.assertEqual(result["hicache_io_read_bytes"], 0)
        self.assertEqual(result["hicache_io_status"], "ok")
        self.assertEqual(result["ssd_read_evidence_status"], "not_observed")
        for field in (
            "avg_ssd_to_host_latency_ms",
            "dram_read_bw_gbps",
            "dram_write_bw_gbps",
        ):
            self.assertIsNone(result[field])
            self.assertEqual(
                result["hicache_io_metadata"]["metric_status"][field]["status"],
                "no_samples",
            )

    def test_legacy_native_api_does_not_hide_working_l2_or_owner(self):
        before, after = _sample(), _sample(completed=True)
        for sample in (before, after):
            for rank in sample["server"]["snapshot"]["ranks"]:
                rank["mooncake"] = {"status": "unavailable"}
        result = summarize_hicache_io(before, after)
        self.assertIsNone(result["avg_ssd_to_host_latency_ms"])
        self.assertEqual(result["ssd_write_ops"], 7)
        self.assertAlmostEqual(result["dram_read_bw_gbps"], 8 / 3)
        self.assertEqual(result["mooncake_io_status"], "partial")

    def test_failed_owner_does_not_drop_request_side_measurements(self):
        before, after = _sample(), _sample(completed=True)
        after["owner"] = {"snapshot": None, "error": "timeout"}
        result = summarize_hicache_io(before, after)
        self.assertIsNone(result["ssd_write_ops"])
        self.assertEqual(result["avg_ssd_to_host_latency_ms"], 25)
        self.assertAlmostEqual(result["dram_write_bw_gbps"], 2.5)

    def test_missing_duplicate_or_restarted_rank_invalidates_all_tp_sum(self):
        for damage in ("missing", "duplicate", "pid", "epoch", "schema"):
            with self.subTest(damage=damage):
                before, after = _sample(), _sample(completed=True)
                snapshot = after["server"]["snapshot"]
                if damage == "missing":
                    snapshot["ranks"].pop()
                elif damage == "duplicate":
                    snapshot["ranks"][1]["tp_rank"] = 0
                elif damage == "schema":
                    snapshot["schema_version"] = 2
                else:
                    snapshot["ranks"][0][damage] = (
                        500 if damage == "pid" else "new-cache"
                    )
                result = summarize_hicache_io(before, after)
                self.assertIsNone(result["dram_read_bw_gbps"])
                self.assertIsNone(result["avg_ssd_to_host_latency_ms"])
                self.assertEqual(result["ssd_write_ops"], 7)

    def test_counter_rollback_is_not_clamped_to_zero(self):
        before, after = _sample(), _sample(completed=True)
        after["server"]["snapshot"]["ranks"][0]["l2"]["read"]["bytes"] = 1
        result = summarize_hicache_io(before, after)
        self.assertIsNone(result["dram_read_bw_gbps"])
        self.assertEqual(result["dram_write_bw_gbps"], 2.5)
        self.assertIn(
            "decreased",
            result["hicache_io_metadata"]["metric_status"]["dram_read_bw_gbps"][
                "reason"
            ],
        )

    def test_native_client_reset_only_invalidates_ssd_fetch(self):
        before, after = _sample(), _sample(completed=True)
        after["server"]["snapshot"]["ranks"][0]["mooncake"][
            "instance_id"
        ] = "new-client"
        result = summarize_hicache_io(before, after)
        self.assertIsNone(result["avg_ssd_to_host_latency_ms"])
        self.assertEqual(result["ssd_write_ops"], 7)
        self.assertAlmostEqual(result["dram_read_bw_gbps"], 8 / 3)

    def test_owner_epoch_schema_and_counter_are_checked(self):
        for field, value in (
            ("instance_id", "new-owner"),
            ("instance_id", None),
            ("schema_version", 2),
            ("data_synced_buckets", 1),
        ):
            with self.subTest(field=field, value=value):
                before, after = _sample(), _sample(completed=True)
                after["owner"]["snapshot"][field] = value
                result = summarize_hicache_io(before, after)
                self.assertIsNone(result["ssd_write_ops"])
                self.assertEqual(result["avg_ssd_to_host_latency_ms"], 25)

    def test_untimed_transfers_keep_bytes_without_inventing_bandwidth(self):
        before, after = _sample(), _sample(completed=True)
        after["server"]["snapshot"]["ranks"][0]["l2"]["read"]["untimed_batches"] = 1
        result = summarize_hicache_io(before, after)
        self.assertIsNone(result["dram_read_bw_gbps"])
        self.assertEqual(result["hicache_io_read_bytes"], 8 * 10**9)
        self.assertEqual(result["dram_write_bw_gbps"], 2.5)

    def test_pending_boundary_keeps_partial_observation_explicit(self):
        before, after = _sample(), _sample(completed=True)
        after["server"]["snapshot"]["ranks"][0]["pending"]["h2d"] = 1
        after["server"]["snapshot"]["ranks"][0]["mooncake"]["ssd_to_host_fetch"][
            "inflight"
        ] = 2
        result = summarize_hicache_io(before, after)
        self.assertAlmostEqual(result["dram_read_bw_gbps"], 8 / 3)
        self.assertEqual(result["hicache_io_status"], "partial")
        self.assertEqual(result["avg_ssd_to_host_latency_ms"], 25)
        self.assertFalse(result["hicache_io_metadata"]["drain"])
        self.assertIn("pending_at_boundary", format_hicache_io_report(result))

    def test_pending_without_completed_samples_is_still_partial(self):
        before, after = _sample(), _sample()
        rank = after["server"]["snapshot"]["ranks"][0]
        rank["pending"]["h2d"] = 1
        rank["mooncake"]["ssd_to_host_fetch"]["inflight"] = 1
        result = summarize_hicache_io(before, after)
        self.assertIsNone(result["dram_read_bw_gbps"])
        self.assertIsNone(result["avg_ssd_to_host_latency_ms"])
        self.assertEqual(result["hicache_io_status"], "partial")
        self.assertEqual(result["mooncake_io_status"], "partial")
        self.assertIn("pending_at_boundary", format_hicache_io_report(result))

    def test_queued_storage_prefetch_marks_ssd_boundary_partial(self):
        before, after = _sample(), _sample(completed=True)
        after["server"]["snapshot"]["ranks"][0]["pending"]["storage_prefetch"] = 1
        result = summarize_hicache_io(before, after)
        self.assertEqual(result["avg_ssd_to_host_latency_ms"], 25)
        self.assertEqual(result["mooncake_io_status"], "partial")
        self.assertEqual(
            result["hicache_io_metadata"]["metric_status"][
                "avg_ssd_to_host_latency_ms"
            ]["status"],
            "pending_at_boundary",
        )

    def test_projection_omits_non_io_server_and_native_fields(self):
        info = _server_info()
        rank = info["internal_states"][0]["hicache_io"]["ranks"][0]
        rank["secret"] = "do-not-save"
        rank["mooncake"]["config"] = {"api_key": "do-not-save"}
        serialized = json.dumps(project_hicache_snapshot(info))
        self.assertNotIn("do-not-save", serialized)
        self.assertNotIn("api_key", serialized)

    def test_owner_key_counts_cannot_substitute_for_data_synced_buckets(self):
        for text in (
            "mooncake_ssd_write_ops_total 99\n",
            _owner_text().replace(
                "data_synced_bucket_completions_v1", "successful_keys"
            ),
            _owner_text() + "mooncake_ssd_data_synced_buckets_completed_total 8\n",
            _owner_text().replace("completed_total 10", "completed_total NaN"),
            _owner_text().replace("completed_total 10", "completed_total -1"),
            _owner_text().replace("completed_total 10", "completed_total 1.5"),
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_owner_snapshot(text)

    def test_report_distinguishes_missing_from_zero_and_labels_units(self):
        result = summarize_hicache_io(_sample(), _sample())
        report = format_hicache_io_report(result)
        self.assertRegex(report, r"Average SSD --> Host Latency \(ms\):\s+N/A")
        self.assertRegex(report, r"SSD write operations \(data-synced buckets\):\s+0")
        self.assertEqual(result["hicache_io_units"]["dram_read_bw_gbps"], "GB/s")


class TestSSDCompletionWindow(CustomTestCase):
    def test_owner_buckets_exclude_history_and_are_not_multiplied_by_tp(self):
        """An older 9 GB/s bucket must not hide this case's owner-aggregate peak."""
        before, after = _ssd_samples()
        result = summarize_hicache_io(before, after)
        self.assertEqual(result["ssd_read_peak_bw_gbps"], 3.0)
        self.assertEqual(result["ssd_write_peak_bw_gbps"], 6.0)
        self.assertEqual(result["ssd_peak_bucket_width_ms"], 100)
        self.assertEqual(result["mooncake_io_status"], "ok")
        self.assertEqual(result["ssd_write_ops"], 7)
        deltas = result["hicache_io_metadata"]["deltas"]
        self.assertEqual(deltas["ssd_kv_io_read"]["bytes"], 375_000_000)
        self.assertEqual(deltas["ssd_kv_io_write"]["bytes"], 650_000_000)
        self.assertFalse(result["hicache_io_metadata"]["drain"])
        self.assertIn("userspace CQE observation", format_hicache_io_report(result))
        for direction in ("read", "write"):
            self.assertEqual(
                result[f"ssd_{direction}_peak_bw_semantics"],
                "bucket_data_cqe_observed_100ms_v1",
            )
            self.assertEqual(
                result["hicache_io_units"][f"ssd_{direction}_peak_bw_gbps"], "GB/s"
            )

    def test_partial_same_bucket_keeps_full_100ms_denominator(self):
        before = _ssd_sample(snapshot_ns=101_000_000, buckets=((1, 10_000_000, 0),))
        after = _ssd_sample(snapshot_ns=199_000_000, buckets=((1, 20_000_000, 0),))
        result = summarize_hicache_io(before, after)
        self.assertEqual(result["ssd_read_peak_bw_gbps"], 0.1)
        self.assertEqual(result["ssd_write_peak_bw_gbps"], 0.0)
        self.assertRegex(
            format_hicache_io_report(result),
            r"SSD Write peak bandwidth, 100 ms \(GB/s\):\s+0\.000",
        )

    def test_exact_bucket_boundaries_keep_baseline_subtraction(self):
        for start, finish in ((99_999_999, 100_000_000), (199_999_999, 200_000_000)):
            with self.subTest(start=start):
                index = start // 100_000_000
                before = _ssd_sample(
                    snapshot_ns=start, buckets=((index, 50_000_000, 0),)
                )
                after = _ssd_sample(
                    snapshot_ns=finish, buckets=((index, 60_000_000, 0),)
                )
                self.assertEqual(
                    summarize_hicache_io(before, after)["ssd_read_peak_bw_gbps"], 0.1
                )

    def test_idle_window_reports_zero_but_missing_capability_is_unavailable(self):
        before = _ssd_sample(snapshot_ns=150_000_000, buckets=((0, 100, 20),))
        after = _ssd_sample(snapshot_ns=550_000_000, buckets=((0, 100, 20),))
        result = summarize_hicache_io(before, after)
        for direction in ("read", "write"):
            field = f"ssd_{direction}_peak_bw_gbps"
            self.assertEqual(result[field], 0)
            self.assertEqual(
                result["hicache_io_metadata"]["metric_status"][field]["status"], "ok"
            )
        missing = summarize_hicache_io(_sample(), _sample(completed=True))
        self.assertIsNone(missing["ssd_read_peak_bw_gbps"])
        self.assertEqual(missing["ssd_write_ops"], 7)
        self.assertAlmostEqual(missing["dram_read_bw_gbps"], 8 / 3)

    def test_uint64_counter_precision_does_not_erase_one_completed_byte(self):
        before = _ssd_sample(
            snapshot_ns=150_000_000,
            buckets=((0, 2**53, 0), (1, 7, 0)),
        )
        after = _ssd_sample(
            snapshot_ns=199_000_000,
            buckets=((0, 2**53, 0), (1, 8, 0)),
        )
        self.assertEqual(
            before["owner"]["snapshot"]["ssd_kv_io"]["completed_bytes_total"]["read"],
            2**53 + 7,
        )
        self.assertEqual(
            summarize_hicache_io(before, after)["ssd_read_peak_bw_gbps"], 1e-8
        )

    def test_observation_loss_invalidates_only_affected_direction(self):
        before, after = _ssd_samples()
        after["owner"]["snapshot"]["ssd_kv_io"]["observation_losses_total"]["read"] = 1
        result = summarize_hicache_io(before, after)
        self.assertIsNone(result["ssd_read_peak_bw_gbps"])
        self.assertEqual(result["ssd_write_peak_bw_gbps"], 6)
        self.assertEqual(result["ssd_write_ops"], 7)
        self.assertEqual(result["avg_ssd_to_host_latency_ms"], 25)
        self.assertIn(
            "lost",
            result["hicache_io_metadata"]["metric_status"]["ssd_read_peak_bw_gbps"][
                "reason"
            ],
        )

    def test_old_loss_outside_case_does_not_invalidate_new_complete_window(self):
        before, after = _ssd_samples()
        for sample in (before, after):
            sample["owner"]["snapshot"]["ssd_kv_io"]["observation_losses_total"][
                "read"
            ] = 1
        self.assertEqual(
            summarize_hicache_io(before, after)["ssd_read_peak_bw_gbps"], 3
        )

    def test_missing_history_is_not_a_peak_over_only_retained_buckets(self):
        before = _ssd_sample(
            snapshot_ns=150_000_000, capacity=4, buckets=((1, 100, 0),)
        )
        after = _ssd_sample(
            snapshot_ns=550_000_000,
            capacity=4,
            buckets=((5, 200, 0),),
            totals=(300, 0),
        )
        result = summarize_hicache_io(before, after)
        self.assertIsNone(result["ssd_read_peak_bw_gbps"])
        self.assertIn(
            "entire case",
            result["hicache_io_metadata"]["metric_status"]["ssd_read_peak_bw_gbps"][
                "reason"
            ],
        )

    def test_corrupt_window_does_not_invalidate_existing_measurements(self):
        for damage in (
            "instance",
            "schema",
            "width",
            "clock",
            "time",
            "total",
            "missing_bucket",
            "duplicate_bucket",
            "boolean",
        ):
            with self.subTest(damage=damage):
                before, after = _ssd_samples()
                snapshot = after["owner"]["snapshot"]["ssd_kv_io"]
                if damage == "instance":
                    snapshot["instance_id"] = "new-instance"
                elif damage == "schema":
                    snapshot["schema_version"] = 2
                elif damage == "width":
                    snapshot["bucket_width_ns"] = 1_000_000_000
                elif damage == "clock":
                    snapshot["clock"] = "wall_time"
                elif damage == "time":
                    after["owner"]["snapshot"]["ssd_kv_io"] = copy.deepcopy(
                        before["owner"]["snapshot"]["ssd_kv_io"]
                    )
                elif damage == "total":
                    snapshot["completed_bytes_total"]["read"] = 1
                elif damage == "missing_bucket":
                    snapshot["buckets"].pop(2)
                elif damage == "duplicate_bucket":
                    snapshot["buckets"].append(copy.deepcopy(snapshot["buckets"][0]))
                else:
                    snapshot["snapshot_ns"] = True
                result = summarize_hicache_io(before, after)
                self.assertIsNone(result["ssd_read_peak_bw_gbps"])
                self.assertEqual(result["ssd_write_ops"], 7)
                self.assertEqual(result["avg_ssd_to_host_latency_ms"], 25)
                self.assertAlmostEqual(result["dram_read_bw_gbps"], 8 / 3)

    def test_malformed_new_wire_samples_preserve_old_owner_counters(self):
        text = _ssd_window_text(snapshot_ns=350_000_000, buckets=((2, 100, 200),))
        for malformed in (
            text.replace("100ms_v1", "other_v1"),
            text + "mooncake_ssd_kv_io_snapshot_ns 350000000\n",
            text.replace("snapshot_ns 350000000", "snapshot_ns NaN"),
            text.replace("snapshot_ns 350000000", "snapshot_ns -1"),
            text.replace("snapshot_ns 350000000", "snapshot_ns 3.5e8"),
            text.replace('direction="read"} 100', 'direction="read"} 100.5'),
            text + 'mooncake_ssd_kv_io_bad{api_key="private-window-marker",oops} 1\n',
        ):
            with self.subTest(malformed=malformed):
                before, after = _ssd_samples()
                after["owner"]["snapshot"] = parse_owner_snapshot(
                    _owner_text(completed=True) + malformed
                )
                result = summarize_hicache_io(before, after)
                self.assertIsNone(result["ssd_read_peak_bw_gbps"])
                self.assertEqual(result["ssd_write_ops"], 7)
                self.assertNotIn("private-window-marker", json.dumps(result))

    def test_bad_durable_bucket_capability_does_not_hide_valid_ssd_peaks(self):
        before, after = _ssd_samples()
        for sample in (before, after):
            snapshot = sample["owner"]["snapshot"]["ssd_kv_io"]
            sample["owner"]["snapshot"] = parse_owner_snapshot(
                "mooncake_ssd_write_ops_total 9999\n"
                + _ssd_window_text(
                    snapshot_ns=snapshot["snapshot_ns"],
                    buckets=tuple(
                        (b["bucket_id"], b["read"], b["write"])
                        for b in snapshot["buckets"]
                    ),
                )
            )
        result = summarize_hicache_io(before, after)
        self.assertIsNone(result["ssd_write_ops"])
        self.assertEqual(result["ssd_read_peak_bw_gbps"], 3)
        self.assertEqual(result["ssd_write_peak_bw_gbps"], 6)

    def test_no_baseline_or_failed_capture_never_assumes_zero(self):
        before, after = _ssd_samples()
        for left, right in (
            ({}, after),
            (before, {}),
            ({"owner": {"error": "timeout"}}, after),
        ):
            with self.subTest(left=left):
                result = summarize_hicache_io(left, right)
                self.assertIsNone(result["ssd_read_peak_bw_gbps"])
                self.assertIsNone(result["ssd_write_peak_bw_gbps"])


class TestHiCacheIOCollector(CustomTestCase):

    def test_reused_collector_takes_fresh_case_baseline_without_polling(self):
        for failed_baseline in (False, True):
            with self.subTest(failed_baseline=failed_baseline):
                before, after = _ssd_samples()
                last = copy.deepcopy(after["owner"]["snapshot"]["ssd_kv_io"])
                last["snapshot_ns"] = 550_000_000
                last["buckets"].append({"bucket_id": 4, "read": 5_000_000, "write": 0})
                windows = [
                    sample["owner"]["snapshot"]["ssd_kv_io"]
                    for sample in (before, after, after)
                ] + [last]
                owner_calls = 0
                collector = HiCacheIOCollector(base_url="http://server", enabled=True)

                async def fetch(*, source, samples=windows, fail_baseline=failed_baseline):
                    nonlocal owner_calls
                    if source == "server":
                        return _server_info()
                    window = samples[owner_calls]
                    owner_calls += 1
                    if fail_baseline and owner_calls == 3:
                        raise TimeoutError("missing baseline")
                    return _owner_text() + _ssd_window_text(
                        snapshot_ns=window["snapshot_ns"],
                        buckets=tuple(
                            (b["bucket_id"], b["read"], b["write"])
                            for b in window["buckets"]
                        ),
                    )

                async def run(active_collector=collector):
                    peaks = []
                    for _ in range(2):
                        async with active_collector:
                            pass
                        peaks.append(active_collector.result["ssd_read_peak_bw_gbps"])
                    return peaks

                with patch.object(collector, "_fetch", side_effect=fetch) as mock:
                    peaks = asyncio.run(run())
                self.assertEqual(peaks, [3.0, None if failed_baseline else 0.05])
                self.assertEqual(mock.call_count, 8)
                self.assertEqual(owner_calls, 4)

    def test_disabled_collector_never_requests_an_endpoint(self):
        collector = HiCacheIOCollector(base_url="http://server", enabled=False)
        with patch.object(collector, "_fetch", new_callable=AsyncMock) as fetch:

            async def run():
                async with collector:
                    pass

            asyncio.run(run())
            fetch.assert_not_awaited()

    def test_telemetry_failure_keeps_other_source_and_does_not_escape(self):
        collector = HiCacheIOCollector(base_url="http://server", enabled=True)
        completed = False

        async def fetch(*, source):
            if source == "owner":
                raise TimeoutError("owner unavailable")
            return _server_info(completed=completed)

        with patch.object(collector, "_fetch", side_effect=fetch):

            async def run():
                nonlocal completed
                async with collector:
                    completed = True

            asyncio.run(run())
        self.assertEqual(collector.result["avg_ssd_to_host_latency_ms"], 25)
        self.assertIsNone(collector.result["ssd_write_ops"])

    def test_parser_error_cannot_leak_unrelated_metric_labels(self):
        collector = HiCacheIOCollector(base_url="http://server", enabled=True)
        completed = False

        async def fetch(*, source):
            if source == "owner":
                return (
                    _owner_text()
                    + 'unrelated_info{api_key="private-test-marker",oops} 1\n'
                )
            return _server_info(completed=completed)

        with patch.object(collector, "_fetch", side_effect=fetch):

            async def run():
                nonlocal completed
                async with collector:
                    completed = True

            asyncio.run(run())
        self.assertIsNone(collector.result["ssd_write_ops"])
        self.assertEqual(collector.result["avg_ssd_to_host_latency_ms"], 25)
        self.assertNotIn("private-test-marker", json.dumps(collector.result))
        self.assertNotIn(
            "private-test-marker", format_hicache_io_report(collector.result)
        )

    def test_production_request_timer_excludes_both_snapshot_pairs(self):
        """Execute the real request-timing block with fake requests, not its reimplementation."""
        serving_path = (
            Path(__file__).resolve().parents[4] / "python/sglang/benchmark/serving.py"
        )
        tree = ast.parse(serving_path.read_text())
        block = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncWith)
            and any(
                isinstance(item.context_expr, ast.Name)
                and item.context_expr.id == "hicache_io"
                for item in node.items
            )
        )
        function = ast.AsyncFunctionDef(
            name="timing_block",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=[
                copy.deepcopy(block),
                ast.Return(value=ast.Name(id="benchmark_duration", ctx=ast.Load())),
            ],
            decorator_list=[],
        )
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        clock = [0.0]
        collector = HiCacheIOCollector(base_url="http://server", enabled=True)

        async def fetch(*, source):
            clock[0] += 5
            return _server_info() if source == "server" else _owner_text()

        async def requests():
            if False:
                yield None

        async def gather(*tasks, **kwargs):
            clock[0] += 2
            return []

        namespace = {
            "flat_io": HiCacheIOCollector(base_url="http://server", enabled=False),
            "hicache_io": collector,
            "benchmark_requests": [],
            "request_generator": requests(),
            "asyncio": SimpleNamespace(gather=gather),
            "tasks": [],
            "pbar": None,
            "time": SimpleNamespace(perf_counter=lambda: clock[0]),
        }
        exec(compile(module, str(serving_path), "exec"), namespace)
        with patch.object(collector, "_fetch", side_effect=fetch):
            duration = asyncio.run(namespace["timing_block"]())
        self.assertEqual(duration, 2)
        self.assertEqual(clock[0], 22)

    def test_request_exception_is_not_suppressed_and_does_not_drain(self):
        collector = HiCacheIOCollector(base_url="http://server", enabled=True)

        async def fetch(*, source):
            return _server_info() if source == "server" else _owner_text()

        with patch.object(collector, "_fetch", side_effect=fetch) as mock:

            async def run():
                async with collector:
                    raise RuntimeError("request interrupted")

            with self.assertRaisesRegex(RuntimeError, "request interrupted"):
                asyncio.run(run())
            self.assertEqual(mock.call_count, 2)

    def test_observation_uses_only_get_and_keeps_owner_headers_separate(self):
        calls = []

        class Response:
            status = 200

            def __init__(self, url):
                self.url = url

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def raise_for_status(self):
                return None

            async def json(self):
                return _server_info()

            async def text(self):
                return _owner_text()

        class Session:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def get(self, url, **kwargs):
                calls.append((url, kwargs))
                return Response(url)

        collector = HiCacheIOCollector(
            base_url="http://server:30076",
            enabled=True,
            owner_metrics_url="http://owner:9302/metrics",
            headers={"Authorization": "private"},
        )
        with patch(
            "sglang.benchmark.hicache_io_metrics.aiohttp.ClientSession", Session
        ):

            async def run():
                async with collector:
                    pass

            asyncio.run(run())
        self.assertEqual(len(calls), 4)
        for url, options in calls:
            self.assertFalse(options["allow_redirects"])
            if url.endswith("/metrics"):
                self.assertEqual(options["headers"], {})
            else:
                self.assertEqual(url, "http://server:30076/server_info")
                self.assertEqual(options["headers"], {"Authorization": "private"})
        self.assertNotIn("private", json.dumps(collector.result))


if __name__ == "__main__":
    unittest.main()
