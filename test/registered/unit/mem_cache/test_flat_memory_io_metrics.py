"""Keep expected capacity pressure distinct from invalid I/O measurements."""

import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
from sglang.benchmark.flat_memory_metrics import (
    FlatIOBucketAccumulator,
    FlatMemoryIOWindow,
    summarize_flat_io,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _snapshot(*, active, pressure):
    return {
        "tp_size": 8,
        "ranks": [
            {
                "tp_rank": rank,
                "pid": 1000 + rank,
                "gpu_id": rank,
                "gds_mode": "compat",
                "pending_backups": 0,
                "pending_prefetches": 0,
                "flat_io_errors": 0,
                "flat_capacity_pressure_offloads": pressure,
                "bandwidth": {
                    "dram_read_total_bytes": 0,
                    "dram_write_total_bytes": 0,
                    "ssd_write_count": 0,
                },
                "io_window": {
                    "enabled": True,
                    "window_id": "window",
                    "active": active,
                    "aborted": False,
                    "overflowed": False,
                    "start_ns": 1,
                    "end_ns": 0 if active else 1_000_000_001,
                    "bucket_ns": 100_000_000,
                    "read_completed_ops": 0,
                    "read_completed_bytes": 0,
                    "write_completed_ops": 0,
                    "write_completed_bytes": 0,
                    "durable_write_batches": 0,
                    "durable_write_bytes": 0,
                    "io_errors": 0,
                    "buckets": [],
                },
            }
            for rank in range(8)
        ],
    }


def _stream_before():
    snapshot = _snapshot(active=True, pressure=0)
    for rank in snapshot["ranks"]:
        rank["io_window"].update(
            schema_version=3,
            bucket_begin=0,
            bucket_end=0,
            dram_read_completed_bytes=0,
            dram_write_completed_bytes=0,
        )
    return snapshot


def _stream_chunk(previous, begin, end, records, *, end_ns=0):
    snapshot = copy.deepcopy(previous)
    fields = {
        "read_bytes": "read_completed_bytes",
        "write_bytes": "durable_write_bytes",
        "dram_read_bytes": "dram_read_completed_bytes",
        "dram_write_bytes": "dram_write_completed_bytes",
    }
    for rank in snapshot["ranks"]:
        window = rank["io_window"]
        window.update(
            active=not end_ns, end_ns=end_ns, bucket_begin=begin, bucket_end=end
        )
        window["buckets"] = []
        for tp_rank, index, value in records:
            if tp_rank != rank["tp_rank"]:
                continue
            window["buckets"].append({"index": index, **dict.fromkeys(fields, value)})
            for counter in fields.values():
                window[counter] += value
            rank["bandwidth"]["dram_read_total_bytes"] += value
            rank["bandwidth"]["dram_write_total_bytes"] += value
    return snapshot


class TestFlatMemoryIOMetrics(CustomTestCase):
    def test_common_pressure_is_not_summed_across_replicated_ranks(self):
        """Three rejected offloads are not 24 rejections merely because TP is eight."""
        before = _snapshot(active=True, pressure=5)
        after = _snapshot(active=False, pressure=8)
        result = summarize_flat_io(before, after, "window", expected_tp_size=8)
        self.assertEqual(result["flat_capacity_pressure_offloads"], 3)

    def test_missing_pressure_telemetry_is_unavailable_not_zero(self):
        before = _snapshot(active=True, pressure=0)
        after = _snapshot(active=False, pressure=3)
        del after["ranks"][0]["flat_capacity_pressure_offloads"]
        result = summarize_flat_io(before, after, "window", expected_tp_size=8)
        self.assertIsNone(result["flat_capacity_pressure_offloads"])

    def test_pressure_does_not_hide_real_io_errors(self):
        """Either a native or scheduler I/O error still invalidates the benchmark."""
        for native in (False, True):
            with self.subTest(native=native):
                before = _snapshot(active=True, pressure=0)
                after = _snapshot(active=False, pressure=3)
                rank = after["ranks"][4]
                if native:
                    rank["io_window"]["io_errors"] = 1
                else:
                    rank["flat_io_errors"] = 1
                with self.assertRaisesRegex(ValueError, "I/O errors"):
                    summarize_flat_io(before, after, "window", expected_tp_size=8)

    def test_inconsistent_or_reset_common_counters_are_rejected(self):
        for pressure in (4, 9):
            with self.subTest(pressure=pressure):
                before = _snapshot(active=True, pressure=5)
                after = _snapshot(active=False, pressure=8)
                after["ranks"][3]["flat_capacity_pressure_offloads"] = pressure
                with self.assertRaises(ValueError):
                    summarize_flat_io(before, after, "window", expected_tp_size=8)

    def test_long_window_merges_tp_peaks_without_retaining_history(self):
        """Long windows must preserve max-of-TP-sums, not sum-of-rank-maxima."""
        before = _stream_before()
        collector = FlatIOBucketAccumulator(before, "window", 8)
        first = _stream_chunk(before, 0, 100, [(0, 0, 100), (1, 0, 7)])
        collector.add(first)
        # An idle gap longer than the old 10000-second limit needs no empty buckets.
        second = _stream_chunk(first, 100, 160001, [(0, 160000, 8), (1, 160000, 200)])
        collector.add(second)
        after = _stream_chunk(second, 160001, 162085, [], end_ns=16_208_485_088_053)
        collector.add(after, final=True)
        summary = collector.summary()
        result = summarize_flat_io(before, after, "window", 8, bucket_summary=summary)
        self.assertAlmostEqual(result["flat_io_window_seconds"], 16208.485088052)
        for field in (
            "flat_ssd_read_peak_bw_gbps",
            "flat_ssd_write_peak_bw_gbps",
            "flat_dram_read_peak_bw_gbps",
            "flat_dram_write_peak_bw_gbps",
        ):
            self.assertAlmostEqual(result[field], 208 / 0.1 / 1e9)
        self.assertAlmostEqual(
            result["flat_ssd_read_bw_gbps"], 315 / 16208.485088052 / 1e9
        )
        self.assertLess(len(json.dumps(summary)), 3000)
        self.assertEqual(summary["collections"], 3)
        broken = copy.deepcopy(summary)
        broken["rank_bytes"][0]["read_bytes"] -= 1
        with self.assertRaisesRegex(ValueError, "cumulative bytes"):
            summarize_flat_io(before, after, "window", 8, bucket_summary=broken)

    def test_incremental_collection_rejects_gaps_duplicates_and_lost_bytes(self):
        """A missing chunk must never become a measured zero or an underestimated peak."""
        before = _stream_before()
        first = _stream_chunk(before, 0, 10, [(0, 2, 100)])
        for failure in ("gap", "duplicate", "rank_boundary", "lost_bytes", "io_error"):
            with self.subTest(failure=failure):
                collector = FlatIOBucketAccumulator(before, "window", 8)
                collector.add(first)
                after = _stream_chunk(first, 10, 21, [], end_ns=2_000_000_001)
                if failure == "gap":
                    for rank in after["ranks"]:
                        rank["io_window"]["bucket_begin"] = 11
                elif failure == "duplicate":
                    after = first
                elif failure == "rank_boundary":
                    after["ranks"][1]["io_window"]["bucket_end"] = 19
                elif failure == "lost_bytes":
                    after["ranks"][0]["io_window"]["read_completed_bytes"] += 1
                else:
                    after["ranks"][0]["io_window"]["io_errors"] = 1
                with self.assertRaises(ValueError):
                    collector.add(after, final=failure != "duplicate")

    def test_completion_at_exact_end_boundary_is_not_dropped(self):
        """The final chunk includes a completion timestamp equal to the shared end."""
        before = _stream_before()
        after = _stream_chunk(before, 0, 11, [(0, 10, 100)], end_ns=1_000_000_001)
        collector = FlatIOBucketAccumulator(before, "window", 8)
        collector.add(after, final=True)
        result = summarize_flat_io(
            before, after, "window", 8, bucket_summary=collector.summary()
        )
        self.assertEqual(result["flat_io_window_seconds"], 1)
        self.assertAlmostEqual(result["flat_ssd_read_peak_bw_gbps"], 100 / 0.1 / 1e9)

    def test_end_waits_for_inflight_collection(self):
        """Ending a benchmark must not race a bucket-consuming control request."""

        async def exercise():
            before = _stream_before()
            first = _stream_chunk(before, 0, 10, [(0, 2, 100)])
            after = _stream_chunk(first, 10, 21, [(1, 10, 50)], end_ns=2_000_000_001)
            entered, release = asyncio.Event(), asyncio.Event()
            calls = []
            window = FlatMemoryIOWindow("http://unused", True, expected_tp_size=8)
            window.window_id = "window"
            window._collection_interval = 0
            window._wait_idle = AsyncMock()

            async def request(action):
                calls.append(action)
                if action == "collect":
                    entered.set()
                    await release.wait()
                    return first
                return {"begin": before, "end": after}[action]

            window._request = request
            await window.__aenter__()
            await entered.wait()
            ending = asyncio.create_task(window.__aexit__(None, None, None))
            await asyncio.sleep(0)
            self.assertNotIn("end", calls)
            release.set()
            await ending
            self.assertEqual(calls, ["begin", "collect", "end"])
            self.assertEqual(window.result["flat_io_status"], "ok")
            self.assertIsNone(window._collector)

        asyncio.run(exercise())

    def test_periodic_failure_keeps_requests_but_invalidates_metrics(self):
        """Telemetry failure must stop consuming buckets, not cancel the request workload."""

        async def exercise():
            window = FlatMemoryIOWindow("http://unused", True, expected_tp_size=8)
            window.window_id = "window"
            window._collection_interval = 0
            window._wait_idle = AsyncMock()
            calls = []

            async def request(action):
                calls.append(action)
                if action == "collect":
                    raise ValueError("missing rank")
                return _stream_before() if action == "begin" else {}

            window._request = request
            with self.assertWarnsRegex(UserWarning, "missing rank"):
                async with window:
                    await window._collector
            self.assertEqual(calls, ["begin", "collect", "abort"])
            self.assertEqual(window.result["flat_io_status"], "unavailable")
            self.assertIsNone(window.result["flat_ssd_read_peak_bw_gbps"])

        asyncio.run(exercise())

    def test_http_error_keeps_server_explanation(self):
        async def exercise():
            response = SimpleNamespace(
                status=409,
                text=AsyncMock(
                    return_value='{"error":"uncollected bucket buffer exhausted"}'
                ),
                raise_for_status=Mock(
                    side_effect=aiohttp.ClientResponseError(
                        SimpleNamespace(real_url="http://unused/flat_memory/io_window"),
                        (),
                        status=409,
                        message="Conflict",
                    )
                ),
            )
            context = AsyncMock()
            context.__aenter__.return_value = response
            session = AsyncMock()
            session.__aenter__.return_value = SimpleNamespace(
                request=Mock(return_value=context)
            )
            with patch(
                "sglang.benchmark.flat_memory_metrics.aiohttp.ClientSession",
                return_value=session,
            ):
                window = FlatMemoryIOWindow("http://unused", True)
                with self.assertRaisesRegex(
                    aiohttp.ClientResponseError, "bucket buffer exhausted"
                ):
                    await window._request("collect")

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
