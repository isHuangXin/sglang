"""Keep expected capacity pressure distinct from invalid I/O measurements."""

import unittest

from sglang.benchmark.flat_memory_metrics import summarize_flat_io
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


if __name__ == "__main__":
    unittest.main()
