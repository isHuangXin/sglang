"""Flat snapshots expose the bound CUDA device to the native I/O consumer."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.benchmark.flat_memory_metrics import _flat_rank_map
from sglang.srt.mem_cache.storage.flat_memory.flat_memory_direct_linker import (
    FlatMemoryDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFlatLinkerMetrics(CustomTestCase):
    def test_snapshot_reports_bound_gpu_not_tp_rank(self):
        """The I/O rank validator accepts snapshots without guessing GPU identity."""
        linker = FlatMemoryDirectLinker.__new__(FlatMemoryDirectLinker)
        linker._closed = False
        linker.tp_rank = 0
        linker.device = torch.device("cuda:3")
        linker.manager = SimpleNamespace(
            get_stats=lambda: {},
            get_bandwidth_report=lambda: {},
            get_capacity_stats=lambda: {},
            get_cio_stats=lambda: {},
            get_io_window=lambda: {},
        )
        linker.transfer = SimpleNamespace(get_stats=lambda: {})
        with patch.object(
            torch.cuda, "current_device", side_effect=AssertionError("No CUDA query")
        ):
            snapshot = linker.get_flat_memory_stats()
        rank = {**snapshot, "tp_rank": 0, "pid": 12345}
        mapped = _flat_rank_map({"tp_size": 1, "ranks": [rank]}, expected_tp_size=1)
        self.assertEqual(mapped[0]["gpu_id"], 3)
        self.assertEqual(snapshot["storage"], {})
        self.assertEqual(snapshot["io_window"], {})


if __name__ == "__main__":
    unittest.main()
