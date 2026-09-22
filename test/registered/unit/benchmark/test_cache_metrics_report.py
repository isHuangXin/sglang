"""CPU checks for cache applicability, server denominators and completed I/O."""

import asyncio
import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.benchmark import flat_memory_report as report
from sglang.benchmark.flat_memory_metrics import (
    FlatMemoryIOWindow,
    consume_native_usage,
    summarize_cache_usage,
    summarize_flat_io,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def window_fixture(active):
    ranks = []
    for index in range(4):
        size = (100_000_000, 300_000_000, 0, 0)[index] if not active else 0
        window = dict(
            schema_version=2,
            enabled=True,
            window_id="owned",
            active=active,
            aborted=False,
            overflowed=False,
            start_ns=1_000_000_000,
            end_ns=0 if active else 1_200_000_000,
            bucket_ns=100_000_000,
            read_completed_ops=int(bool(size)),
            read_completed_bytes=size,
            write_completed_ops=int(bool(size)),
            write_completed_bytes=size,
            durable_write_batches=int(bool(size)),
            durable_write_bytes=size,
            dram_read_completed_bytes=size,
            dram_write_completed_bytes=size,
            io_errors=0,
            buckets=(
                []
                if not size
                else [
                    dict(
                        index=index,
                        read_bytes=size,
                        write_bytes=size,
                        dram_read_bytes=size,
                        dram_write_bytes=size,
                    )
                ]
            ),
        )
        ranks.append(
            dict(
                tp_rank=index,
                tp_size=4,
                pid=10 + index,
                gpu_id=index,
                gds_mode="compat",
                pending_backups=0,
                pending_prefetches=0,
                flat_io_errors=0,
                flat_capacity_pressure_offloads=0,
                bandwidth=dict(
                    ssd_write_count=0 if active else index + 1,
                    dram_read_total_bytes=10**15,
                ),
                io_window=window,
            )
        )
    return dict(tp_size=4, ranks=ranks)


class TestCacheMetricsReport(CustomTestCase):
    def test_server_denominator_and_raw_nonmutation(self):
        raw = dict(
            input_lens=[1],
            completed=1,
            errors=[""],
            server_prompt_tokens=[6547200],
            server_cached_tokens=[4712448],
            cached_tokens_details=[
                dict(
                    cache_source_mode="tiered",
                    device=3139584,
                    host=1493760,
                    storage=79104,
                    unknown=dict(keep=True),
                )
            ],
        )
        original = deepcopy(raw)
        projected = report.project_cache_reports(raw)
        self.assertEqual(projected["total_prompt_tokens"], 6547200)
        self.assertAlmostEqual(projected["cache_hit_rate"], 4712448 / 6547200)
        self.assertAlmostEqual(projected["storage_hit_rate_pct"], 100 * 79104 / 6547200)
        self.assertEqual(projected["offdevice_cached_tokens"], 1572864)
        self.assertEqual(projected["host_cached_tokens"], 1493760)
        self.assertIsNone(projected["flat_ssd_read_bw_gbps"])
        self.assertEqual(
            projected["cache_report_metadata"]["flat_ssd_read_bw_gbps"]["status"],
            "not_applicable",
        )
        self.assertEqual(raw, original)
        self.assertNotIn("L3 SSD", report.format_cache_reports(projected))
        impossible = dict(
            cache_source_mode="native",
            server_prompt_tokens=[1, 99],
            server_cached_tokens=[2, 48],
            cached_tokens_details=[
                dict(device=2, host=0, storage=0),
                dict(device=48, host=0, storage=0),
            ],
        )
        self.assertIsNone(
            report.project_cache_reports(impossible)["total_cached_tokens"]
        )
        impossible["server_prompt_tokens"] = [50, 50]
        impossible["cached_tokens_details"] = [
            dict(device=1, host=0, storage=0),
            dict(device=49, host=0, storage=0),
        ]
        self.assertIsNone(
            report.project_cache_reports(impossible)["device_cached_tokens"]
        )

    def test_cold_zero_warm_missing_and_conflicting_modes(self):
        raw = dict(
            server_info=dict(enable_hierarchical_cache=False),
            server_prompt_tokens=[100],
            server_cached_tokens=[0],
            cached_tokens_details=[None],
        )
        cold = report.project_cache_reports(raw)
        self.assertEqual(cold["total_cached_tokens_host"], 0)
        self.assertEqual(
            cold["cache_report_metadata"]["total_cached_tokens_host"]["status"],
            "measured",
        )
        raw["server_cached_tokens"] = [50]
        self.assertIsNone(report.project_cache_reports(raw)["host_cached_tokens"])
        unknown = report.project_cache_reports(
            dict(total_cached_tokens=99, collect_flat_memory_io=True)
        )
        self.assertEqual(unknown["metrics_mode"], "unknown")
        self.assertTrue(
            all(unknown[name] is None for name in report.REPORT_FIELD_NAMES)
        )
        mode = report.resolve_metrics_mode(
            before=dict(radix_cache_backend="flat_memory"),
            after=dict(flat_memory_attached=False),
        )
        self.assertEqual(mode["metrics_mode_status"], "mixed")
        self.assertTrue(
            all(
                report.project_cache_reports(raw, mode=mode)[name] is None
                for name in report.REPORT_FIELD_NAMES
            )
        )
        mode = report.resolve_metrics_mode(
            before=dict(radix_cache_backend="flat_memory", flat_memory_attached=False)
        )
        self.assertEqual(mode["metrics_mode"], "native")
        self.assertEqual(
            report.resolve_metrics_mode(before=dict(flat_memory=None))["metrics_mode"],
            "unknown",
        )
        self.assertEqual(
            report.resolve_metrics_mode(
                details=[dict(cache_source_mode="mooncake_tiered_gds")]
            )["metrics_mode"],
            "native",
        )

    def test_flat_weighted_means_and_authoritative_nulls(self):
        raw = dict(
            cached_tokens_details=[
                dict(
                    cache_source_mode="flat",
                    flat_prefetch_dram_ms=20,
                    flat_prefetch_dram_ops=2,
                    flat_prefetch_ssd_ms=0,
                    flat_prefetch_ssd_ops=0,
                ),
                dict(
                    cache_source_mode="flat",
                    flat_prefetch_dram_ms=0,
                    flat_prefetch_dram_ops=0,
                    flat_prefetch_ssd_ms=90,
                    flat_prefetch_ssd_ops=3,
                ),
            ],
            server_prompt_tokens=[100, 100],
            server_cached_tokens=[50, 50],
        )
        projected = report.project_cache_reports(raw)
        self.assertEqual(projected["flat_prefetch_latency_ms"], 22)
        for field in report.REPORT_FIELDS:
            if field["group"] == "cache":
                self.assertIsNone(projected[field["name"]])
        projected["flat_prefetch_latency_ms"] = None
        projected.update(raw)
        again = report.project_cache_reports(projected)
        self.assertIsNone(again["flat_prefetch_latency_ms"])
        self.assertIsNone(again["total_prompt_tokens"])
        self.assertEqual(
            set(report.REPORT_FIELD_NAMES), set(again) & set(report.REPORT_FIELD_NAMES)
        )
        zero = report.project_cache_reports(
            dict(
                cache_source_mode="flat",
                cached_tokens_details=[
                    dict(
                        flat_prefetch_dram_ms=0,
                        flat_prefetch_dram_ops=0,
                        flat_prefetch_ssd_ms=0,
                        flat_prefetch_ssd_ops=0,
                    )
                ],
            )
        )
        self.assertEqual(
            zero["cache_report_metadata"]["flat_prefetch_latency_ms"]["reason"],
            "no_samples",
        )
        self.assertIsNone(zero["flat_prefetch_latency_ms"])
        header = dict(
            cache_report_schema_version=1, metrics_mode="flat", total_cached_tokens=None
        )
        self.assertEqual(
            report.project_cache_reports(header)["cache_report_metadata"][
                "total_cached_tokens"
            ]["status"],
            "not_applicable",
        )
        header.update(cache_report_schema_version=99, flat_prefetch_latency_ms=10)
        future = report.project_cache_reports(header)
        self.assertEqual(future["metrics_mode"], "unknown")
        self.assertTrue(all(future[name] is None for name in report.REPORT_FIELD_NAMES))
        header.update(cache_report_schema_version=1, metrics_mode="future")
        self.assertIsNone(
            report.project_cache_reports(header)["flat_prefetch_latency_ms"]
        )

    def test_cumulative_usage_and_metadata_only_updates(self):
        output = SimpleNamespace(
            success=True,
            server_usage={},
            server_prompt_tokens=None,
            server_completion_tokens=None,
            server_cached_tokens=None,
            cached_tokens_details=None,
            cache_source_mode=None,
        )
        meta = dict(
            prompt_tokens=100,
            cached_tokens=25,
            cached_tokens_details=dict(
                device=25,
                host=0,
                storage=0,
                cache_source_mode="native",
                future="preserved",
            ),
        )
        consume_native_usage(output, meta)
        consume_native_usage(output, meta)
        consume_native_usage(output, dict(completion_tokens=7))
        self.assertEqual(summarize_cache_usage([output])["total_cached_tokens"], 25)
        self.assertEqual(output.server_completion_tokens, 7)
        self.assertEqual(output.cached_tokens_details["future"], "preserved")
        self.assertEqual(output.cache_source_mode, "native")
        output.cached_tokens_details["future"] = "changed"
        self.assertEqual(meta["cached_tokens_details"]["future"], "preserved")

    def test_completed_windows_aligned_peaks_and_invalid_evidence(self):
        before, after = window_fixture(True), window_fixture(False)
        values = summarize_flat_io(before, after, "owned", 4)
        self.assertEqual(values["flat_dram_read_bw_gbps"], 2)
        self.assertEqual(values["flat_dram_read_peak_bw_gbps"], 3)
        self.assertEqual(values["flat_ssd_new_kv_blocks"], 10)
        raw = dict(
            cache_source_mode="flat",
            flat_io_status="ok",
            flat_io_metadata=dict(before=before, after=after),
        )
        self.assertEqual(report.project_cache_reports(raw)["flat_ssd_write_bw_gbps"], 2)
        after["ranks"][0]["io_window"]["overflowed"] = True
        self.assertIsNone(report.project_cache_reports(raw)["flat_ssd_write_bw_gbps"])
        with self.assertRaises(ValueError):
            summarize_flat_io(before, after, "owned", 4)
        after["ranks"][0]["io_window"]["overflowed"] = False
        for snapshot in (before, after):
            for rank in snapshot["ranks"]:
                rank["io_window"]["schema_version"] = 1
        self.assertIsNone(
            summarize_flat_io(before, after, "owned", 4)["flat_dram_read_bw_gbps"]
        )
        after["ranks"].pop()
        with self.assertRaises(ValueError):
            summarize_flat_io(before, after, "owned", 4)

    def test_collector_mode_gate_failure_cleanup_and_file_loader(self):
        async def exercise():
            native = FlatMemoryIOWindow(
                "http://unused",
                True,
                mode=report.resolve_metrics_mode(
                    before=dict(enable_hierarchical_cache=True)
                ),
            )
            with patch.object(native, "_request", new=AsyncMock()) as request:
                async with native:
                    pass
                request.assert_not_awaited()
            flat = FlatMemoryIOWindow(
                "http://unused",
                True,
                mode=report.resolve_metrics_mode(
                    before=dict(radix_cache_backend="flat_memory")
                ),
            )
            with patch.object(flat, "_wait_idle", new=AsyncMock()), patch.object(
                flat,
                "_request",
                new=AsyncMock(side_effect=[TimeoutError("lost begin"), {}]),
            ) as request, self.assertWarns(UserWarning):
                async with flat:
                    completed = True
                self.assertTrue(completed)
                self.assertEqual(
                    [call.args[0] for call in request.await_args_list],
                    ["begin", "abort"],
                )
                self.assertEqual(flat.result["flat_io_status"], "unavailable")
            info = dict(
                tp_size=4,
                api_key="secret",
                internal_states=[
                    dict(flat_memory=window_fixture(True), avg_spec_accept_length=2.5)
                ],
            )
            with patch.object(flat, "_request", new=AsyncMock(return_value=info)):
                await flat._wait_idle(before=True)
            self.assertNotIn("secret", repr(flat.result))
            self.assertNotIn("api_key", flat.server_info_before)
            self.assertEqual(flat.server_info_before["cache_source_mode"], "flat")
            self.assertEqual(
                flat.server_info_before["internal_states"][0]["avg_spec_accept_length"],
                2.5,
            )

        asyncio.run(exercise())
        spec = importlib.util.spec_from_file_location(
            "standalone_report", Path(report.__file__)
        )
        standalone = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(standalone)
        text = standalone.format_flat_memory_report({})
        self.assertIn("Cache Hit Statistics", text)
        self.assertIn("Flat Completed/Durable I/O Window", text)
        self.assertNotIn(": None", text)
        self.assertNotIn(": 0", text)


if __name__ == "__main__":
    unittest.main()
