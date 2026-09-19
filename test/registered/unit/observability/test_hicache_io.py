"""Readonly snapshots must not poll events or consume ACKs from the native path."""

import ast
import copy
import importlib.util
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_SRT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def _load_module(relative):
    spec = importlib.util.spec_from_file_location(
        "hicache_test_module", _SRT / relative
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_methods(relative, class_name, names, namespace=None):
    tree = ast.parse((_SRT / relative).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    scope = {} if namespace is None else dict(namespace)
    exec(
        compile(ast.fix_missing_locations(module), str(_SRT / relative), "exec"), scope
    )
    return type(class_name, (), {name: scope[name] for name in names})


io = _load_module("observability/hicache_io.py")
HostIOMetrics = _load_module("observability/hicache_io_metrics.py").HostIOMetrics
UnifiedRadixCache = _load_methods(
    "mem_cache/unified_radix_cache.py",
    "UnifiedRadixCache",
    (
        "writing_check",
        "loading_check",
        "_log_write_ack_metrics",
        "reset",
        "_reset_full",
    ),
)
MooncakeStore = _load_methods(
    "mem_cache/storage/mooncake_store/mooncake_store.py",
    "MooncakeStore",
    ("get_io_stats_snapshot",),
)


def ack(*, timed=True, num_bytes=4096, elapsed=2.125001):
    return SimpleNamespace(
        node_ids=[1],
        start_event=Mock(elapsed_time=Mock(return_value=elapsed)),
        finish_event=Mock(),
        timing_enabled=timed,
        num_bytes=num_bytes,
        num_tokens_by_pool={},
    )


def cache_stub():
    cache = UnifiedRadixCache()
    cache.hicache_io_counters = io.HiCacheIOCounters()
    cache.host_memory_mode = "cache"
    cache.cache_controller = SimpleNamespace(
        load_queue=[],
        ack_load_queue=[],
        write_queue=[],
        ack_write_queue=[],
        enable_storage=False,
        storage_backend=None,
        reset=Mock(),
        mem_pool_host=Mock(),
        host_io_snapshot=Mock(side_effect=AssertionError("native snapshot")),
        collect_host_io_metrics=Mock(side_effect=AssertionError("native collect")),
    )
    cache.ongoing_load_back = {}
    cache.ongoing_write_through = {}
    cache.ongoing_prefetch = {}
    cache.ongoing_backup = {}
    cache.buffer_pipeline = None
    cache.metrics_collector = None
    cache.dec_lock_ref = Mock()
    cache.dec_host_lock_ref = Mock()
    cache.tree_core = Mock()
    cache._finish_write_through_ack = Mock(
        side_effect=lambda node_id: cache.ongoing_write_through.pop(node_id)
    )
    cache.linker = None
    cache.session = SimpleNamespace(slots={})
    cache.session_refs = Mock()
    cache.check_hicache_events = Mock(side_effect=AssertionError("event advancement"))
    return cache


class TestHiCacheIOAccounting(CustomTestCase):
    def test_normal_ack_accounting_is_once_and_independent_of_prometheus(self):
        for direction in ("read", "write"):
            for metrics in (None, Mock()):
                with self.subTest(direction=direction, metrics=metrics is not None):
                    cache = cache_stub()
                    cache.metrics_collector = metrics
                    item = ack()
                    if direction == "read":
                        cache.cache_controller.ack_load_queue.append(item)
                        cache.ongoing_load_back[1] = (object(), object(), object())
                        consume = cache.loading_check
                    else:
                        cache.cache_controller.ack_write_queue.append(item)
                        cache.ongoing_write_through[1] = object()
                        consume = cache.writing_check
                    before = io.snapshot_unified_hicache(cache)
                    consume(finish_count=0)
                    self.assertEqual(io.snapshot_unified_hicache(cache), before)
                    item.finish_event.synchronize.assert_not_called()
                    item.start_event.elapsed_time.assert_not_called()
                    consume(finish_count=1)
                    after = io.snapshot_unified_hicache(cache)
                    self.assertEqual(
                        after["l2"][direction],
                        dict(
                            bytes=4096,
                            duration_ns=2125001,
                            batches=1,
                            untimed_batches=0,
                        ),
                    )
                    self.assertEqual(
                        after["pending"]["h2d" if direction == "read" else "d2h"], 0
                    )
                    consume(finish_count=0)
                    self.assertEqual(io.snapshot_unified_hicache(cache), after)
                    item.start_event.elapsed_time.assert_called_once_with(
                        item.finish_event
                    )
                    item.finish_event.synchronize.assert_called_once()

    def test_blocking_writeback_is_accounted_and_flush_keeps_epoch(self):
        cache = cache_stub()
        cache.ongoing_write_through[1] = object()
        item = ack()
        cache.cache_controller.ack_write_queue.append(item)
        cache.writing_check(write_back=True)
        cache.writing_check(write_back=True)
        before = cache.hicache_io_counters.snapshot()
        self.assertEqual(before["l2"]["write"]["batches"], 1)
        cache.reset()
        self.assertEqual(cache.hicache_io_counters.snapshot(), before)
        self.assertNotEqual(io.HiCacheIOCounters().epoch, before["epoch"])
        item.start_event.elapsed_time.assert_called_once()

    def test_missing_and_invalid_timing_never_become_zero_time_bandwidth(self):
        for timed, elapsed in (
            (False, 2),
            (True, 0),
            (True, float("nan")),
            (True, -1),
            (True, RuntimeError("unsupported")),
        ):
            with self.subTest(timed=timed, elapsed=elapsed):
                counters = io.HiCacheIOCounters()
                item = ack(timed=timed, elapsed=elapsed)
                if isinstance(elapsed, Exception):
                    item.start_event.elapsed_time.side_effect = elapsed
                counters.account_completed("read", item)
                self.assertEqual(
                    counters.snapshot()["l2"]["read"],
                    dict(bytes=4096, duration_ns=0, batches=1, untimed_batches=1),
                )
                item.finish_event.query.assert_not_called()
                item.finish_event.synchronize.assert_not_called()

    def test_snapshot_never_advances_even_ready_events_or_native_exporter(self):
        cache = cache_stub()
        item = ack()
        item.finish_event.query.side_effect = AssertionError("queried")
        item.finish_event.synchronize.side_effect = AssertionError("synchronized")
        cache.cache_controller.ack_load_queue = [item]
        cache.cache_controller.ack_write_queue = [item, item]
        cache.cache_controller.load_queue = [object()]
        cache.cache_controller.write_queue = [object()]
        cache.ongoing_prefetch = {"a": object(), "b": object()}
        cache.ongoing_backup = {1: object()}
        first = io.snapshot_unified_hicache(cache)
        second = io.snapshot_unified_hicache(cache)
        self.assertEqual(first, second)
        self.assertEqual(
            first["pending"],
            dict(h2d=2, d2h=3, storage_prefetch=2, storage_backup=1),
        )
        first["l2"]["read"]["bytes"] = 999
        self.assertEqual(io.snapshot_unified_hicache(cache), second)
        item.start_event.elapsed_time.assert_not_called()
        cache.cache_controller.reset.assert_not_called()

    def test_unsupported_payload_is_explicit_in_both_exporters(self):
        counters = io.HiCacheIOCounters()
        item = ack(num_bytes=None)
        counters.account_completed("read", item)
        result = counters.snapshot()
        self.assertEqual(result["status"], "unsupported")
        self.assertNotIn("l2", result)
        native = HostIOMetrics(enabled=True)
        native.record(direction="read", completion=item, num_bytes=None)
        self.assertFalse(native.snapshot()["enabled"])
        self.assertIn("unsupported payload", native.error)

    def test_default_native_exporter_still_collects_and_resets(self):
        native = HostIOMetrics(enabled=True)
        item = ack()
        item.finish_event.query.return_value = False
        native.record(direction="read", completion=item, num_bytes=4096)
        self.assertEqual(native.snapshot()["read"]["bytes"], 0)
        item.finish_event.query.return_value = True
        self.assertEqual(
            native.snapshot()["read"], dict(bytes=4096, batches=1, elapsed_ms=2.125001)
        )
        native.reset()
        self.assertEqual(native.snapshot()["generation"], 1)
        self.assertEqual(native.snapshot()["read"]["bytes"], 0)


class TestHiCacheIOCollection(CustomTestCase):
    def test_old_or_failing_native_client_does_not_erase_l2(self):
        for store in (
            SimpleNamespace(),
            SimpleNamespace(
                get_io_stats_snapshot=Mock(side_effect=RuntimeError("native error"))
            ),
        ):
            with self.subTest(store=store):
                backend = MooncakeStore()
                backend.store = store
                backend.get_storage_io_snapshot = Mock(
                    side_effect=AssertionError("native window fallback")
                )
                cache = cache_stub()
                cache.cache_controller.enable_storage = True
                cache.cache_controller.storage_backend = backend
                module = SimpleNamespace(MooncakeStore=MooncakeStore)
                with patch.dict(
                    "sys.modules",
                    {
                        "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store": module
                    },
                ):
                    result = io.snapshot_unified_hicache(cache)
                self.assertEqual(result["status"], "ok")
                self.assertIn("l2", result)
                self.assertEqual(result["mooncake"]["status"], "unavailable")
                backend.get_storage_io_snapshot.assert_not_called()

    def test_optional_native_snapshot_is_returned_without_reset_or_conversion(self):
        expected = {
            "schema_version": 1,
            "instance_id": "native-epoch",
            "capabilities": ["ssd_to_host_fetch_v1"],
            "ssd_to_host_fetch": {
                "bytes": 4096,
                "latency_ns_sum": 1234567,
                "batches": 1,
                "errors": 0,
                "inflight": 2,
            },
        }
        backend = MooncakeStore()
        backend.store = SimpleNamespace(
            get_io_stats_snapshot=lambda: copy.deepcopy(expected)
        )
        self.assertEqual(backend.get_io_stats_snapshot(), expected)
        self.assertEqual(backend.get_io_stats_snapshot(), expected)
        del backend.store
        with self.assertRaises(AttributeError):
            backend.get_io_stats_snapshot()

    def test_local_snapshot_failure_still_gathers_every_tp_rank(self):
        cache = cache_stub()
        del cache.cache_controller.load_queue
        group = object()

        def gather(output, local, *, group):
            output[:] = [
                dict(tp_rank=i, pp_rank=0, dp_rank=0, status="ok") for i in range(8)
            ]
            output[3] = local
            output.reverse()

        dist = ModuleType("torch.distributed")
        dist.all_gather_object = Mock(side_effect=gather)
        torch = ModuleType("torch")
        torch.distributed = dist
        with patch.dict(
            "sys.modules",
            {
                "torch": torch,
                "torch.distributed": dist,
                "sglang.srt.mem_cache.unified_radix_cache": SimpleNamespace(
                    UnifiedRadixCache=UnifiedRadixCache
                ),
            },
        ):
            result = io.collect_hicache_io(
                cache=cache,
                tp_rank=3,
                tp_size=8,
                pp_rank=0,
                pp_size=1,
                dp_rank=None,
                dp_size=1,
                attn_cp_size=1,
                attn_dcp_size=1,
                tp_cpu_group=group,
            )
        dist.all_gather_object.assert_called_once()
        self.assertIs(dist.all_gather_object.call_args.kwargs["group"], group)
        self.assertEqual([rank["tp_rank"] for rank in result["ranks"]], list(range(8)))
        self.assertEqual(result["ranks"][3]["status"], "error")
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["scope"], "completed_accounted_window")
        self.assertFalse(result["drain"])

    def test_unsupported_topology_returns_before_any_collective_or_snapshot(self):
        for field in ("pp_size", "dp_size", "attn_cp_size", "attn_dcp_size"):
            args = dict(
                cache=object(),
                tp_rank=0,
                tp_size=8,
                pp_rank=0,
                pp_size=1,
                dp_rank=0,
                dp_size=1,
                attn_cp_size=1,
                attn_dcp_size=1,
                tp_cpu_group=object(),
            )
            args[field] = 2
            result = io.collect_hicache_io(**args)
            self.assertEqual(result["status"], "unsupported")
            self.assertEqual(result["ranks"], [])


if __name__ == "__main__":
    unittest.main()
