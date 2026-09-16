"""Completion-only accounting must not turn server_info into an I/O progress hook."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.hicache_io import (
    HiCacheIOCounters,
    collect_hicache_io,
    gather_hicache_io,
    snapshot_unified_hicache,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def ack(*, node_id=1, ready=False, timed=True, num_bytes=4096, duration_ms=2.125001):
    finish = Mock()
    finish.query.return_value = ready
    start = Mock()
    start.elapsed_time.return_value = duration_ms
    return HiCacheAck(
        start_event=start,
        finish_event=finish,
        node_ids=[node_id],
        num_bytes=num_bytes,
        timing_enabled=timed,
    )


def cache_stub():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.hicache_io_counters = HiCacheIOCounters()
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
    )
    cache.ongoing_load_back = {}
    cache.ongoing_write_through = {}
    cache.ongoing_prefetch = {}
    cache.ongoing_backup = {}
    cache.buffer_pipeline = None
    cache.enable_storage_metrics = False
    cache.metrics_collector = None
    cache.pp_rank = 0
    cache._all_reduce = Mock()
    cache.dec_lock_ref = Mock()
    cache.dec_host_lock_ref = Mock()
    cache.tree_core = Mock(write_back_duplicate_reclaim_digest=0)
    cache._finish_write_through_ack = Mock(
        side_effect=lambda node_id: cache.ongoing_write_through.pop(node_id)
    )
    cache.linker = None
    cache.session = SimpleNamespace(slots={})
    cache.session_refs = Mock()
    return cache


def native_snapshot():
    return {
        "schema_version": 1,
        "instance_id": "native-client-epoch",
        "capabilities": ["ssd_to_host_fetch_v1"],
        "ssd_to_host_fetch": {
            "bytes": 4096,
            "latency_ns_sum": 1234567,
            "batches": 1,
            "errors": 0,
            "inflight": 2,
        },
    }


class TestHiCacheIOAccounting(CustomTestCase):
    def test_read_and_write_only_count_normally_consumed_acks_once(self):
        for direction in ("read", "write"):
            for metrics_enabled in (False, True):
                with self.subTest(direction=direction, metrics=metrics_enabled):
                    cache = cache_stub()
                    cache.metrics_collector = Mock() if metrics_enabled else None
                    item = ack()
                    if direction == "read":
                        cache.cache_controller.ack_load_queue.append(item)
                        cache.ongoing_load_back[1] = (object(), object(), object())
                        consume = cache.loading_check
                    else:
                        cache.cache_controller.ack_write_queue.append(item)
                        cache.ongoing_write_through[1] = object()
                        consume = cache.writing_check
                    before = snapshot_unified_hicache(cache)
                    consume()
                    self.assertEqual(snapshot_unified_hicache(cache), before)
                    item.finish_event.synchronize.assert_not_called()
                    item.start_event.elapsed_time.assert_not_called()
                    # A ready event is still outside the accounted window until consumed.
                    item.finish_event.query.return_value = True
                    self.assertEqual(snapshot_unified_hicache(cache), before)
                    consume()
                    after = snapshot_unified_hicache(cache)
                    self.assertEqual(
                        after["l2"][direction],
                        {
                            "bytes": 4096,
                            "duration_ns": 2125001,
                            "batches": 1,
                            "untimed_batches": 0,
                        },
                    )
                    self.assertEqual(
                        after["pending"]["h2d" if direction == "read" else "d2h"], 0
                    )
                    consume()
                    self.assertEqual(snapshot_unified_hicache(cache), after)
                    item.start_event.elapsed_time.assert_called_once_with(
                        item.finish_event
                    )
                    item.finish_event.synchronize.assert_called_once()

    def test_write_back_normal_blocking_path_accounts_once(self):
        cache = cache_stub()
        cache.ongoing_write_through[1] = object()
        item = ack(ready=True)
        cache.cache_controller.ack_write_queue.append(item)
        cache.writing_check(write_back=True)
        cache.writing_check(write_back=True)
        self.assertEqual(
            cache.hicache_io_counters.snapshot()["l2"]["write"]["batches"], 1
        )
        item.start_event.elapsed_time.assert_called_once()

    def test_unknown_and_invalid_timing_are_explicitly_untimed(self):
        for timed, elapsed in (
            (False, 2),
            (True, 0),
            (True, float("nan")),
            (True, -1),
            (True, RuntimeError("timing unsupported")),
        ):
            with self.subTest(timed=timed, elapsed=elapsed):
                counters = HiCacheIOCounters()
                item = ack(timed=timed, ready=True, duration_ms=elapsed)
                if isinstance(elapsed, Exception):
                    item.start_event.elapsed_time.side_effect = elapsed
                counters.account_completed("read", item)
                self.assertEqual(
                    counters.snapshot()["l2"]["read"],
                    {
                        "bytes": 4096,
                        "duration_ns": 0,
                        "batches": 1,
                        "untimed_batches": 1,
                    },
                )
                item.finish_event.query.assert_not_called()
                item.finish_event.synchronize.assert_not_called()
                if not timed:
                    item.start_event.elapsed_time.assert_not_called()

    def test_snapshot_never_polls_advances_flushes_or_resets(self):
        cache = cache_stub()
        item = ack(ready=True)
        item.finish_event.query.side_effect = AssertionError(
            "snapshot queried a GPU event"
        )
        item.finish_event.synchronize.side_effect = AssertionError(
            "snapshot synchronized"
        )
        for name in (
            "check_hicache_events",
            "_finish_write_through_ack",
            "writing_check",
            "loading_check",
            "reset",
            "flush",
            "drain",
        ):
            setattr(cache, name, Mock(side_effect=AssertionError(name)))
        cache.cache_controller.ack_load_queue = [item]
        cache.cache_controller.ack_write_queue = [item, item]
        cache.cache_controller.load_queue = [object()]
        cache.cache_controller.write_queue = [object()]
        cache.ongoing_prefetch = {"a": object(), "b": object()}
        cache.ongoing_backup = {1: object()}
        first = snapshot_unified_hicache(cache)
        second = snapshot_unified_hicache(cache)
        self.assertEqual(first, second)
        self.assertEqual(
            first["pending"],
            {"h2d": 2, "d2h": 3, "storage_prefetch": 2, "storage_backup": 1},
        )
        first["l2"]["read"]["bytes"] = 999
        self.assertEqual(snapshot_unified_hicache(cache), second)
        item.start_event.elapsed_time.assert_not_called()
        cache.cache_controller.reset.assert_not_called()

    def test_epoch_and_totals_survive_normal_tree_reset(self):
        cache = cache_stub()
        cache.hicache_io_counters.account_completed("read", ack(ready=True))
        before = cache.hicache_io_counters.snapshot()
        cache.reset()
        self.assertEqual(cache.hicache_io_counters.snapshot(), before)
        self.assertNotEqual(HiCacheIOCounters().epoch, before["epoch"])

    def test_unsupported_payload_is_not_exported_as_zero(self):
        counters = HiCacheIOCounters()
        counters.account_completed("read", ack(ready=True, num_bytes=None))
        result = counters.snapshot()
        self.assertEqual(result["status"], "unsupported")
        self.assertNotIn("l2", result)


class TestMooncakeSnapshot(CustomTestCase):
    def test_old_native_api_is_unavailable_without_reset_fallback(self):
        backend = MooncakeStore.__new__(MooncakeStore)
        legacy = Mock(side_effect=AssertionError("legacy reset API called"))
        backend.store = SimpleNamespace(get_and_reset_io_stats=legacy)
        backend.get_stats = Mock(side_effect=AssertionError("get_stats called"))
        result = backend.get_io_stats_snapshot()
        self.assertEqual(result["status"], "unavailable")
        legacy.assert_not_called()
        backend.get_stats.assert_not_called()

    def test_native_snapshot_keeps_ns_epoch_and_capability(self):
        backend = MooncakeStore.__new__(MooncakeStore)
        expected = native_snapshot()
        backend.store = SimpleNamespace(
            get_io_stats_snapshot=Mock(side_effect=lambda: copy.deepcopy(expected))
        )
        backend.get_stats = Mock(side_effect=AssertionError("get_stats called"))
        cache = cache_stub()
        cache.cache_controller.storage_backend = backend
        cache.cache_controller.enable_storage = True
        self.assertEqual(snapshot_unified_hicache(cache)["mooncake"], expected)
        self.assertEqual(snapshot_unified_hicache(cache)["mooncake"], expected)
        backend.get_stats.assert_not_called()

    def test_missing_internal_store_field_is_not_legacy_api_detection(self):
        backend = MooncakeStore.__new__(MooncakeStore)
        cache = cache_stub()
        cache.cache_controller.storage_backend = backend
        cache.cache_controller.enable_storage = True
        with self.assertRaises(AttributeError):
            snapshot_unified_hicache(cache)

    def test_native_failure_preserves_independent_l2_counters(self):
        backend = MooncakeStore.__new__(MooncakeStore)
        backend.store = SimpleNamespace(
            get_io_stats_snapshot=Mock(side_effect=RuntimeError("native error"))
        )
        cache = cache_stub()
        cache.cache_controller.storage_backend = backend
        cache.cache_controller.enable_storage = True
        result = snapshot_unified_hicache(cache)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mooncake"]["status"], "unavailable")
        self.assertIn("native error", result["mooncake"]["reason"])
        self.assertIn("l2", result)


class TestHiCacheIOGather(CustomTestCase):
    def test_every_rank_gathers_even_when_local_snapshot_fails(self):
        gathered_ranks = []
        for tp_rank in range(8):
            with self.subTest(tp_rank=tp_rank):

                def gather(local):
                    gathered_ranks.append(local)
                    result = [
                        dict(tp_rank=i, pp_rank=0, dp_rank=0, status="ok")
                        for i in range(8)
                    ]
                    result[tp_rank] = local
                    return list(reversed(result))

                result = gather_hicache_io(
                    snapshot=Mock(side_effect=ValueError("local snapshot error")),
                    tp_rank=tp_rank,
                    tp_size=8,
                    pp_rank=0,
                    pp_size=1,
                    dp_rank=None,
                    dp_size=1,
                    gather=gather,
                )
                self.assertEqual(
                    [rank["tp_rank"] for rank in result["ranks"]], list(range(8))
                )
                self.assertEqual(result["ranks"][tp_rank]["status"], "error")
                self.assertNotIn("l2", result["ranks"][tp_rank])
                self.assertEqual(result["scope"], "completed_accounted_window")
                self.assertFalse(result["drain"])
        self.assertEqual(len(gathered_ranks), 8)

    def test_management_collector_gathers_internal_field_errors(self):
        cache = cache_stub()
        del cache.cache_controller.load_queue
        group = object()

        def all_gather(result, rank, *, group):
            result[:] = [
                dict(tp_rank=i, pp_rank=0, dp_rank=0, status="ok") for i in range(8)
            ]
            result[3] = rank

        with patch(
            "torch.distributed.all_gather_object", side_effect=all_gather
        ) as collective:
            result = collect_hicache_io(
                cache=cache,
                tp_rank=3,
                tp_size=8,
                pp_rank=0,
                pp_size=1,
                dp_rank=0,
                dp_size=1,
                attn_cp_size=1,
                attn_dcp_size=1,
                tp_cpu_group=group,
            )
        collective.assert_called_once()
        self.assertIs(collective.call_args.kwargs["group"], group)
        self.assertEqual(result["ranks"][3]["status"], "error")
        self.assertIn("AttributeError", result["ranks"][3]["error"])

    def test_scheduler_internal_state_attaches_snapshot_by_default(self):
        from contextlib import ExitStack

        from sglang.srt.distributed.parallel_state_wrapper import ParallelState
        from sglang.srt.managers import scheduler as scheduler_module
        from sglang.srt.managers.io_struct import GetInternalStateReq

        cache = cache_stub()
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(weight_load_mem_usage=0), graph_memory_usage=0
        )
        scheduler = SimpleNamespace(
            tree_cache=cache,
            ps=ParallelState.trivial(tp_size=8),
            tp_cpu_group=object(),
            metrics_reporter=SimpleNamespace(last_gen_throughput=0),
            draft_worker=None,
            tp_worker=worker,
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: SimpleNamespace(mem_usage=0)
            ),
            startup_available_gpu_memory_gb=0,
            max_total_num_tokens=0,
            swa_tokens_per_layer=None,
            startup_time=0,
            flat_memory_cache=None,
            max_running_requests=1,
            spec_algorithm=SimpleNamespace(
                is_none=lambda: True, is_dspark=lambda: False
            ),
        )

        def all_gather(result, local, *, group):
            result[:] = [dict(local, tp_rank=i) for i in range(8)]

        with ExitStack() as stack:
            collective = stack.enter_context(
                patch("torch.distributed.all_gather_object", side_effect=all_gather)
            )
            stack.enter_context(
                patch.object(
                    scheduler_module,
                    "get_context",
                    return_value=SimpleNamespace(resolved_server_args_dict=lambda: {}),
                )
            )
            stack.enter_context(
                patch.object(
                    scheduler_module,
                    "get_parallel",
                    return_value=SimpleNamespace(
                        enable_dp_attention=False, dp_size=1, tp_size=8, pp_size=1
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    scheduler_module,
                    "get_exec",
                    return_value=SimpleNamespace(
                        moe=SimpleNamespace(elastic_ep_backend=None)
                    ),
                )
            )
            stack.enter_context(
                patch.object(scheduler_module, "build_memory_usage", return_value={})
            )
            stack.enter_context(
                patch.object(scheduler_module, "RECORD_STEP_TIME", False)
            )
            stack.enter_context(
                patch.object(
                    scheduler_module.envs.SGLANG_EXPOSE_OWN_ENV_VARS,
                    "get",
                    return_value=False,
                )
            )
            response = scheduler_module.Scheduler.get_internal_state(
                scheduler, GetInternalStateReq()
            )
        collective.assert_called_once()
        result = response.internal_state["hicache_io"]
        self.assertEqual(result["tp_size"], 8)
        self.assertEqual(len(result["ranks"]), 8)
        self.assertEqual(result["ranks"][0]["epoch"], cache.hicache_io_counters.epoch)
        self.assertEqual(result["ranks"][0]["status"], "ok")
        self.assertEqual(result["scope"], "completed_accounted_window")

    def test_unsupported_parallel_topology_does_not_claim_full_rankset(self):
        with patch("torch.distributed.all_gather_object") as collective:
            result = collect_hicache_io(
                cache=cache_stub(),
                tp_rank=0,
                tp_size=8,
                pp_rank=0,
                pp_size=2,
                dp_rank=0,
                dp_size=1,
                attn_cp_size=1,
                attn_dcp_size=1,
                tp_cpu_group=object(),
            )
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["ranks"], [])
        collective.assert_not_called()


if __name__ == "__main__":
    unittest.main()
