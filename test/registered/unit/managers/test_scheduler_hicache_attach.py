"""Runtime HiCache attach/detach lands on the config bags.

The attach RPC used to mutate the scheduler's ServerArgs so the readback would
show the change; the namespace readers never saw it. Both now go through
get_context().override, so get_memory() and the resolved-config readback agree
and the published instance stays as the launcher left it.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.io_struct import (
    AttachHiCacheStorageReqInput,
    DetachHiCacheStorageReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_memory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerHiCacheAttach(CustomTestCase):
    def _scheduler(self, **fields):
        override = get_context().override_server_args(
            enable_hierarchical_cache=True, **fields
        )
        self.server_args = override.install()
        self.addCleanup(override.restore)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = self.server_args
        scheduler.enable_hierarchical_cache = True
        scheduler.enable_hicache_storage = False
        scheduler.is_fully_idle = lambda: True
        scheduler.tree_cache = SimpleNamespace(
            attach_storage_backend=lambda **kwargs: (True, "attached"),
            detach_storage_backend=lambda: (True, "detached"),
        )
        return scheduler

    def test_pressure_retry_works_without_miss_retry_polling(self):
        scheduler = self._scheduler(hicache_storage_prefetch_retry_poll_interval=0)
        scheduler.enable_hicache_storage = True
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache._storage_prefetch_deferred_rids = {"r"}
        scheduler.tree_cache = cache
        scheduler._prefetch_kvcache = MagicMock()
        req = SimpleNamespace(rid="r", storage_prefetch_retry_attempts=0)

        scheduler._retry_deferred_storage_prefetch(req)
        scheduler._retry_deferred_storage_prefetch(req)

        scheduler._prefetch_kvcache.assert_called_once_with(req)
        self.assertEqual(req.storage_prefetch_retry_attempts, 0)
        self.assertEqual(get_memory().hicache_storage_prefetch_retry_poll_interval, 0)

    def test_rejected_pressure_retry_does_not_spend_query_attempts(self):
        scheduler = self._scheduler(hicache_storage_prefetch_retry_poll_interval=0)
        scheduler.enable_hicache_storage = True
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache._storage_prefetch_deferred_rids = {"r"}
        scheduler.tree_cache = cache
        scheduler._prefetch_kvcache = MagicMock(
            side_effect=lambda req: cache._storage_prefetch_deferred_rids.add(req.rid)
        )
        req = SimpleNamespace(rid="r", storage_prefetch_retry_attempts=2)

        scheduler._retry_deferred_storage_prefetch(req)

        self.assertTrue(cache.pop_storage_prefetch_deferred("r"))
        self.assertEqual(req.storage_prefetch_retry_attempts, 2)

    def test_no_pressure_retry_when_storage_is_disabled(self):
        scheduler = self._scheduler()
        scheduler._prefetch_kvcache = MagicMock()
        scheduler._retry_deferred_storage_prefetch(SimpleNamespace(rid="r"))
        scheduler._prefetch_kvcache.assert_not_called()

    def test_attach_reaches_the_namespace_readers(self):
        scheduler = self._scheduler(hicache_storage_backend=None)
        out = scheduler.attach_hicache_storage_wrapped(
            AttachHiCacheStorageReqInput(
                hicache_storage_backend="file",
                hicache_write_policy="write_through",
            )
        )

        self.assertTrue(out.success)
        self.assertEqual(get_memory().hicache_storage_backend, "file")
        self.assertEqual(get_memory().hicache_write_policy, "write_through")
        self.assertEqual(
            get_context().resolved_server_args_dict()["hicache_storage_backend"],
            "file",
        )
        self.assertIsNone(self.server_args.hicache_storage_backend)

    def test_detach_clears_the_backend_for_the_same_readers(self):
        scheduler = self._scheduler(hicache_storage_backend="file")
        scheduler.enable_hicache_storage = True

        out = scheduler.detach_hicache_storage_wrapped(DetachHiCacheStorageReqInput())

        self.assertTrue(out.success)
        self.assertIsNone(get_memory().hicache_storage_backend)
        self.assertIsNone(
            get_context().resolved_server_args_dict()["hicache_storage_backend"]
        )
        # The record is not written any more: the attach is a declaration on
        # it and the detach is a bag override (asserted above), so the two are
        # meant to differ here.
        self.assertEqual(
            resolution_result(self.server_args, "hicache_storage_backend"), "file"
        )

    def _prefill_case(
        self, *, tiered=True, mixed=False, preempt=False, miss_interval=0
    ):
        scheduler = self._scheduler(
            hicache_storage_prefetch_retry_poll_interval=miss_interval
        )
        events = []

        def request(rid):
            req = MagicMock()
            req.rid, req.priority, req.beam_group = rid, None, None
            req.inflight_middle_chunks = 0
            req.storage_prefetch_retry_pending = False
            req.storage_prefetch_retry_wait_polls = (
                req.storage_prefetch_retry_attempts
            ) = 0
            req.extend_range = SimpleNamespace(length=4)
            req.kv = SimpleNamespace(holds_mamba=False)
            return req

        chunk, waiting, displaced = (
            request(rid) for rid in ("chunk", "waiting", "displaced")
        )
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tiered_runtime = SimpleNamespace(pending={}) if tiered else None
        cache._storage_prefetch_deferred_rids = {waiting.rid}
        cache._storage_prefetch_missed_rids = set()
        cache.check_hicache_events = MagicMock(
            side_effect=lambda: events.append("poll")
        )
        cache.check_prefetch_progress = MagicMock(
            side_effect=lambda rid: events.append("per-request-poll") or True
        )
        cache.pop_prefetch_loaded_tokens = MagicMock(return_value=0)
        cache.pop_tiered_cache_sources = MagicMock(return_value=[])
        cache.pop_prefetch_latency = MagicMock(return_value=(0.0, 0))
        cache.release_prefetch_hold = MagicMock()
        cache.ready_to_load_host_cache = MagicMock(return_value=0)
        cache.cache_controller = SimpleNamespace(
            storage_backend=SimpleNamespace(gpu_page_bytes=1)
        )
        scheduler.tree_cache = cache
        scheduler._prefetch_kvcache = MagicMock(
            side_effect=lambda req: events.append("retry:" + req.rid)
        )
        scheduler.enable_hicache_storage = True
        scheduler.waiting_queue, scheduler.chunked_req = [waiting], chunk
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.enable_priority_preemption = preempt
        scheduler.is_hybrid_swa = True
        scheduler.is_mixed_chunk = mixed
        scheduler.enable_lora = scheduler.enable_dynamic_chunking = False
        scheduler.enable_priority_scheduling = False
        scheduler.min_free_slots_delayer = scheduler.dllm_config = None
        scheduler.max_queued_requests = None
        scheduler.get_num_allocatable_reqs = MagicMock(return_value=8)
        scheduler.policy = MagicMock()
        scheduler.chunked_prefill_size, scheduler.page_size = 4, 1
        scheduler.max_prefill_tokens = 16
        scheduler.max_prefill_bs = scheduler.max_running_requests = 8
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        scheduler.req_to_token_pool = SimpleNamespace()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(extend_attention_block_m=64),
                prefill_aware_swa=False,
            )
        )
        scheduler.disaggregation_mode = scheduler_module.DisaggregationMode.NULL
        scheduler.truncation_align_size = None
        scheduler.model_config, scheduler.spec_algorithm = MagicMock(), MagicMock()
        scheduler.enable_overlap = True
        scheduler._slow_stage = lambda *args, **kwargs: nullcontext()
        scheduler.load_inquirer = MagicMock()
        scheduler.load_inquirer._get_num_pending_tokens.return_value = 0
        running = MagicMock(
            reqs=[request("decode")] if mixed else [],
            batch_is_full=False,
            return_logprob=False,
        )
        running.is_empty.side_effect = lambda: not running.reqs
        running.prepare_for_decode.side_effect = lambda: events.append("decode-alloc")
        new_batch = MagicMock(return_logprob=False, input_embeds=None)
        new_batch.prepare_for_extend.side_effect = lambda: events.append("extend-alloc")
        rejected = set()
        preemptions = [displaced] if preempt else []

        def make_adder(*args, **kwargs):
            events.append("adder")
            adder = SimpleNamespace(
                can_run_list=[], preempt_list=list(preemptions), new_chunked_req=None
            )
            preemptions.clear()

            def add_chunk(req):
                events.append("admit:" + req.rid)
                adder.can_run_list.append(req)
                return req

            def add_request(req, **kwargs):
                events.append("admit:" + req.rid)
                if req.rid in rejected:
                    return scheduler_module.AddReqResult.NO_TOKEN
                adder.can_run_list.append(req)
                return scheduler_module.AddReqResult.CONTINUE

            adder.add_chunked_req, adder.add_one_req = add_chunk, add_request
            return adder

        def drive():
            with (
                patch.object(scheduler_module, "PrefillAdder", side_effect=make_adder),
                patch.object(
                    scheduler_module.ScheduleBatch, "init_new", return_value=new_batch
                ),
                patch.object(
                    scheduler_module.PrefillStats, "from_adder", return_value=None
                ),
            ):
                return scheduler._get_new_batch_prefill_raw(None, running)

        return SimpleNamespace(
            scheduler=scheduler,
            cache=cache,
            events=events,
            drive=drive,
            waiting=waiting,
            displaced=displaced,
            rejected=rejected,
            request=request,
        )

    def test_tiered_progress_precedes_admission_and_mixed_allocation(self):
        """GDS cannot take reclaimable capacity after a chunk has been admitted."""
        for preempt in (False, True):
            with self.subTest(preempt=preempt):
                case = self._prefill_case(mixed=True, preempt=preempt)
                case.drive()
                adder = case.events.index("adder")
                mutations = [
                    i
                    for i, event in enumerate(case.events)
                    if event == "poll" or event.startswith("retry:")
                ]
                self.assertTrue(mutations)
                self.assertTrue(all(i < adder for i in mutations), case.events)
                case.cache.check_prefetch_progress.assert_not_called()
                self.assertLess(
                    case.events.index("extend-alloc"), case.events.index("decode-alloc")
                )
                if preempt:
                    self.assertIn(case.displaced, case.scheduler.waiting_queue)
                    self.assertIn(
                        case.displaced.rid, case.cache._storage_prefetch_deferred_rids
                    )
                    case.displaced.time_stats.set_wait_queue_entry_time.assert_called_once()
                    case.events.clear()
                    case.drive()
                    self.assertLess(
                        case.events.index("retry:displaced"), case.events.index("adder")
                    )

    def test_ordinary_storage_keeps_per_request_progress(self):
        case = self._prefill_case(tiered=False)
        case.drive()
        case.cache.check_prefetch_progress.assert_called_once_with(case.waiting.rid)
        self.assertGreater(
            case.events.index("retry:waiting"), case.events.index("adder")
        )

    def test_pressure_retry_is_not_repeated_by_miss_retry(self):
        case = self._prefill_case(miss_interval=1)
        case.waiting.storage_prefetch_retry_pending = True
        case.waiting.storage_prefetch_retry_wait_polls = 1
        case.drive()
        case.scheduler._prefetch_kvcache.assert_called_once_with(case.waiting)
        self.assertEqual(case.waiting.storage_prefetch_retry_attempts, 0)

    def test_tiered_early_return_still_advances_storage(self):
        case = self._prefill_case()
        case.scheduler.waiting_queue = []
        case.scheduler.chunked_req = None
        case.drive()
        case.cache.check_hicache_events.assert_called_once()
        self.assertNotIn("adder", case.events)

    def test_new_deferrals_wait_for_next_prefill_pass(self):
        """A newly released anchor must not be repinned by the same retry sweep."""
        for phase in ("grammar", "poll", "retry"):
            with self.subTest(phase=phase):
                case = self._prefill_case()
                case.cache._storage_prefetch_deferred_rids.clear()
                case.rejected.add(case.waiting.rid)

                def defer():
                    case.cache._storage_prefetch_deferred_rids.add(case.waiting.rid)

                if phase == "grammar":
                    case.scheduler.grammar_manager.has_waiting_grammars.return_value = (
                        True
                    )
                    case.scheduler.grammar_manager.get_ready_grammar_requests.side_effect = lambda: (
                        defer() or []
                    )
                elif phase == "poll":
                    case.cache.check_hicache_events.side_effect = lambda: (
                        case.events.append("poll"),
                        defer(),
                    )
                else:
                    earlier = case.request("earlier")
                    case.scheduler.waiting_queue.insert(0, earlier)
                    case.cache._storage_prefetch_deferred_rids.add(earlier.rid)
                    case.scheduler._prefetch_kvcache.side_effect = lambda req: (
                        case.events.append("retry:" + req.rid),
                        defer(),
                    )
                case.drive()
                if phase == "retry":
                    case.scheduler._prefetch_kvcache.assert_called_once_with(earlier)
                else:
                    case.scheduler._prefetch_kvcache.assert_not_called()
                self.assertIn(
                    case.waiting.rid, case.cache._storage_prefetch_deferred_rids
                )
                case.scheduler.grammar_manager.has_waiting_grammars.return_value = False
                case.cache.check_hicache_events.side_effect = (
                    lambda: case.events.append("poll")
                )
                case.scheduler._prefetch_kvcache.reset_mock()
                case.scheduler._prefetch_kvcache.side_effect = (
                    lambda req: case.events.append("retry:" + req.rid)
                )
                case.events.clear()
                case.drive()
                case.scheduler._prefetch_kvcache.assert_called_once_with(case.waiting)
                self.assertLess(
                    case.events.index("retry:waiting"), case.events.index("adder")
                )


if __name__ == "__main__":
    unittest.main()
