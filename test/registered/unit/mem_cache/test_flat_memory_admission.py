"""Real CPU Full/SWA matching and admission protect restored prefixes."""

import unittest
from array import array
from concurrent.futures import Future
from types import SimpleNamespace as NS

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.flat_memory_cache import FlatMemoryCache
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _Storage:
    def __init__(self):
        self.layer_done_counter = object()
        self.storage_generation = 0
        self.queries = {}
        self.loads = {}
        self.transfers = {}

    def submit_lookup(self, rid, transfers):
        future = Future()
        full = next(t for t in transfers if t.name == PoolName.KV)
        future.set_result(list(range(1, len(full.keys) + 1)))
        self.queries[rid] = future
        return future

    refresh_lookup = submit_lookup

    def get_lookup_generation(self, rid):
        return 0 if rid in self.queries else -1

    def poll_queries(self):
        pass

    def release_lookup(self, rid):
        self.queries.pop(rid, None)

    def load(self, rid, transfers):
        self.loads[rid] = Future()
        self.transfers[rid] = transfers
        return True

    def start_preparing_loads(self):
        return 0

    def complete(self, rid):
        full = next(t for t in self.transfers[rid] if t.name == PoolName.KV)
        self.loads[rid].set_result(
            NS(
                error=None,
                dram_pages=len(full.keys),
                ssd_pages=0,
                mixed_pages=0,
                elapsed_seconds=0.001,
                medium="dram",
            )
        )

    def load_ready(self, rid):
        return self.loads[rid].done()

    def take_load_result(self, rid):
        return self.loads.pop(rid).result()

    def forget_request(self, rid):
        self.release_lookup(rid)
        self.loads.pop(rid, None)
        self.transfers.pop(rid, None)

    def offload(self, transfers):
        return False

    def num_completed_offloads(self):
        return 0


class TestFlatMemoryAdmission(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            model_path="dummy", page_size=1, device="cpu"
        )
        override.install()
        self.addCleanup(override.restore)
        self.request_pool = ReqToTokenPool(
            size=4, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        pool = SWAKVPool(
            size=64,
            size_swa=64,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            device="cpu",
        )
        self.allocator = SWATokenToKVPoolAllocator(
            size=64,
            size_swa=64,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
            self.cache = UnifiedRadixCache(
                params=CacheInitParams(
                    req_to_token_pool=self.request_pool,
                    token_to_kv_pool_allocator=self.allocator,
                    page_size=1,
                    disable=False,
                    sliding_window_size=4,
                    tree_components=(ComponentType.FULL, ComponentType.SWA),
                )
            )
        self.storage = _Storage()
        self.runtime = FlatMemoryCache(
            cache=self.cache,
            cache_linker=self.storage,
            config={"prefetch_threshold": 1},
        )
        self.cache.flat_memory = self.runtime
        self.cache.linker = self.runtime
        self.cache.storage_metrics_collector = None

    def _req(self, rid="request", length=12, output=1):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", range(10, 10 + length)),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=output),
        )
        req.init_next_round_input(self.cache)
        return req

    def _adder(self, running=()):
        return PrefillAdder(
            page_size=1,
            tree_cache=self.cache,
            token_to_kv_pool_allocator=self.allocator,
            running_batch=NS(reqs=list(running)),
            new_token_ratio=1.0,
            rem_input_tokens=64,
            rem_chunk_tokens=16,
        )

    def _restored(self):
        req = self._req()
        state = self.runtime.prefetches[req.rid]
        self.assertIsNone(state.hold_params, "Metadata queries must not pin HBM")
        self.runtime.poll()
        adder = self._adder()
        self.runtime.prepare_admission([req], adder, has_chunked_req=False)
        self.assertEqual(state.phase, "read")
        self.storage.complete(req.rid)
        self.runtime.poll()
        self.assertEqual(state.phase, "ready")
        return req, state

    def _match_len(self, req):
        return len(
            self.cache.match_prefix(
                MatchPrefixParams(key=RadixKey(req.origin_input_ids[:-1]))
            ).device_indices
        )

    def test_early_release_reproduces_loss_with_real_swa_eviction(self):
        """The former early-unpin interleaving loses a genuinely restored prefix."""
        req, state = self._restored()
        loaded = state.loaded_tokens
        self.assertEqual(self._match_len(req), loaded)
        self.runtime._release_hold(state)
        self.cache.evict(EvictParams(swa_num_tokens=64))
        self.assertLess(self._match_len(req), loaded)
        self.runtime.release_request(req.rid)

    def test_real_admission_hands_off_without_an_unprotected_interval(self):
        """Real Full/SWA eviction cannot remove the prefix before request-lock handoff."""
        req, state = self._restored()
        loaded = state.loaded_tokens
        self.cache.evict(EvictParams(swa_num_tokens=64, num_tokens=64))
        self.assertEqual(self._match_len(req), loaded)
        adder = self._adder()
        self.runtime.prepare_admission([req], adder, has_chunked_req=False)
        req.init_next_round_input(self.cache)
        self.runtime.before_admission(req, adder)
        result = adder.add_one_req(
            req, has_chunked_req=False, truncation_align_size=None
        )
        self.assertIn(req, adder.can_run_list)
        self.assertIn(
            result, (AddReqResult.CONTINUE, AddReqResult.NO_TOKEN, AddReqResult.OTHER)
        )
        self.runtime.after_admission(req, adder)
        self.runtime.finish_admission(adder, can_progress=True)
        self.assertEqual(req.storage_hit_length, loaded)
        self.assertEqual(self.runtime.admission_dropped_tokens, 0)
        self.assertNotIn(req.rid, self.runtime.prefetches)
        self.cache.evict(EvictParams(swa_num_tokens=64, num_tokens=64))
        self.assertEqual(self._match_len(req), loaded)
        self.cache.dec_lock_ref(
            req.last_node,
            DecLockRefParams(
                swa_uuid_for_lock=req.swa_uuid_for_lock,
                skip_lock_node_ids=req.skip_lock_node_ids,
            ),
        )
        self.cache.evict(EvictParams(swa_num_tokens=64, num_tokens=64))
        self.assertLess(self._match_len(req), loaded)

    def test_skipped_admission_keeps_restored_prefix_until_next_pass(self):
        """A skipped admission pass must not expose a restored prefix to eviction."""
        req, state = self._restored()
        loaded = state.loaded_tokens
        adder = self._adder()
        self.runtime.prepare_admission([req], adder, has_chunked_req=False)
        self.runtime.finish_admission(adder, can_progress=False)
        self.assertFalse(state.bypass)
        self.assertIsNotNone(state.hold_params)
        self.cache.evict(EvictParams(swa_num_tokens=64, num_tokens=64))
        self.assertEqual(self._match_len(req), loaded)

        adder = self._adder()
        self.runtime.prepare_admission([req], adder, has_chunked_req=False)
        req.init_next_round_input(self.cache)
        self.runtime.before_admission(req, adder)
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertIn(req, adder.can_run_list)
        self.runtime.after_admission(req, adder)
        self.runtime.finish_admission(adder, can_progress=True)
        self.assertEqual(req.storage_hit_length, loaded)
        self.assertEqual(self.runtime.ready_dropped_tokens, 0)
        self.assertEqual(self.runtime.admission_dropped_tokens, 0)
        self.cache.dec_lock_ref(
            req.last_node,
            DecLockRefParams(
                swa_uuid_for_lock=req.swa_uuid_for_lock,
                skip_lock_node_ids=req.skip_lock_node_ids,
            ),
        )

    def test_restore_reservation_accounts_for_admitted_chunk_future_input(self):
        """A concurrent restore must reserve the admitted chunk's remaining input."""
        req = self._req(length=32, output=6)
        self.runtime.release_request(req.rid)
        adder = self._adder()
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertIn(req, adder.can_run_list)
        self.assertEqual(req.extend_range.end, 16)
        self.addCleanup(
            self.cache.dec_lock_ref,
            req.last_node,
            DecLockRefParams(
                swa_uuid_for_lock=req.swa_uuid_for_lock,
                skip_lock_node_ids=req.skip_lock_node_ids,
            ),
        )
        original = (adder.rem_total_tokens, adder.cur_rem_tokens, adder.rem_swa_tokens)
        adder.set_flat_restore_reservation(full_tokens=3, swa_tokens=2)
        # 16 future input + 6 decode + 1 page, plus the 3-token restore reserve.
        self.assertEqual(adder.rem_total_tokens, original[0] - 26)
        self.assertEqual(adder.cur_rem_tokens, original[1] - 26)
        self.assertEqual(adder.rem_swa_tokens, original[2] - 6)
        adder.set_flat_restore_reservation(full_tokens=0, swa_tokens=0)
        self.assertEqual(
            original,
            (adder.rem_total_tokens, adder.cur_rem_tokens, adder.rem_swa_tokens),
        )

    def test_reservations_replace_not_accumulate_and_preserve_running_headroom(self):
        running = self._req("running", length=8, output=6)
        adder = self._adder([running])
        original = (adder.rem_total_tokens, adder.cur_rem_tokens, adder.rem_swa_tokens)
        adder.set_flat_restore_reservation(full_tokens=8, swa_tokens=3)
        reserved = (adder.rem_total_tokens, adder.cur_rem_tokens, adder.rem_swa_tokens)
        self.assertLess(reserved[0], original[0] - 8)
        self.assertLess(reserved[2], original[2] - 3)
        adder.set_flat_restore_reservation(full_tokens=8, swa_tokens=3)
        self.assertEqual(
            reserved,
            (adder.rem_total_tokens, adder.cur_rem_tokens, adder.rem_swa_tokens),
        )
        adder.set_flat_restore_reservation(full_tokens=0, swa_tokens=0)
        self.assertEqual(
            original,
            (adder.rem_total_tokens, adder.cur_rem_tokens, adder.rem_swa_tokens),
        )
        self.runtime.release_request(running.rid)

    def test_oversized_restore_does_not_reserve_or_allocate(self):
        req = self._req(length=60, output=8)
        adder = self._adder()
        free = self.allocator.full_available_size(), self.allocator.swa_available_size()
        plan = adder.plan_flat_restore(
            req,
            restored_prefix_len=59,
            full_tokens=59,
            swa_tokens=4,
        )
        self.assertFalse(plan.can_restore)
        self.assertTrue(plan.never_fits)
        self.assertEqual(
            free,
            (self.allocator.full_available_size(), self.allocator.swa_available_size()),
        )
        self.runtime.release_request(req.rid)


if __name__ == "__main__":
    unittest.main()
