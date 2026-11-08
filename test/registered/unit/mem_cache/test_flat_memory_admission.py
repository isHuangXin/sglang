"""Real CPU Full/SWA matching and admission protect restored prefixes."""

import unittest
from array import array
from concurrent.futures import Future
from types import SimpleNamespace as NS
from unittest.mock import patch

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
from sglang.srt.mem_cache.flat_memory_cache import (
    _RECLAIM_SYNC_POLLS,
    FlatMemoryCache,
    request_storage_namespace,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.flat_memory.payload import (
    MEDIA_HINT_ANY,
    MEDIA_HINT_DRAM,
    MEDIA_HINT_SSD,
)
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
        self.offload_enabled = False
        self.backups = []
        self.source_safe = set()
        self.ssd_rids = set()
        self.media_hints = []
        self.reclaimed_pages = set()

    def take_reclaimed_pages(self):
        pages, self.reclaimed_pages = self.reclaimed_pages, set()
        return pages

    def restore_ssd_pages(self, rid, *, start, end):
        return end - start if rid in self.ssd_rids else 0

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

    def offload(self, transfers, media_hint=0):
        if not self.offload_enabled:
            return False
        self.media_hints.append(media_hint)
        self.backups.append(Future())
        return True

    def complete_backup(self, index=0, *, success=True, unsafe=False):
        self.backups[index].set_result(
            NS(success=success, capacity_rejected=False, unsafe=unsafe)
        )

    def num_source_safe_offloads(self):
        count = 0
        for future in self.backups:
            if future not in self.source_safe:
                break
            count += 1
        return count

    def num_completed_offloads(self):
        count = 0
        for future in self.backups:
            if not future.done():
                break
            count += 1
        return count

    def pop_completed_offload_result(self):
        future = self.backups.pop(0)
        self.source_safe.discard(future)
        return future.result()


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

    def _req(self, rid="request", length=12, output=1, start=10):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", range(start, start + length)),
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

    def test_private_restore_eviction_revalidates_gpu_only_ready_request(self):
        """Real Full/SWA allocation must invalidate a peer's unleased GPU-only readiness."""
        cached, restored = self._restored()
        self.runtime.release_request(cached.rid)
        covered = self._req("covered")
        covered_state = self.runtime.prefetches[covered.rid]
        self.assertIsNone(covered_state.future)
        self.assertEqual(covered_state.phase, "ready")
        self.assertEqual(self._match_len(covered), restored.loaded_tokens)
        competitor = self._req("competitor", length=60, start=100)
        self.runtime.poll()
        adder = self._adder()
        self.runtime.prepare_admission(
            [covered, competitor], adder, has_chunked_req=False
        )
        self.assertEqual(self.runtime.prefetches[competitor.rid].phase, "read")
        self.assertLess(self._match_len(covered), restored.loaded_tokens)
        self.assertFalse(self.runtime.is_ready(covered.rid))
        self.assertEqual(covered_state.phase, "query")
        self.assertEqual(covered_state.query_start, 0)
        self.assertIsNone(covered_state.prepared)
        self.assertEqual(list(self.storage.loads), [competitor.rid])
        self.runtime.release_request(covered.rid)
        self.runtime.release_request(competitor.rid)
        self.storage.complete(competitor.rid)
        self.runtime.poll()
        self.assertEqual(self.runtime._leases, [])

    def test_dram_restore_is_not_blocked_by_an_in_flight_ssd_restore(self):
        """FC-MAS: a DRAM-resident prefix must not wait for an SSD read to finish."""
        ssd = self._req("ssd", length=12, start=10)
        dram = self._req("dram", length=12, start=40)
        self.storage.ssd_rids.add(ssd.rid)
        self.runtime.poll()
        self.runtime.prepare_admission([ssd], self._adder(), has_chunked_req=False)
        self.assertEqual(self.runtime.prefetches[ssd.rid].phase, "read")
        self.runtime.prepare_admission([ssd, dram], self._adder(), has_chunked_req=False)
        self.assertEqual(self.runtime.prefetches[dram.rid].phase, "read")
        self.assertEqual(self.runtime._leases, [ssd.rid, dram.rid])
        for req in (ssd, dram):
            self.runtime.release_request(req.rid)
            self.storage.complete(req.rid)
        self.runtime.poll()
        self.assertEqual(self.runtime._leases, [])

    def test_restore_read_starts_while_a_chunked_prefill_runs(self):
        """A restore must overlap a long chunked prefill instead of waiting for it."""
        chunked = self._req("chunked", length=32, output=2)
        self.runtime.release_request(chunked.rid)
        restore = self._req("restore", length=12, start=60)
        self.runtime.poll()
        adder = self._adder()
        adder.add_one_req(chunked, has_chunked_req=False, truncation_align_size=None)
        self.assertEqual(chunked.extend_range.end, 16)
        self.addCleanup(
            self.cache.dec_lock_ref,
            chunked.last_node,
            DecLockRefParams(
                swa_uuid_for_lock=chunked.swa_uuid_for_lock,
                skip_lock_node_ids=chunked.skip_lock_node_ids,
            ),
        )
        self.runtime.prepare_admission([restore], adder, has_chunked_req=True)
        state = self.runtime.prefetches[restore.rid]
        self.assertEqual(state.phase, "read")
        # The chunk's 16 remaining input tokens stay charged against admission.
        self.assertGreaterEqual(
            adder._flat_full_reservation, state.full_reservation + 16
        )
        self.runtime.release_request(restore.rid)
        self.storage.complete(restore.rid)
        self.runtime.poll()
        self.assertEqual(self.runtime._leases, [])

    def test_first_admission_fixes_the_write_scope_hint_for_its_backups(self):
        """FC-GVA Table 2: a short turn on a cached prefix writes to DRAM, even later."""
        req, state = self._restored()
        node_id = self.cache.resolve_node_handle(state.hold_node).id
        # A retracted request re-admits with its own tokens cached; its first
        # admission's scope must survive, here pre-set to the opposite medium.
        req.flat_media_hint = MEDIA_HINT_SSD
        adder = self._adder()
        self.runtime.prepare_admission([req], adder, has_chunked_req=False)
        req.init_next_round_input(self.cache)
        self.runtime.before_admission(req, adder)
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.runtime.after_admission(req, adder)
        self.runtime.finish_admission(adder, can_progress=True)
        self.assertEqual(req.flat_media_hint, MEDIA_HINT_SSD)
        req.flat_media_hint = None
        self.assertEqual(
            self.runtime._classify_scope(
                cached_tokens=len(req.prefix_indices),
                input_tokens=len(req.origin_input_ids),
            ),
            MEDIA_HINT_DRAM,
            "This restored request is a short turn on a cached prefix",
        )
        req.flat_media_hint = MEDIA_HINT_DRAM

        node = self.cache.resolve_node_handle(node_id)
        node.external_cache_stored, node.hit_count = False, 1
        self.storage.offload_enabled = True
        with self.runtime.write_scope(req):
            self.runtime.offload_nodes([node_id])
        node.external_cache_stored = False
        self.runtime.tree_linker._offload_node(node_id)  # outside any scope
        self.assertEqual(self.storage.media_hints, [MEDIA_HINT_DRAM, 0])
        for index in range(2):
            self.storage.complete_backup(index)
        self.runtime.poll()
        self.cache.dec_lock_ref(
            req.last_node,
            DecLockRefParams(
                swa_uuid_for_lock=req.swa_uuid_for_lock,
                skip_lock_node_ids=req.skip_lock_node_ids,
            ),
        )

    def test_scope_rules_follow_table_2_with_short_turn_before_agentic(self):
        classify = self.runtime._classify_scope
        tau = self.runtime.scope_short_turn_tokens
        # Turn 1: nothing cached at arrival, however short.
        self.assertEqual(classify(cached_tokens=0, input_tokens=10), MEDIA_HINT_SSD)
        # A long agentic context with a short new suffix is still a short turn.
        self.assertEqual(
            classify(cached_tokens=200_000, input_tokens=200_000 + tau - 1),
            MEDIA_HINT_DRAM,
        )
        self.assertEqual(
            classify(cached_tokens=200_000, input_tokens=200_000 + tau), MEDIA_HINT_SSD
        )
        # With L_agent set, a long context outranks the short-turn rule.
        self.runtime.scope_agent_context_tokens = 200_000
        self.assertEqual(
            classify(cached_tokens=200_000, input_tokens=200_001), MEDIA_HINT_SSD
        )
        self.assertEqual(
            classify(cached_tokens=100_000, input_tokens=100_001), MEDIA_HINT_DRAM
        )
        self.runtime.scope_media_hints = False
        self.assertEqual(classify(cached_tokens=8, input_tokens=9), MEDIA_HINT_ANY)

    def test_reclaimed_page_lets_its_node_back_up_again(self):
        """After native reclamation, a node still marked stored would never be
        written again, so its KV is lost once HBM evicts it."""
        req, state = self._restored()
        node_id = self.cache.resolve_node_handle(state.hold_node).id
        node = self.cache.resolve_node_handle(node_id)
        node.external_cache_stored = False
        self.storage.offload_enabled = True
        self.runtime.offload_nodes([node_id])
        self.storage.complete_backup()
        self.runtime.poll()
        self.assertTrue(node.external_cache_stored)
        page = request_storage_namespace(node.key) + node.hash_value[0]
        self.storage.reclaimed_pages = {page}
        self.runtime._polls = _RECLAIM_SYNC_POLLS - 1
        self.runtime.poll()
        self.assertFalse(node.external_cache_stored)
        self.assertEqual(self.runtime.reclaim_invalidated_nodes, 1)
        self.assertNotIn(page, self.runtime._page_nodes)
        self.runtime.release_request(req.rid)

    def test_reclaimed_page_after_a_split_resolves_to_the_ancestor(self):
        parent = NS(id=7, hash_value=["h0", "h1"], parent=None)
        child = NS(id=8, hash_value=["h2"], parent=parent)
        nodes = {7: parent, 8: child}
        self.runtime._page_nodes = {"ns:h1": (8, "h1"), "ns:h2": (8, "h2")}
        self.storage.reclaimed_pages = {"ns:h1", "ns:h2", "ns:unknown"}
        with patch.object(self.cache.tree_core, "try_node_by_id", nodes.get):
            self.assertEqual(self.runtime._reclaimed_node_ids(), {7, 8})

    def test_shared_head_rule_sends_only_the_prompt_head_to_dram(self):
        """Under write-through a system prompt is stored before it is shared."""
        root = NS(key=None, parent=None, hit_count=0)
        head = NS(key=[0] * 4096, parent=root, hit_count=1)
        body = NS(key=[0] * 4096, parent=head, hit_count=1)
        self.runtime._write_scope_hint = MEDIA_HINT_SSD
        self.assertEqual(self.runtime._node_media_hint(head), MEDIA_HINT_SSD)
        self.runtime.scope_shared_head_tokens = 4096
        self.assertEqual(self.runtime._node_media_hint(head), MEDIA_HINT_DRAM)
        self.assertEqual(self.runtime._node_media_hint(body), MEDIA_HINT_SSD)
        self.runtime._write_scope_hint = MEDIA_HINT_ANY

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

    def _blocked_by_backup(self):
        cached, restored = self._restored()
        node_id = self.cache.resolve_node_handle(restored.hold_node).id
        self.runtime.release_request(cached.rid)
        self.storage.offload_enabled = True
        self.runtime.tree_linker._offload_node(node_id)
        req = self._req("waiting", length=60, start=100)
        state = self.runtime.prefetches[req.rid]
        self.runtime.poll()
        adder = self._adder()
        self.runtime.prepare_admission([req], adder, has_chunked_req=False)
        self.assertEqual(state.phase, "allocate")
        self.assertEqual(
            req.flat_prefetch_stats["flat_restore_reason"], "full_decode_headroom"
        )
        return req, state, adder

    def test_pending_backup_retirement_enables_restore_without_recompute(self):
        """An idle compute pass must wait for backup-owned capacity before recomputing."""
        req, state, adder = self._blocked_by_backup()
        self.runtime.finish_admission(adder, can_progress=False)
        self.assertFalse(state.bypass)
        self.assertIsNone(state.hold_params)
        self.assertIsNone(state.prepared)
        self.assertNotIn(req.rid, self.storage.loads)
        self.assertEqual(len(self.runtime.pending_offloads), 1)

        self.storage.complete_backup()
        self.runtime.poll()
        adder = self._adder()
        self.runtime.prepare_admission([req], adder, has_chunked_req=False)
        self.assertEqual(state.phase, "read")
        self.storage.complete(req.rid)
        self.runtime.poll()
        adder = self._adder()
        self.runtime.prepare_admission([req], adder, has_chunked_req=False)
        req.init_next_round_input(self.cache)
        self.runtime.before_admission(req, adder)
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertIn(req, adder.can_run_list)
        self.runtime.after_admission(req, adder)
        self.runtime.finish_admission(adder, can_progress=True)
        self.assertEqual(req.storage_read_tokens, 59)
        self.assertEqual(req.storage_hit_length, 59)
        self.assertEqual(self.runtime.lease_fallbacks, 0)
        self.cache.dec_lock_ref(
            req.last_node,
            DecLockRefParams(
                swa_uuid_for_lock=req.swa_uuid_for_lock,
                skip_lock_node_ids=req.skip_lock_node_ids,
            ),
        )

    def test_source_release_enables_real_allocation_before_durable_completion(self):
        """A safely copied backup must stop pinning pages while its disk write waits."""
        req, state, adder = self._blocked_by_backup()
        self.runtime.finish_admission(adder, can_progress=False)
        pending = self.runtime.pending_offloads[0]
        node_id = pending.lock_node_id
        self.cache.resolve_node_handle(node_id).external_cache_stored = False
        self.storage.source_safe.add(self.storage.backups[0])
        self.runtime.poll()
        self.assertFalse(pending.source_lock_held)
        self.assertFalse(self.storage.backups[0].done())
        self.assertEqual(len(self.runtime.pending_offloads), 1)

        adder = self._adder()
        self.runtime.prepare_admission([req], adder, has_chunked_req=False)
        self.assertEqual(state.phase, "read")
        self.assertIsNone(self.cache.tree_core.try_node_by_id(node_id))
        self.assertFalse(self.storage.backups[0].done())
        self.storage.complete_backup()
        self.runtime.poll()
        self.assertFalse(self.runtime.pending_offloads)
        self.runtime.release_request(req.rid)
        self.storage.complete(req.rid)
        self.runtime.poll()
        self.assertEqual(self.allocator.full_available_size(), 64)

    def test_allocation_wait_timeout_survives_query_refresh(self):
        """Refreshing a lookup cannot renew a resource wait or keep backup pages hostage."""
        req, state, adder = self._blocked_by_backup()
        self.runtime.finish_admission(adder, can_progress=False)
        deadline = state.allocation_wait_deadline
        self.assertGreater(deadline, 0)
        result = self.runtime._rematch(state)
        self.runtime._refresh_query(state, result)
        self.assertEqual(state.phase, "query")
        self.assertEqual(state.allocation_wait_deadline, deadline)
        with patch(
            "sglang.srt.mem_cache.flat_memory_cache.time.monotonic",
            return_value=deadline + 1,
        ):
            self.runtime.poll()
        self.assertTrue(state.bypass)
        self.assertEqual(state.revoke_reason, "allocation_wait_timeout")
        self.assertEqual(
            req.flat_prefetch_stats["flat_restore_plan_reason"], "full_decode_headroom"
        )
        self.assertEqual(len(self.runtime.pending_offloads), 1)
        self.assertFalse(self.storage.backups[0].done())
        self.assertNotIn(req.rid, self.storage.loads)
        self.runtime.release_request(req.rid)
        self.storage.complete_backup()
        self.runtime.poll()

    def test_backup_completion_does_not_credit_another_owners_pages(self):
        """Retiring one backup must not allocate pages still pinned by another owner."""
        req, state, adder = self._blocked_by_backup()
        node_id = self.runtime.pending_offloads[0].lock_node_id
        other_owner = self.cache.inc_lock_ref(node_id).to_dec_params()
        self.runtime.finish_admission(adder, can_progress=False)
        self.assertFalse(state.bypass)
        self.storage.complete_backup()
        self.runtime.poll()
        adder = self._adder()
        self.runtime.prepare_admission([req], adder, has_chunked_req=False)
        self.runtime.finish_admission(adder, can_progress=False)
        self.assertTrue(state.bypass)
        self.assertEqual(state.revoke_reason, "no_admission_progress")
        self.assertNotIn(req.rid, self.storage.loads)
        self.cache.dec_lock_ref(node_id, other_owner)
        self.runtime.release_request(req.rid)

    def test_cancelled_allocate_wait_preserves_backup_until_safe_failure(self):
        """Cancelling a waiter cannot release the source lock of its independent backup."""
        req, state, adder = self._blocked_by_backup()
        self.runtime.finish_admission(adder, can_progress=False)
        before = self.allocator.full_available_size() + self.cache.full_evictable_size()
        self.runtime.release_request(req.rid)
        self.assertNotIn(req.rid, self.runtime.prefetches)
        self.assertEqual(len(self.runtime.pending_offloads), 1)
        self.assertEqual(
            self.allocator.full_available_size() + self.cache.full_evictable_size(),
            before,
        )
        self.assertEqual(
            req.flat_prefetch_stats["flat_restore_allocation_wait_outcome"], "cancelled"
        )
        self.storage.complete_backup(success=False)
        self.runtime.poll()
        self.assertFalse(self.runtime.pending_offloads)
        self.assertEqual(self.runtime.io_errors, 1)
        self.assertEqual(
            self.allocator.full_available_size() + self.cache.full_evictable_size(), 64
        )

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
