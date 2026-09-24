import copy
import time
import unittest
from array import array
from concurrent.futures import Future
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    IncLockRefResult,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer, SidecarPoolSpec
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import PrefetchOperation
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _STRATEGIES,
    StackBuildResult,
    StackStrategy,
    _apply_stack_result,
    _DeepSeekV4Strategy,
    _DsaStrategy,
    _MambaStrategy,
    _MiniMaxSparseStrategy,
    _PlainKvStrategy,
    _select_strategy,
    _SwaStrategy,
    register_stack_strategy,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.mem_cache.unified_cache.tiered_mooncake_runtime import (
    TieredMooncakeRuntime,
    _LoadRequest,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedLRUList,
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, _OngoingPrefetch
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _mock_kvcache(cls):
    return MagicMock(spec=cls)


FULL = ComponentType.FULL
SWA = ComponentType.SWA
MAMBA = ComponentType.MAMBA


class TestUnifiedRadixHiCacheDispatch(unittest.TestCase):
    def test_strategy_registry_ordering(self):
        order = [type(s) for s in _STRATEGIES]
        # DeepSeekV4 inherits from SWAKVPool, so it must resolve before _SwaStrategy.
        self.assertLess(order.index(_DeepSeekV4Strategy), order.index(_SwaStrategy))
        self.assertLess(
            order.index(_MiniMaxSparseStrategy), order.index(_PlainKvStrategy)
        )
        self.assertEqual(order[-1], _PlainKvStrategy)

    def test_deepseek_v4_full_swa(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        kvcache = _mock_kvcache(DeepSeekV4TokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL, SWA})
        self.assertIsInstance(strategy, _DeepSeekV4Strategy)

    def test_mamba(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        kvcache = _mock_kvcache(HybridLinearKVPool)
        strategy = _select_strategy(kvcache, {FULL, MAMBA})
        self.assertIsInstance(strategy, _MambaStrategy)

    def test_swa(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        kvcache = _mock_kvcache(SWAKVPool)
        strategy = _select_strategy(kvcache, {FULL, SWA})
        self.assertIsInstance(strategy, _SwaStrategy)

    def test_dsa(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        kvcache = _mock_kvcache(DSATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _DsaStrategy)

    def test_minimax_sparse(self):
        from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

        kvcache = _mock_kvcache(MiniMaxSparseKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _MiniMaxSparseStrategy)

    def test_minimax_sparse_build_registers_indexer_sidecar(self):
        strategy = _MiniMaxSparseStrategy()
        host_pool_group = MagicMock()
        kv_host_pool = object()
        host_pool_group.get_pool.return_value = kv_host_pool
        cache_controller = MagicMock()
        cache = MagicMock(page_size=4)
        kvcache = MagicMock()
        kvcache.index_k_pool = object()
        kvcache.main_pool.layer_num = 8
        params = MagicMock()
        params.tp_cache_group = None
        params.pp_rank = 0
        params.pp_size = 1
        server_args = MagicMock()

        with patch.object(
            hybrid_pool_assembler,
            "build_minimax_sparse_hicache_stack",
            return_value=(host_pool_group, cache_controller),
        ) as build_stack:
            result = strategy.build(
                cache=cache,
                kvcache=kvcache,
                params=params,
                server_args=server_args,
                load_cache_event=object(),
            )

        build_stack.assert_called_once()
        self.assertIs(build_stack.call_args.kwargs["sparse_pool"], kvcache)
        self.assertIs(result.host_pool_group, host_pool_group)
        self.assertIs(result.cache_controller, cache_controller)
        self.assertIs(result.component_host_pools[FULL], kv_host_pool)
        self.assertEqual(result.pools_desc, "KV + INDEXER(k-only)")
        self.assertEqual(result.transfer_layer_num, 8)
        self.assertEqual(len(result.sidecars), 1)
        self.assertEqual(result.sidecars[0].pool_name, PoolName.INDEXER)
        self.assertEqual(result.sidecars[0].indices_from_pool, PoolName.KV)

    def test_plain_kv_fallback(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        kvcache = _mock_kvcache(MHATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _PlainKvStrategy)

    def test_mla_routes_to_plain(self):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        kvcache = _mock_kvcache(MLATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _PlainKvStrategy)

    def test_unknown_combo_raises(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        for cls in (SWAKVPool, DeepSeekV4TokenToKVPool):
            kvcache = _mock_kvcache(cls)
            with self.assertRaises(AssertionError) as cm:
                _select_strategy(kvcache, {FULL})
            self.assertIn("No matching HiCache strategy", str(cm.exception))

    def test_register_custom_strategy_takes_precedence(self):
        class _CustomStrategy(StackStrategy):
            def matches(self, kvcache, components):
                return components == {FULL}

            def build(self, **_):
                raise NotImplementedError

        custom = _CustomStrategy()
        original = list(hybrid_pool_assembler._STRATEGIES)
        try:
            register_stack_strategy(custom)
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

            kvcache = _mock_kvcache(MHATokenToKVPool)
            self.assertIs(_select_strategy(kvcache, {FULL}), custom)
        finally:
            hybrid_pool_assembler._STRATEGIES[:] = original


class TestApplyStackResult(unittest.TestCase):
    @staticmethod
    def _fake_cache(component_types):
        cache = MagicMock()
        cache.components = {ct: MagicMock() for ct in component_types}
        return cache

    def test_wires_components_sidecars_and_counters(self):
        full_host, swa_host, mamba_host = MagicMock(), MagicMock(), MagicMock()
        cache = self._fake_cache([FULL, SWA, MAMBA])
        kvcache = MagicMock()
        params = MagicMock()
        controller = MagicMock()
        sidecar = SidecarPoolSpec(
            pool_name=PoolName.INDEXER, indices_from_pool=PoolName.KV
        )
        result = StackBuildResult(
            host_pool_group=MagicMock(),
            cache_controller=controller,
            component_host_pools={FULL: full_host, SWA: swa_host, MAMBA: mamba_host},
            sidecars=[sidecar],
            register_req_to_token_counter=True,
            transfer_layer_num=8,
            pools_desc="KV + SWA + MAMBA",
        )

        _apply_stack_result(cache, kvcache, params, result)

        self.assertIs(cache.host_pool_group, result.host_pool_group)
        self.assertIs(cache.cache_controller, controller)
        self.assertIs(cache.full_kv_pool_host, full_host)
        self.assertIs(cache.swa_kv_pool_host, swa_host)
        self.assertIs(cache.mamba_pool_host, mamba_host)
        self.assertIs(cache.components[FULL]._full_kv_pool_host, full_host)
        self.assertIs(cache.components[SWA]._swa_kv_pool_host, swa_host)
        self.assertIs(cache.components[MAMBA]._mamba_pool_host, mamba_host)
        cache.register_sidecar_pool.assert_called_once_with(sidecar)
        kvcache.register_layer_transfer_counter.assert_called_once_with(
            controller.layer_done_counter
        )
        params.req_to_token_pool.register_layer_transfer_counter.assert_called_once_with(
            controller.layer_done_counter
        )

    def test_skips_req_to_token_counter_when_flag_false(self):
        cache = self._fake_cache([FULL])
        kvcache = MagicMock()
        params = MagicMock()
        result = StackBuildResult(
            host_pool_group=MagicMock(),
            cache_controller=MagicMock(),
            component_host_pools={FULL: MagicMock()},
            sidecars=[],
            register_req_to_token_counter=False,
            transfer_layer_num=1,
            pools_desc="KV",
        )

        _apply_stack_result(cache, kvcache, params, result)

        kvcache.register_layer_transfer_counter.assert_called_once()
        params.req_to_token_pool.register_layer_transfer_counter.assert_not_called()
        cache.register_sidecar_pool.assert_not_called()


class _PrefetchHostPool:
    def __init__(self, available):
        self.available = available
        self.next_index = 0
        self.allocated = set()
        self.released = []

    def available_size(self):
        return self.available

    def alloc(self, size):
        if size > self.available:
            return None
        indices = torch.arange(self.next_index, self.next_index + size)
        self.next_index += size
        self.available -= size
        self.allocated.update(indices.tolist())
        return indices

    def free(self, indices):
        values = indices.tolist()
        assert set(values) <= self.allocated, "Host slots released twice"
        self.allocated.difference_update(values)
        self.released.extend(values)
        self.available += len(values)

    def reclaim(self, size, *args):
        self.available += size


class TestNativePrefetchCapacity(CustomTestCase):
    PAGE = 256

    def _cache(self, free_pages=6):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            page_size=self.PAGE,
            enable_storage=True,
            is_root=lambda node: node == 0,
            has_swa_host_pool=False,
            is_eagle=False,
            prefetch_anchor_info=lambda node: (None, None),
        )
        cache.prefetch_threshold = self.PAGE
        cache.host_memory_mode = "cache"
        cache.buffer_pipeline = None
        cache.tiered_runtime = None
        cache.linker = None
        cache.tree_components = [FULL]
        cache.components = {}
        cache.ongoing_prefetch = {}
        cache.completed_prefetch_holds = {}
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache._storage_prefetch_missed_rids = set()
        cache._storage_prefetch_deferred_rids = set()
        cache._prefetch_outcome_stats = {
            "attempts": 0,
            "issued": 0,
            "declined_too_short": 0,
            "declined_rate_limited": 0,
        }
        cache._all_reduce = MagicMock()
        cache.evict_host = MagicMock()
        cache.inc_host_lock_ref = MagicMock(
            return_value=SimpleNamespace(to_dec_params=lambda: None)
        )
        cache.dec_host_lock_ref = MagicMock()
        cache._build_sidecar_transfers = MagicMock(return_value=[])
        cache.cache_controller = SimpleNamespace(
            mem_pool_host=_PrefetchHostPool(free_pages * self.PAGE),
            prefetch_hit_queue=Queue(),
            ack_prefetch_queue=Queue(),
            ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(),
            extra_host_mem_release_queues={},
            prefetch_buffer=Queue(),
            prefetch_tokens_occupied=0,
            prefetch_capacity_limit=8 * self.PAGE,
            prefetch_rate_limited=MagicMock(return_value=False),
            append_host_mem_release=MagicMock(),
            prefetch=MagicMock(return_value=SimpleNamespace()),
        )
        return cache

    def _drain_hit(self, cache, endpoints):
        operation = PrefetchOperation("r", [1024] * (8 * self.PAGE), None)
        operation.hash_value = [f"page-{i}" for i in range(8)]
        operation.storage_hit_count = 8 * self.PAGE
        operation.pool_storage_result.restorable_prefix_pages = endpoints
        aux = PoolTransfer(name=PoolName.SWA, host_indices=torch.arange(self.PAGE))
        cache.ongoing_prefetch["r"] = _OngoingPrefetch(
            0, operation.token_ids, None, operation, None, {SWA: [aux]}
        )
        cache.cache_controller.prefetch_tokens_occupied = 8 * self.PAGE
        cache.cache_controller.prefetch_hit_queue.put(operation)
        cache._drain_storage_control_queues_impl(1, 0, 0, 0, {}, False)
        return operation, aux

    def test_pressure_shrink_uses_a_restorable_endpoint(self):
        """A six-page allocation must not target a checkpoint stored only at four/eight."""
        cache = self._cache()
        operation, _ = self._drain_hit(cache, [4, 8])
        self.assertEqual(operation.storage_hit_count, 4 * self.PAGE)
        self.assertEqual(len(operation.host_indices), 4 * self.PAGE)
        self.assertEqual(operation.hash_value, [f"page-{i}" for i in range(4)])
        self.assertIs(cache.cache_controller.prefetch_buffer.get_nowait(), operation)

    def test_dense_prefix_still_uses_all_available_pages(self):
        cache = self._cache()
        operation, _ = self._drain_hit(cache, None)
        self.assertEqual(operation.storage_hit_count, 6 * self.PAGE)

    def test_no_fitting_checkpoint_releases_aux_without_a_read(self):
        cache = self._cache()
        _, aux = self._drain_hit(cache, [8])
        controller = cache.cache_controller
        self.assertTrue(controller.prefetch_buffer.empty())
        self.assertNotIn("r", cache.ongoing_prefetch)
        self.assertEqual(controller.prefetch_tokens_occupied, 0)
        controller.append_host_mem_release.assert_called_once_with(extra_pools=[aux])
        self.assertFalse(controller.mem_pool_host.allocated)

    def test_rank_capacity_shrink_frees_provisional_surplus(self):
        cache = self._cache(free_pages=8)
        calls = []

        def reduce_capacity(value, op):
            calls.append(value.item())
            if len(calls) == 1:
                value.fill_(6 * self.PAGE)

        cache._all_reduce.side_effect = reduce_capacity
        operation, _ = self._drain_hit(cache, [4, 8])
        self.assertEqual(operation.storage_hit_count, 4 * self.PAGE)
        self.assertEqual(
            len(cache.cache_controller.mem_pool_host.released), 4 * self.PAGE
        )
        self.assertEqual(
            len(cache.cache_controller.mem_pool_host.allocated), 4 * self.PAGE
        )

    def test_full_reclaim_requests_only_missing_tokens(self):
        cache = self._cache(free_pages=2)
        cache.evict_host.side_effect = cache.cache_controller.mem_pool_host.reclaim
        operation, _ = self._drain_hit(cache, [4, 8])
        cache.evict_host.assert_called_once_with(6 * self.PAGE)
        self.assertEqual(operation.storage_hit_count, 8 * self.PAGE)

    def test_swa_reclaim_requests_only_missing_tokens(self):
        cache = self._cache()
        pool = _PrefetchHostPool(2 * self.PAGE)
        component = SWAComponent.__new__(SWAComponent)
        component.cache = cache
        component._swa_kv_pool_host = pool
        component.full_window_pages = 4
        cache.evict_host.side_effect = pool.reclaim

        def alloc(size, *, pool: PoolName, reclaim):
            indices = component._swa_kv_pool_host.alloc(size)
            if indices is None:
                reclaim(size)
                indices = component._swa_kv_pool_host.alloc(size)
            return indices

        cache.host_pool_group = SimpleNamespace(alloc=alloc)
        result = component.prepare_prefetch(0, prefetch_tokens=8 * self.PAGE)
        self.assertEqual(len(result.host_indices), 4 * self.PAGE)
        cache.evict_host.assert_called_once_with(2 * self.PAGE, SWA)

    def test_projected_budget_defers_before_any_host_allocation(self):
        cache = self._cache()
        cache.cache_controller.prefetch_tokens_occupied = 6 * self.PAGE
        cache.prefetch_from_storage("r", 0, [1024] * (4 * self.PAGE))
        cache.inc_host_lock_ref.assert_not_called()
        cache._build_sidecar_transfers.assert_not_called()
        cache.cache_controller.prefetch.assert_not_called()
        self.assertTrue(cache.pop_storage_prefetch_deferred("r"))
        self.assertFalse(cache.pop_storage_prefetch_deferred("r"))
        self.assertFalse(cache.pop_storage_prefetch_miss("r"))

    def test_oversized_query_preserves_input_and_demand_accounting(self):
        cache = self._cache()
        tokens = [1024] * (12 * self.PAGE)
        cache.prefetch_from_storage(
            "r", 0, tokens, matched_prefix_tokens=[1024] * (2 * self.PAGE)
        )
        call = cache.cache_controller.prefetch.call_args
        self.assertEqual(len(call.args[1]), 8 * self.PAGE)
        self.assertEqual(len(tokens), 12 * self.PAGE)
        operation = cache.cache_controller.prefetch.return_value
        self.assertEqual(operation.stats_requested_tokens, 12 * self.PAGE)
        self.assertEqual(operation.stats_total_tokens, 14 * self.PAGE)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 8 * self.PAGE)

    def _complete_prefetch(self, cache, *, matched_pages=0, overlap_pages=0):
        tokens = [1024] * (4 * self.PAGE)
        operation = PrefetchOperation("r", tokens, None)
        operation.completed_tokens = len(tokens)
        operation.hash_value = ["page"] * 4
        operation.stats_requested_tokens = len(tokens)
        operation.stats_total_tokens = len(tokens) + matched_pages * self.PAGE
        indices = cache.cache_controller.mem_pool_host.alloc(len(tokens))
        cache.ongoing_prefetch["r"] = _OngoingPrefetch(
            0, tokens, indices, operation, None, {}
        )
        cache.cache_controller.prefetch_tokens_occupied += len(tokens)
        root, leaf = UnifiedTreeNode((FULL,)), UnifiedTreeNode((FULL,))
        leaf.parent = root
        leaf.component_data[FULL].host_value = torch.arange(len(tokens))
        cache.tree_core.root_node = root
        cache.tree_core.is_write_back = False
        cache.tree_core._update_evictable_leaf_sets = MagicMock()
        cache.tree_core.insert_host = MagicMock(
            return_value=SimpleNamespace(
                cache_actions=[],
                host_insert_dropped=False,
                inserted_host_node=leaf.id,
                prefix_len=overlap_pages * self.PAGE,
            )
        )
        cache.tree_core.commit_hicache_transfers = MagicMock()
        cache._apply_cache_actions = MagicMock()
        cache._check_hybrid_prefetch_result = MagicMock(return_value=True)
        cache.enable_storage_metrics = False
        full = FullComponent.__new__(FullComponent)
        full.tree_core = cache.tree_core

        def acquire(node_id):
            cache.tree_core.commit_hicache_transfers.assert_called_once()
            return full.acquire_component_lock(leaf, IncLockRefResult(), lock_host=True)

        def release(node_id, params):
            if node_id == leaf.id:
                full.release_component_lock(leaf, params, lock_host=True)

        cache.inc_host_lock_ref.side_effect = acquire
        cache.dec_host_lock_ref.side_effect = release
        cache._handle_prefetch_result(operation)
        return leaf

    def test_completed_prefetch_survives_until_handoff(self):
        """Popping hit statistics must not expose a queued prefix to eviction."""
        for overlap in (0, 4):
            with self.subTest(overlap=overlap):
                cache = self._cache()
                leaf = self._complete_prefetch(cache, overlap_pages=overlap)
                self.assertTrue(cache.check_prefetch_progress("r"))
                cache.pop_prefetch_loaded_tokens("r")
                self.assertFalse(UnifiedTreeCore._is_host_leaf(cache.tree_core, leaf))
                self.assertEqual(leaf.component_data[FULL].host_lock_ref, 1)
                cache.prefetch_from_storage("r", 0, [1024] * self.PAGE)
                cache.cache_controller.prefetch.assert_not_called()
                allocated = set(cache.cache_controller.mem_pool_host.allocated)

                cache.release_prefetch_hold("r")
                cache.release_prefetch_hold("r")
                self.assertTrue(UnifiedTreeCore._is_host_leaf(cache.tree_core, leaf))
                self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)
                self.assertEqual(
                    cache.cache_controller.mem_pool_host.allocated, allocated
                )

    def test_completed_hold_charges_full_prefix_within_budget(self):
        """A short fetched suffix must not undercharge the retained ancestor path."""
        for capacity, retained in ((8, 6), (6, 0)):
            with self.subTest(capacity=capacity):
                cache = self._cache()
                cc = cache.cache_controller
                cc.prefetch_capacity_limit = capacity * self.PAGE
                cc.prefetch_tokens_occupied = 2 * self.PAGE
                leaf = self._complete_prefetch(cache, matched_pages=2)
                self.assertEqual(
                    cc.prefetch_tokens_occupied, (2 + retained) * self.PAGE
                )
                self.assertEqual(
                    leaf.component_data[FULL].host_lock_ref, int(retained > 0)
                )
                cache.release_prefetch_hold("r")
                self.assertEqual(cc.prefetch_tokens_occupied, 2 * self.PAGE)

    def test_completed_hold_released_on_abort_or_detach(self):
        """Completed reads are no longer in ongoing_prefetch when cleanup runs."""
        for cleanup in ("abort", "detach"):
            with self.subTest(cleanup=cleanup):
                cache = self._cache()
                leaf = self._complete_prefetch(cache)
                self.assertFalse(cache.ongoing_prefetch)
                if cleanup == "abort":
                    cache.release_aborted_request("r")
                else:
                    cache.ongoing_backup = {}
                    attachment = StorageAttachment.__new__(StorageAttachment)
                    attachment._cache = cache
                    attachment._release_pending_storage_ops()
                self.assertFalse(cache.completed_prefetch_holds)
                self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)
                self.assertEqual(leaf.component_data[FULL].host_lock_ref, 0)

    def test_swa_host_hold_survives_split_and_releases_both_halves(self):
        """Splitting a held window must not expose its parent to host eviction."""
        root, child, parent = (UnifiedTreeNode((SWA,)) for _ in range(3))
        child.parent = root
        child.key = list(range(8))
        child.component_data[SWA].host_value = torch.arange(8)
        lru = UnifiedLRUList(SWA, (SWA,), use_host_ptr=True)
        lru.insert_mru(child)
        component = SWAComponent.__new__(SWAComponent)
        component.tree_core = SimpleNamespace(root_node=root, host_lru_lists={SWA: lru})
        component.sliding_window_size = 6
        params = component.acquire_component_lock(
            child, IncLockRefResult(), lock_host=True
        ).to_dec_params()

        parent.key, child.key = child.key[:4], child.key[4:]
        parent.parent, child.parent = root, parent
        component.redistribute_on_node_split(parent, child)
        self.assertIsNone(lru.get_lru_no_host_lock())
        self.assertEqual(parent.component_data[SWA].host_lock_ref, 1)
        self.assertEqual(child.component_data[SWA].host_lock_ref, 1)
        component.release_component_lock(child, params, lock_host=True)
        self.assertEqual(parent.component_data[SWA].host_lock_ref, 0)
        self.assertEqual(child.component_data[SWA].host_lock_ref, 0)
        self.assertIsNotNone(lru.get_lru_no_host_lock())

    def test_storage_disabled_does_not_touch_host(self):
        cache = self._cache()
        cache.enable_storage = False
        cache.prefetch_from_storage("r", 0, [1024] * (4 * self.PAGE))
        cache.inc_host_lock_ref.assert_not_called()
        cache.cache_controller.prefetch.assert_not_called()
        cache.evict_host.assert_not_called()


class TestTieredGDSSWAFrontier(CustomTestCase):
    def _fixture(self):
        pool = SWAKVPool(
            size=64,
            size_swa=64,
            page_size=1,
            dtype=torch.float32,
            head_num=1,
            head_dim=2,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            device="cpu",
        )
        allocator = SWATokenToKVPoolAllocator(
            size=64,
            size_swa=64,
            page_size=1,
            dtype=torch.float32,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
        rows = ReqToTokenPool(
            size=2,
            max_context_len=64,
            device="cpu",
            enable_memory_saver=False,
        )
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
            cache = UnifiedRadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=rows,
                    token_to_kv_pool_allocator=allocator,
                    page_size=1,
                    sliding_window_size=4,
                    tree_components=(FULL, SWA),
                )
            )
        req = Req("gds", "", array("q", range(29)), SamplingParams(max_new_tokens=1))
        req.init_next_round_input(cache)
        rows.alloc([req])
        self._extend(cache, req, 8)
        free_swa_out_of_window_slots(
            req,
            8,
            sliding_window_size=4,
            page_size=1,
            req_to_token_pool=rows,
            token_to_kv_pool_allocator=allocator,
        )
        cache.cache_unfinished_req(req, chunked=True)
        self.assertEqual(req.kv.swa_evicted_seqlen, 4)
        self.assertEqual(len(req.prefix_indices), 8)
        return cache, req

    def _extend(self, cache, req, tokens):
        start = len(req.prefix_indices)
        end = start + tokens
        slots = cache.token_to_kv_pool_allocator.alloc(tokens)
        self.assertIsNotNone(slots)
        cache.req_to_token_pool.write((req.kv.req_pool_idx, slice(start, end)), slots)
        req.set_extend_range(start, end)
        req.kv.kv_allocated_len = req.kv.kv_committed_len = end

    def _publish(self, cache, req, *, duplicate):
        loads = UnifiedCacheLinkerWrapper(cache, None, configure_tree=False)
        hit = ExternalCacheHitMarker(
            prefix_key=RadixKey(req.origin_input_ids[:20]),
            tail_hashes=[str(i) for i in range(8, 20)],
            device_hit_len=8,
        )

        def prepare():
            return loads.prepare_load(
                _LoadRequest(
                    rid=req.rid,
                    prefix_indices=req.prefix_indices,
                    last_node=req.last_node,
                    kv=copy.copy(req.kv),
                    priority=0,
                ),
                hit=hit,
            )

        prepared = prepare()
        self.assertEqual(prepared.req.kv.swa_evicted_seqlen, 16)
        self.assertEqual(req.kv.swa_evicted_seqlen, 4)
        if duplicate:
            loads.commit_prepared_load(prepare(), queue_io=False)
        op = SimpleNamespace(
            req=req,
            prepared=prepared,
            prefix_len=8,
            full_tokens=12,
            anchor=req.last_node,
            started=time.monotonic(),
            device_lock=cache.inc_lock_ref(req.last_node).to_dec_params(),
            host_lock=cache.inc_host_lock_ref(req.last_node).to_dec_params(),
        )
        runtime = TieredMooncakeRuntime.__new__(TieredMooncakeRuntime)
        runtime.cache, runtime.loads = cache, loads
        runtime.controller = SimpleNamespace(
            l2_transfer_engine=SimpleNamespace(device_to_host_stream=MagicMock()),
            ack_write_queue=[],
            prefetch_tokens_occupied=12,
            prefetch_capacity_limit=12 if duplicate else 32,
        )
        runtime.pending = {req.rid: op}
        runtime.completed_holds = {}
        runtime.sources, runtime.loaded, runtime.latencies = {}, {}, {}
        with patch.object(cache, "writing_check"), patch.object(
            cache, "_execute_and_commit_kv_backup"
        ):
            runtime._commit(
                req.rid, op, {PoolName.KV.value: [2] * 12, PoolName.SWA.value: [2] * 12}
            )
        return runtime, prepared

    def _admit(self, cache, req):
        cache._dec_req_lock(req)
        req.init_next_round_input(cache)
        lock = cache.inc_lock_ref(req.last_node)
        req.swa_uuid_for_lock = lock.swa_uuid_for_lock
        req.skip_lock_node_ids = lock.skip_lock_node_ids
        cache.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, len(req.prefix_indices))), req.prefix_indices
        )

    def _assert_released(self, cache, req, runtime):
        runtime.release_hold(req.rid)
        cache._dec_req_lock(req)
        cache.req_to_token_pool.free(req)
        cache.evict(EvictParams(num_tokens=64, swa_num_tokens=64))
        allocator = cache.token_to_kv_pool_allocator
        self.assertEqual(allocator.full_available_size(), 64)
        self.assertEqual(allocator.swa_available_size(), 64)
        self.assertFalse(runtime.pending)
        self.assertFalse(runtime.completed_holds)
        self.assertEqual(runtime.controller.prefetch_tokens_occupied, 0)

    def test_duplicate_restore_then_shorter_chunk_recovers_swa(self):
        """A farther duplicate restore must not mark the next computed chunk dead."""
        cache, req = self._fixture()
        runtime, prepared = self._publish(cache, req, duplicate=True)
        self.assertFalse(any(prepared.adopted_ranges.values()))
        self.assertFalse(runtime.completed_holds)
        far_match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(req.origin_input_ids[:20]), req=req)
        )
        full_hold = cache.inc_lock_ref(
            far_match.last_device_node, (SWA,)
        ).to_dec_params()
        cache.evict(EvictParams(swa_num_tokens=4))
        self._admit(cache, req)
        self.assertEqual(len(req.prefix_indices), 8)
        self._extend(cache, req, 4)
        cache.cache_unfinished_req(req, chunked=True)
        self.assertEqual(req.kv.cache_protected_len, 12)
        self.assertEqual(len(req.prefix_indices), 12)
        cache.dec_lock_ref(far_match.last_device_node, full_hold)
        self._assert_released(cache, req, runtime)

    def test_query_anchor_can_consume_admitted_full_capacity(self):
        """An unallocated GDS query can pin all capacity promised to a chunk."""
        override = get_context().override_server_args(enable_hierarchical_cache=True)
        override.install()
        self.addCleanup(override.restore)
        cache, chunk = self._fixture()
        self._extend(cache, chunk, 8)
        cache.cache_unfinished_req(chunk, chunked=True)
        self.assertEqual(chunk.kv.cache_protected_len, 16)
        allocator = cache.token_to_kv_pool_allocator
        waiting = Req(
            "waiting", "", array("q", range(100, 153)), SamplingParams(max_new_tokens=1)
        )
        waiting.init_next_round_input(cache)
        loads = UnifiedCacheLinkerWrapper(cache, None, configure_tree=False)
        prepared = loads.prepare_load(
            _LoadRequest(
                rid=waiting.rid,
                prefix_indices=waiting.prefix_indices,
                last_node=waiting.last_node,
                kv=copy.copy(waiting.kv),
                priority=0,
            ),
            hit=ExternalCacheHitMarker(
                prefix_key=RadixKey(waiting.origin_input_ids[:48]),
                tail_hashes=[str(i) for i in range(48)],
                device_hit_len=0,
            ),
        )
        loads.commit_prepared_load(prepared, queue_io=False)
        waiting.init_next_round_input(cache)
        self.assertEqual(len(waiting.prefix_indices), 48)
        self.assertEqual(allocator.full_available_size(), 0)
        self.assertEqual(cache.full_evictable_size(), 48)

        def adder():
            return PrefillAdder(
                page_size=1,
                tree_cache=cache,
                token_to_kv_pool_allocator=allocator,
                running_batch=SimpleNamespace(reqs=[]),
                new_token_ratio=1.0,
                rem_input_tokens=4,
                rem_chunk_tokens=4,
            )

        admitted = adder()
        admitted.add_chunked_req(chunk)
        self.assertEqual(admitted.can_run_list, [chunk])
        runtime = TieredMooncakeRuntime.__new__(TieredMooncakeRuntime)
        runtime.cache, runtime.closed = cache, False
        runtime.controller = SimpleNamespace(
            prefetch_capacity_limit=8, prefetch_tokens_occupied=0
        )
        runtime.query_worker = SimpleNamespace(submit=lambda *args: Future())
        runtime.pending, runtime.completed_holds = {}, {}
        runtime.sources, runtime.loaded, runtime.latencies = {}, {}, {}
        cache.prefetch_threshold = 4
        runtime.prefetch(waiting)
        self.assertIn(waiting.rid, runtime.pending)
        self.assertFalse(runtime.pending[waiting.rid].query.done())
        self.assertEqual(runtime.full_reserved_tokens, 0)
        self.assertEqual(runtime.controller.prefetch_tokens_occupied, 0)
        self.assertEqual(cache.full_evictable_size(), 0)
        cache.evict(EvictParams(num_tokens=4))
        self.assertIsNone(allocator.alloc(4))
        after_prefetch = adder()
        after_prefetch.add_chunked_req(chunk)
        self.assertEqual(after_prefetch.can_run_list, [])
        runtime._discard(waiting.rid, runtime.pending[waiting.rid])
        self.assertEqual(cache.full_evictable_size(), 48)
        self._assert_released(cache, chunk, runtime)

    def test_full_restore_preserves_request_eviction_and_cached_slots(self):
        """Restored tree slots stay protected without overwriting real eviction progress."""
        cache, req = self._fixture()
        runtime, _ = self._publish(cache, req, duplicate=False)
        self.assertEqual(req.kv.swa_evicted_seqlen, 4)
        self._admit(cache, req)
        self.assertEqual(req.kv.cache_protected_len, 20)
        allocator = cache.token_to_kv_pool_allocator
        cached_swa = allocator.translate_loc_from_full_to_swa(
            req.prefix_indices
        ).clone()
        self._extend(cache, req, 8)
        free_swa_out_of_window_slots(
            req,
            28,
            sliding_window_size=4,
            page_size=1,
            req_to_token_pool=cache.req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
        )
        self.assertEqual(req.kv.swa_evicted_seqlen, 24)
        self.assertTrue(
            torch.equal(
                allocator.translate_loc_from_full_to_swa(req.prefix_indices), cached_swa
            )
        )
        cache.cache_unfinished_req(req, chunked=True)
        self.assertEqual(req.kv.cache_protected_len, 28)
        self._assert_released(cache, req, runtime)


if __name__ == "__main__":
    unittest.main()
