"""Unit tests for host-pool allocation and free-list bookkeeping."""

import threading
import unittest
import unittest.mock
from types import SimpleNamespace

import torch
from sglang.benchmark.native_io_metrics import _validate_host_rank
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
    LogicalHostPool,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry, base
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.observability.hicache_io_metrics import host_pool_capacities
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHostKVCache(CustomTestCase):
    def setUp(self):
        self.page_size = 2
        # Small device pool is enough to construct the host pool.
        self.device_pool = MHATokenToKVPool(
            size=self.page_size * 2,
            page_size=self.page_size,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        self.host_pool = MHATokenToKVPoolHost(
            device_pool=self.device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )

    def test_double_alloc(self):
        indices = self.host_pool.alloc(4)
        self.assertEqual(len(indices), 4)
        # Mimic bookkeeping corruption: push an already-used slot back to the
        # head of free_slots so the next alloc would hand out an in-use slot.
        leak = torch.tensor([int(indices[0])])
        self.host_pool.free_slots = torch.cat([leak, self.host_pool.free_slots])
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.alloc(4)
        msg = str(ctx.exception)
        self.assertIn("Double-alloc", msg)
        self.assertIn(f"[{int(leak[0])}]", msg)

    def test_double_free(self):
        indices = self.host_pool.alloc(4)
        self.assertEqual(len(indices), 4)
        self.host_pool.free(indices[:2])
        # indices[1] is double freed.
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices[1:])
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(f"[{int(indices[1])}]", msg)

    def test_free_unallocated(self):
        indices = torch.tensor([1])
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices)
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(f"[{int(indices[0])}]", msg)

    def test_free_after_clear(self):
        indices = self.host_pool.alloc(4)
        self.host_pool.clear()
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices)
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(str(indices.tolist()), msg)

    def test_shm_allocator(self):
        shm_host_pool = MHATokenToKVPoolHost(
            device_pool=self.device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="shm",
        )
        self.assertIsNotNone(shm_host_pool.fd)
        self.assertGreaterEqual(shm_host_pool.fd, 0)

        indices = shm_host_pool.alloc(4)
        self.assertEqual(len(indices), 4)
        shm_host_pool.free(indices)

    def test_empty_free_keeps_release_list_empty(self):
        self.assertEqual(self.host_pool.free(torch.empty(0, dtype=torch.int64)), 0)
        self.assertEqual(self.host_pool.num_release_slots, 0)
        self.assertEqual(self.host_pool.release_slots, [])

    def test_slot_capacities_deduplicate_shared_pools(self):
        """Repeated ordinary pools retain their unique token-slot capacities."""
        group = HostPoolGroup(
            [
                PoolEntry(
                    name=name,
                    host_pool=self.host_pool,
                    device_pool=self.device_pool,
                    layer_mapper=lambda layer: layer,
                )
                for name in (PoolName.KV, PoolName.SWA)
            ]
        )
        capacities = host_pool_capacities(host_pool=group, device_pool=None)
        self.assertEqual(
            capacities["host_capacity_bytes"],
            self.host_pool.size * self.host_pool.size_per_token,
        )
        self.assertEqual(
            capacities["device_capacity_bytes"],
            self.device_pool.size * self.host_pool.size_per_token,
        )
        self.assertEqual(capacities["bytes_per_token"], self.host_pool.size_per_token)


class TestLazyHostPoolRelease(CustomTestCase):
    @staticmethod
    def _make_mamba_pool():
        pool = MambaPoolHost.__new__(MambaPoolHost)
        pool.size = 8
        pool.page_size = 1
        pool.device = "cpu"
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    @staticmethod
    def _make_deepseek_v4_pool():
        pool = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
        pool.size = 8
        pool.slot_page_size = 2
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    @staticmethod
    def _make_logical_pool():
        return LogicalHostPool(size=8, page_size=2)

    def _assert_lazy_release(self, pool):
        self.assertEqual(pool.free(torch.empty(0, dtype=torch.int64)), 0)
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])

        allocated = pool.alloc(6)
        free_slots_before = pool.free_slots

        pool.free(allocated[:2])

        # free() should keep the primary free-list untouched and only record
        # the released chunk for a later merge.
        self.assertIs(pool.free_slots, free_slots_before)
        self.assertEqual(pool.num_release_slots, 2)
        self.assertEqual(len(pool.release_slots), 1)
        self.assertEqual(pool.available_size(), 4)

        # Consume the primary free-list first without merging pending slots.
        self.assertTrue(torch.equal(pool.alloc(2), torch.tensor([6, 7])))
        self.assertEqual(pool.num_release_slots, 2)

        # Once the primary free-list is exhausted, alloc() merges and reuses
        # the pending slots.
        self.assertTrue(torch.equal(pool.alloc(2), torch.tensor([0, 1])))
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])
        self.assertEqual(pool.available_size(), 0)

        pool.free(torch.tensor([0, 1]))
        pool.clear()
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])
        self.assertEqual(pool.available_size(), 8)

        # Exercise the general merge path with multiple released chunks.
        allocated = pool.alloc(8)
        pool.free(allocated[:2])
        pool.free(allocated[2:4])
        self.assertEqual(len(pool.release_slots), 2)
        self.assertTrue(torch.equal(pool.alloc(4), torch.tensor([0, 1, 2, 3])))
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])

    def test_mamba_pool_lazy_release(self):
        self._assert_lazy_release(self._make_mamba_pool())

    def test_deepseek_v4_pool_lazy_release(self):
        pool = self._make_deepseek_v4_pool()
        self._assert_lazy_release(pool)

        # Preserve the pool's page-aligned allocation behavior.
        pool.clear()
        self.assertEqual(len(pool.alloc(1)), 2)

    def test_logical_pool_lazy_release(self):
        pool = self._make_logical_pool()
        self._assert_lazy_release(pool)

        # Preserve the logical pool's strict page-alignment checks.
        pool.clear()
        with self.assertRaises(ValueError):
            pool.alloc(1)
        with self.assertRaises(ValueError):
            pool.free(torch.tensor([0]))


class TestHostMemoryBudget(CustomTestCase):
    # Pinned so the two budget reads below see identical free memory; the real
    # psutil value drifts between calls and would flake the equality checks.
    _AVAILABLE = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 64 * (1024**3)

    def _budget_with_ranks(self, ranks):
        # Deliberate single-accessor stub: isolates the budget math from the
        # topology derivation, which the ranks_per_host case below covers.
        fake_mem = unittest.mock.Mock(available=self._AVAILABLE)
        with (
            unittest.mock.patch.object(base, "ranks_per_host", return_value=ranks),
            unittest.mock.patch.object(
                base.psutil, "virtual_memory", return_value=fake_mem
            ),
        ):
            return base.host_memory_budget_bytes()

    def test_budget_is_split_across_co_located_ranks(self):
        solo = self._budget_with_ranks(1)
        self.assertEqual(self._budget_with_ranks(4), solo // 4)

    def test_reserve_is_taken_before_the_split(self):
        # Each rank must not get its own copy of the reserve.
        budget = self._budget_with_ranks(8)
        self.assertLessEqual(
            budget * 8, self._AVAILABLE - base.HICACHE_HOST_MEMORY_RESERVE_BYTES
        )

    def test_ranks_per_host_divides_world_size_by_nodes(self):
        # The launcher slices ranks uniformly across nodes, so the co-located
        # rank count is world_size // nnodes — no hostname collective.
        fake_group = unittest.mock.Mock(world_size=16)
        with (
            get_context().override_server_args(nnodes=2),
            unittest.mock.patch.object(
                torch.distributed, "is_initialized", return_value=True
            ),
            unittest.mock.patch.object(
                base, "get_world_group", return_value=fake_group
            ),
        ):
            self.assertEqual(base.ranks_per_host(), 8)


class TestHostPoolGroup(CustomTestCase):
    @staticmethod
    def _group(**sizes):
        return HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName(name),
                    host_pool=LogicalHostPool(size=size, page_size=1),
                    device_pool=None,
                    layer_mapper=lambda layer_id: layer_id,
                    is_primary_index_anchor=name == PoolName.KV.value,
                )
                for name, size in sizes.items()
            ]
        )

    def test_resolve_and_release_multi_pool_allocation(self):
        group = self._group(kv=4, swa=2)
        primary = group.alloc(2)
        transfers = [
            PoolTransfer(name=PoolName.SWA, device_indices=torch.arange(2)),
            PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.SWA),
        ]

        self.assertIsNotNone(
            group.resolve_host_transfers(
                transfers,
                primary_device_indices=torch.arange(2),
                primary_host_indices=primary,
            )
        )
        self.assertIs(transfers[1].host_indices, transfers[0].host_indices)
        group.free(primary)
        group.release_transfers(transfers)
        self.assertEqual(group.available_size(), 4)
        self.assertEqual(group.available_size(PoolName.SWA), 2)

    def test_resolve_rolls_back_partial_allocation(self):
        group = self._group(kv=4, swa=2, mamba=1)
        transfers = [
            PoolTransfer(name=PoolName.SWA, device_indices=torch.arange(2)),
            PoolTransfer(name=PoolName.MAMBA, device_indices=torch.arange(2)),
        ]

        self.assertIsNone(group.resolve_host_transfers(transfers))
        self.assertIsNone(transfers[0].host_indices)
        self.assertEqual(group.available_size(PoolName.SWA), 2)


class TestV4HostPoolCapacities(CustomTestCase):
    @staticmethod
    def _set_host_buffer(pool, *, pages, layers, page_bytes, layout):
        pool.size = pages * 256
        pool.size_per_token = page_bytes
        pool.can_use_write_back_jit = False
        if layout == "layer_first":
            pool.kv_buffer = [
                torch.empty((pages, page_bytes), dtype=torch.uint8)
                for _ in range(layers)
            ]
        else:
            pool.kv_buffer = torch.empty(
                (pages, layers, 1, page_bytes), dtype=torch.uint8
            )

    @classmethod
    def _paged(cls, buffers, *, layout="page_first_direct"):
        pool = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
        pool.device_buffers = buffers
        cls._set_host_buffer(
            pool, pages=3, layers=len(buffers), page_bytes=20, layout=layout
        )
        return pool

    @classmethod
    def _state(cls, *, slots, width, dtype, pages, layout):
        pool = DeepSeekV4StateHostPool.__new__(DeepSeekV4StateHostPool)
        pool.state_pools = [
            SimpleNamespace(
                ring_size=2,
                kv_score_buffer=SimpleNamespace(
                    kv_score=torch.empty((slots, width), dtype=dtype)
                ),
            )
            for _ in range(2)
        ]
        pool.device_page_views = []
        pool._init_device_page_views()
        cls._set_host_buffer(
            pool, pages=pages, layers=2, page_bytes=pool.state_page_bytes, layout=layout
        )
        return pool

    @staticmethod
    def _group(*entries):
        anchor = PoolEntry(
            name=PoolName.KV,
            host_pool=LogicalHostPool(size=1024, page_size=256),
            device_pool=SimpleNamespace(size=1024),
            layer_mapper=lambda layer: layer,
            is_primary_index_anchor=True,
        )
        return HostPoolGroup(
            [anchor]
            + [
                PoolEntry(
                    name=name,
                    host_pool=host,
                    device_pool=device,
                    layer_mapper=lambda layer: layer,
                )
                for name, host, device in entries
            ]
        )

    @staticmethod
    def _rank(capacities):
        return {
            **capacities,
            "pid": 1,
            "generation": 0,
            "tp_rank": 0,
            "tp_size": 1,
            "pending": 0,
            "enabled": True,
            "io_backend": "direct",
            "idle": True,
            **{
                direction: {"bytes": 0, "batches": 0, "elapsed_ms": 0.0}
                for direction in ("read", "write")
            },
        }

    def test_mixed_pools_include_both_states_without_device_pool(self):
        """V4 status includes state pages without dereferencing absent device pools."""
        for layout in ("layer_first", "page_first_direct"):
            with self.subTest(layout=layout):
                paged = self._paged(
                    [torch.empty((7, 20), dtype=torch.uint8) for _ in range(2)],
                    layout=layout,
                )
                state = self._state(
                    slots=7, width=3, dtype=torch.uint8, pages=4, layout=layout
                )
                indexer = self._state(
                    slots=9, width=4, dtype=torch.float16, pages=2, layout=layout
                )
                group = self._group(
                    (PoolName.SWA, paged, SimpleNamespace(size=1536)),
                    (PoolName.DEEPSEEK_V4_C4_STATE, state, None),
                    (PoolName.DEEPSEEK_V4_C4_INDEXER_STATE, indexer, None),
                )
                capacities = host_pool_capacities(host_pool=group, device_pool=None)
                # Host: 120 + 48 + 64. Device: 280 + 36 + 128 (complete rings).
                self.assertEqual(capacities["host_capacity_bytes"], 232)
                self.assertEqual(capacities["device_capacity_bytes"], 444)
                self.assertEqual(capacities["bytes_per_token"], 0)
                _validate_host_rank(self._rank(capacities), tp_size=1)

    def test_unified_views_exclude_unregistered_storage_and_duplicate_aliases(self):
        """Shared storage counts only registered ranges, including each alias once."""
        storage = torch.empty((11, 20), dtype=torch.uint8)
        compressed = storage[2:9]
        paged = self._paged([compressed, storage[9:]])
        alias = self._paged([compressed.view_as(compressed)])
        group = self._group(
            (PoolName.SWA, paged, None),
            (PoolName.INDEXER, alias, None),
            (PoolName.DEEPSEEK_V4_C4_STATE, paged, None),
        )
        capacities = host_pool_capacities(host_pool=group, device_pool=None)
        self.assertEqual(capacities["host_capacity_bytes"], 180)
        self.assertEqual(capacities["device_capacity_bytes"], 180)

    def test_benchmark_accepts_zero_anchor_but_rejects_invalid_capacities(self):
        """A payload-free anchor is valid; missing capacity and malformed values are not."""
        rank = self._rank(
            {
                "host_capacity_bytes": 232,
                "device_capacity_bytes": 444,
                "bytes_per_token": 0,
            }
        )
        _validate_host_rank(rank, tp_size=1)
        for field, value in (
            ("bytes_per_token", -1),
            ("bytes_per_token", True),
            ("bytes_per_token", 1.5),
            ("host_capacity_bytes", 0),
            ("device_capacity_bytes", 0),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                _validate_host_rank({**rank, field: value}, tp_size=1)


if __name__ == "__main__":
    unittest.main()
