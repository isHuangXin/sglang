"""Regression tests for physical V4 L2 payloads and resolved layer/sidecar transfers."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2Transfer
from sglang.srt.mem_cache.l2_transfer_metrics import l2_transfer_num_bytes
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
    LogicalHostPool,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def paged_pool(*, layers=5, page_size=256, item_bytes=37440, layout="layer_first"):
    pool = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
    pool.layer_num = layers
    pool.slot_page_size = pool.page_size = page_size
    pool.item_bytes = pool.size_per_token = item_bytes
    pool.dtype = torch.uint8
    pool.layout = layout
    pool.device = "cpu"
    pool.size = page_size * 8
    pool.can_use_write_back_jit = False
    pool.get_page_buffer_meta = Mock(
        side_effect=AssertionError("metadata must not run")
    )
    return pool


def state_pool(*, layers=3, page_size=128, item_bytes=147456):
    pool = DeepSeekV4StateHostPool.__new__(DeepSeekV4StateHostPool)
    pool.layer_num = layers
    pool.swa_page_size = pool.page_size = page_size
    pool.state_page_bytes = pool.size_per_token = item_bytes
    pool.dtype = torch.uint8
    pool.layout = "layer_first"
    pool.can_use_write_back_jit = False
    pool.get_page_buffer_meta = Mock(
        side_effect=AssertionError("metadata must not run")
    )
    return pool


def transfer(pool, slots, *, mapper=None, device_pool=None):
    indices = torch.arange(slots, dtype=torch.int64)
    return L2Transfer(pool, device_pool, indices, indices, mapper)


class TestL2TransferMetrics(CustomTestCase):
    def test_v4_pages_count_physical_rows_and_all_layers(self):
        for layout in ("layer_first", "page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                pool = paged_pool(layout=layout)
                xfer = transfer(pool, 512)
                xfer = xfer._replace(host_indices=xfer.host_indices.reshape(2, 256))
                with patch.object(torch.Tensor, "tolist", side_effect=AssertionError):
                    self.assertEqual(
                        l2_transfer_num_bytes([xfer], io_backend="kernel"), 374400
                    )
                pool.get_page_buffer_meta.assert_not_called()
                self.assertEqual(pool.size_per_token, 37440)

    def test_v4_state_rows_include_dtype_width(self):
        pool = state_pool()
        pool.dtype = torch.int16
        self.assertEqual(
            l2_transfer_num_bytes([transfer(pool, 256)], io_backend="kernel"),
            2 * 3 * 147456 * 2,
        )
        self.assertIsNone(
            l2_transfer_num_bytes([transfer(pool, 255)], io_backend="kernel")
        )
        pool.get_page_buffer_meta.assert_not_called()

    def test_c4_partial_rows_copy_values_and_scales_not_page_padding(self):
        pool = paged_pool()
        for slots in (1, 63, 257):
            with self.subTest(slots=slots):
                xfer = transfer(pool, slots, mapper={0: 1, 2: 4}.get)
                self.assertEqual(
                    l2_transfer_num_bytes([xfer], io_backend="kernel"),
                    slots * 584 * 5,
                )
                self.assertEqual(
                    l2_transfer_num_bytes([xfer], io_backend="kernel", layer_num=4),
                    slots * 584 * 2,
                )

    def test_v4_meter_matches_real_backup_dispatch_arguments(self):
        for pool, slots, per_page_bytes in (
            (paged_pool(), 512, 37440),
            (state_pool(), 256, 147456),
        ):
            with self.subTest(pool=type(pool).__name__):
                pool.device_ptrs = torch.zeros(pool.layer_num, dtype=torch.uint64)
                pool.data_ptrs = torch.zeros(pool.layer_num, dtype=torch.uint64)
                xfer = transfer(pool, slots)
                with patch(
                    "sglang.srt.mem_cache.memory_pool_host.transfer_kv_all_layer_mla",
                    create=True,
                ) as copy_kernel:
                    pool.backup_from_device_all_layer(
                        None, xfer.host_indices, xfer.device_indices, "kernel"
                    )
                args = copy_kernel.call_args.kwargs
                self.assertEqual(args["src_indices"].numel(), 2)
                self.assertEqual(args["item_size"], per_page_bytes)
                self.assertEqual(
                    l2_transfer_num_bytes([xfer], io_backend="kernel"),
                    args["src_indices"].numel()
                    * args["item_size"]
                    * args["num_layers"],
                )

    def test_c4_meter_matches_real_partial_dispatch_arguments(self):
        pool = paged_pool()
        pool.device_ptrs = torch.zeros(pool.layer_num, dtype=torch.uint64)
        pool.data_ptrs = torch.zeros(pool.layer_num, dtype=torch.uint64)
        xfer = transfer(pool, 63, mapper={0: 1, 2: 4}.get)
        with patch(
            "sglang.srt.mem_cache.memory_pool_host.transfer_cache_dsv4_mla"
        ) as copy_kernel:
            pool.backup_from_device_all_layer(
                None, xfer.host_indices, xfer.device_indices, "kernel"
            )
            self.assertEqual(copy_kernel.call_args.kwargs["src_ptrs"].numel(), 5)
            copy_kernel.reset_mock()
            for layer in (1, 4):
                pool.load_to_device_per_layer(
                    None, xfer.host_indices, xfer.device_indices, layer, "kernel"
                )
        copied_bytes = sum(
            call.kwargs["src_indices"].numel() * call.kwargs["src_ptrs"].numel() * 584
            for call in copy_kernel.call_args_list
        )
        self.assertEqual(
            l2_transfer_num_bytes([xfer], io_backend="kernel", layer_num=4),
            copied_bytes,
        )
        self.assertEqual(copied_bytes, 63 * 584 * 2)

    def test_resolved_logical_anchor_sidecars_and_packed_draft(self):
        anchor = LogicalHostPool(2048, 256)
        swa = paged_pool(layers=5, page_size=128, item_bytes=75264)
        c4 = paged_pool(layers=2)
        state = state_pool(layers=2)
        entries = [
            PoolEntry(PoolName.KV, anchor, None, {}.get, is_primary_index_anchor=True),
            PoolEntry(
                PoolName.SWA,
                swa,
                object(),
                {i: i for i in range(5)}.get,
                packed_draft_device_pools=(object(),),
            ),
            PoolEntry(PoolName.DEEPSEEK_V4_C4, c4, None, {1: 0, 3: 1}.get),
            PoolEntry(PoolName.DEEPSEEK_V4_C4_STATE, state, None, {1: 0, 3: 1}.get),
        ]
        group = HostPoolGroup(entries)
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = group
        controller.layer_num = 4
        controller.io_backend = "kernel"
        primary_indices = torch.arange(512)
        swa_indices = torch.arange(256)
        sidecars = [
            PoolTransfer(
                PoolName.SWA, host_indices=swa_indices, device_indices=swa_indices
            ),
            PoolTransfer(PoolName.DEEPSEEK_V4_C4, indices_from_pool=PoolName.KV),
            PoolTransfer(PoolName.DEEPSEEK_V4_C4_STATE, indices_from_pool=PoolName.SWA),
        ]
        resolved = group.resolve_host_transfers(
            sidecars,
            primary_host_indices=primary_indices,
            primary_device_indices=primary_indices,
        )
        writes = controller._l2_transfers(primary_indices, primary_indices, resolved)
        loads = controller._l2_load_transfers(
            primary_indices, primary_indices, resolved
        )
        expected = 2 * 5 * 75264 + 2 * 2 * 37440 + 2 * 2 * 147456
        self.assertEqual(controller._transfer_num_bytes(writes), expected)
        self.assertEqual(controller._transfer_num_bytes(loads, layer_num=4), expected)
        self.assertEqual(len(loads), len(writes) + 1)
        self.assertTrue(loads[-1].is_draft)
        # Skipping a target layer reduces actual H2D even though D2H copies all layers.
        restricted = loads[2]._replace(layer_mapper={1: 0}.get)
        loads[2] = restricted
        self.assertEqual(
            controller._transfer_num_bytes(loads, layer_num=4), expected - 2 * 37440
        )

    def test_mla_and_indexer_count_only_actual_load_layers(self):
        device = SimpleNamespace(layer_num=3, layer_shard_enabled=False)
        mla = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        mla.layer_num = 4
        mla.size_per_token = 4 * 656
        mla.page_size = 64
        mla.layout = "page_first"
        mla.device_pool = device
        indexer = DSAIndexerPoolHost.__new__(DSAIndexerPoolHost)
        indexer.layer_num = 4
        indexer.indexer_page_stride_size = 64 * 132
        indexer.page_size = 64
        indexer.layout = "page_first"
        indexer.device_pool = device
        xfers = [transfer(pool, 128, device_pool=device) for pool in (mla, indexer)]
        self.assertEqual(
            l2_transfer_num_bytes(xfers, io_backend="kernel"), 128 * 4 * (656 + 132)
        )
        self.assertEqual(
            l2_transfer_num_bytes(xfers, io_backend="kernel", layer_num=4),
            128 * 3 * (656 + 132),
        )
        xfers.extend(
            xfer._replace(layer_mapper={0: 3}.get, is_draft=True) for xfer in xfers[:]
        )
        self.assertEqual(
            l2_transfer_num_bytes(xfers, io_backend="kernel", layer_num=4),
            128 * 4 * (656 + 132),
        )

    def test_unknown_pools_and_unprovable_layouts_are_not_zero(self):
        unknown = SimpleNamespace(layout="layer_first", size_per_token=100)
        self.assertIsNone(
            l2_transfer_num_bytes([transfer(unknown, 64)], io_backend="kernel")
        )
        self.assertIsNone(
            l2_transfer_num_bytes(
                [transfer(paged_pool(), 256)], io_backend="kernel_ascend"
            )
        )
        xfer = transfer(paged_pool(), 256)
        self.assertIsNone(
            l2_transfer_num_bytes(
                [xfer._replace(device_indices=torch.arange(255))], io_backend="kernel"
            )
        )
        self.assertIsNone(
            l2_transfer_num_bytes(
                [xfer._replace(layer_mapper=lambda _: 5)],
                io_backend="kernel",
                layer_num=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
