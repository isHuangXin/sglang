"""FLAT_MEMORY: Payload metering for resolved L2 DMA descriptions, not storage pages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
    LogicalHostPool,
)
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

if TYPE_CHECKING:
    from sglang.srt.mem_cache.l2_transfer import L2Transfer


def _bytes_per_layer(transfer: L2Transfer, *, io_backend: str) -> int | None:
    pool = transfer.host_pool
    if type(pool) is LogicalHostPool:
        return 0
    if type(pool) not in (
        DeepSeekV4PagedHostPool,
        DeepSeekV4StateHostPool,
        MHATokenToKVPoolHost,
        MLATokenToKVPoolHost,
        DSAIndexerPoolHost,
    ):
        return None
    if transfer.host_indices is None or transfer.device_indices is None:
        return None
    num_slots = transfer.host_indices.numel()
    if num_slots != transfer.device_indices.numel():
        return None
    if num_slots == 0:
        return 0
    if io_backend not in ("kernel", "direct") or pool.layout not in (
        "layer_first",
        "page_first",
        "page_first_direct",
    ):
        return None
    if type(pool) is DeepSeekV4PagedHostPool:
        if num_slots % pool.slot_page_size:
            # transfer_cache_dsv4_mla uses kValueBytes + kScaleBytes from
            # kernels/jit/include/sgl_kernel/deepseek_v4/kvcacheio.cuh, not padding.
            return num_slots * (576 + 8)
        return num_slots // pool.slot_page_size * pool.item_bytes * pool.dtype.itemsize
    if type(pool) is DeepSeekV4StateHostPool:
        if num_slots % pool.swa_page_size:
            return None
        return (
            num_slots
            // pool.swa_page_size
            * pool.state_page_bytes
            * pool.dtype.itemsize
        )
    # CP ownership and DCP index expansion are outside this metering capability.
    if (
        pool.dcp_size != 1
        or pool.device_pool is None
        or pool.device_pool.layer_shard_enabled
    ):
        return None
    if type(pool) is DSAIndexerPoolHost:
        if num_slots % pool.page_size:
            return None
        return num_slots // pool.page_size * pool.indexer_page_stride_size
    if io_backend == "direct" and num_slots % pool.page_size:
        return None
    if pool.layer_num == 0 or pool.size_per_token % pool.layer_num:
        return None
    return num_slots * (pool.size_per_token // pool.layer_num)


def l2_transfer_num_bytes(
    transfers: list[L2Transfer], *, io_backend: str, layer_num: int | None = None
) -> int | None:
    """None means unsupported; layer_num=None measures an all-layer D2H submission."""
    total = 0
    primary = transfers[0] if transfers else None
    for transfer in transfers:
        per_layer = _bytes_per_layer(transfer, io_backend=io_backend)
        if per_layer is None:
            return None
        if per_layer == 0:
            continue
        pool = transfer.host_pool
        if layer_num is None:
            total += per_layer * pool.layer_num
            continue
        for layer_id in range(layer_num):
            local_layer = transfer.load_layer_id(
                layer_id, is_primary=transfer is primary
            )
            if local_layer is None:
                continue
            if type(pool) in (
                MHATokenToKVPoolHost,
                MLATokenToKVPoolHost,
                DSAIndexerPoolHost,
            ):
                if not transfer.is_draft and not pool._is_device_layer_owned(
                    transfer.device_pool, local_layer
                ):
                    continue
            if not 0 <= local_layer < pool.layer_num:
                return None
            total += per_layer
    return total
