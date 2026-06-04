"""
FlatMemoryStore - SGLang HiCache storage backend powered by flat_memory_system.

Thin adapter that implements HiCacheStorage interface and delegates all
storage operations to flat_memory_system.manager.FlatMemoryManager.

Usage:
    python -m sglang.launch_server \
        --hicache-storage-backend flat_memory \
        --hicache-storage-backend-extra-config '{
            "policy": "LATENCY_FIRST",
            "dram_capacity_gb": 10,
            "ssd_capacity_gb": 100,
            "ssd_path": "/data2/huangxin/flat_memory_sys_kvcache_storage_dir/experiment_4"
        }'
"""

import logging
import time
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.metrics.collector import StorageMetrics

# Import the independent flat_memory_system library
from flat_memory_system.manager import FlatMemoryConfig, FlatMemoryManager

logger = logging.getLogger(__name__)


class FlatMemoryStore(HiCacheStorage):
    """
    SGLang HiCache storage backend using Flat Memory System.

    Manages DRAM + SSD as a unified flat address space via FlatMemoryManager.
    No tiered eviction. Placement strategy decides where each KVCache page
    is stored at write time.
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig = None,
        mem_pool: HostKVCache = None,
    ):
        # Parse config from SGLang's extra_config
        extra_config = (
            getattr(storage_config, "extra_config", None)
            if storage_config
            else None
        )
        fm_config = FlatMemoryConfig.from_dict(extra_config)

        # Create the core manager (all storage logic lives there)
        self.manager = FlatMemoryManager(fm_config)

        # SGLang-specific config
        if storage_config is not None:
            self.is_mla_backend = storage_config.is_mla_model
            self.local_rank = storage_config.tp_rank
            self.pp_rank = storage_config.pp_rank
            self.pp_size = storage_config.pp_size
        else:
            self.is_mla_backend = False
            self.local_rank = 0
            self.pp_rank = 0
            self.pp_size = 1

        self.enable_pp = self.pp_size > 1
        if self.enable_pp:
            self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"
            self.mla_suffix = f"{self.pp_rank}"
        else:
            self.mha_suffix = f"{self.local_rank}"
            self.mla_suffix = ""

        # Metrics accumulators
        self.gb_per_page = None
        self.prefetch_pgs = []
        self.backup_pgs = []
        self.prefetch_bandwidth = []
        self.backup_bandwidth = []

    # ---- Memory pool registration (same pattern as MooncakeStore) ----

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        assert self.mem_pool_host.layout in [
            "page_first", "page_first_direct", "page_head",
        ], "FlatMemoryStore only supports page_first, page_first_direct, or page_head layout"
        bytes_per_page = mem_pool_host.get_ksize_per_token() * mem_pool_host.page_size
        self.gb_per_page = bytes_per_page / (1 << 30)
        logger.info(
            f"FlatMemoryStore: registered mem_pool_host, "
            f"bytes_per_page={bytes_per_page}, layout={mem_pool_host.layout}"
        )

    # ---- Key encoding helpers (same as MooncakeStore) ----

    def _get_mha_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mha_suffix}_k")
            key_list.append(f"{key_}_{self.mha_suffix}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _get_mla_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mla_suffix}_k")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _batch_preprocess(self, keys, host_indices):
        assert len(keys) > 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size
        if self.is_mla_backend:
            return self._get_mla_buffer_meta(keys, host_indices)
        else:
            return self._get_mha_buffer_meta(keys, host_indices)

    def _batch_postprocess(self, results: List[bool], is_set_operate=False):
        if self.is_mla_backend:
            return results
        else:
            kv_pairs = zip(results[::2], results[1::2])
            return [k and v for k, v in kv_pairs]

    # ---- HiCacheStorage v1 interface (used by HiCacheController) ----

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)
        if key_strs:
            logger.info(
                f"[PREFETCH-DEBUG] batch_set_v1: n_keys={len(key_strs)}, "
                f"first_key={key_strs[0]}"
            )
        results = self.manager.batch_put(key_strs, buffer_ptrs, buffer_sizes)
        return self._batch_postprocess(results, is_set_operate=True)

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)
        results = self.manager.batch_get(key_strs, buffer_ptrs, buffer_sizes)
        return self._batch_postprocess(results, is_set_operate=False)

    # ---- HiCacheStorage legacy interface ----

    def set(
        self,
        key,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        assert target_location is not None and target_sizes is not None
        if self.manager.exists(key):
            return True
        return self.manager.put(key, target_location, target_sizes)

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if len(keys) == 0:
            return False

        # Called from _generic_page_set: batch_set(hash_values, data)
        # where data is a list of torch.Tensor, target_locations/target_sizes are None.
        # In this case, extract pointers and sizes from the tensor list.
        if target_locations is None and values is not None:
            target_locations = [v.data_ptr() for v in values]
            target_sizes = [v.nbytes for v in values]

        assert target_locations is not None and target_sizes is not None
        assert len(keys) == len(target_locations) == len(target_sizes)

        start_time = time.perf_counter()

        # Validate: reject batch if any entry is None
        for i in range(len(keys)):
            if keys[i] is None or target_locations[i] is None or target_sizes[i] is None:
                return False

        # Pass all keys directly to C++ BatchPutCoalesced which handles
        # deduplication internally (single lock), avoiding per-key exists() overhead.
        results = self.manager.batch_put(keys, target_locations, target_sizes)
        all_ok = all(results)
        if len(keys) > 0:
            logger.info(
                f"[PREFETCH-DEBUG] batch_set: n_keys={len(keys)}, "
                f"first_key={keys[0][:32]}, all_ok={all_ok}"
            )

        end_time = time.perf_counter()

        self.backup_pgs.append(len(keys))
        if end_time > start_time:
            self.backup_bandwidth.append(
                len(keys) / (end_time - start_time) * self.gb_per_page
            )

        return all_ok

    def get(
        self,
        key,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        assert target_location is not None and target_sizes is not None
        return self.manager.get(key, target_location, target_sizes) >= 0

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> int:
        assert len(keys) == len(target_locations) == len(target_sizes)
        if len(keys) == 0:
            return 0

        if self.is_mla_backend:
            key_multiplier = 1
        else:
            key_multiplier = 2

        start_time = time.perf_counter()
        results = self.manager.batch_get(keys, target_locations, target_sizes)
        # Find first failure
        for i, ok in enumerate(results):
            if not ok:
                end_time = time.perf_counter()
                return i // key_multiplier
        end_time = time.perf_counter()

        self.prefetch_pgs.append(len(keys))
        if end_time > start_time:
            self.prefetch_bandwidth.append(
                len(keys) / (end_time - start_time) * self.gb_per_page
            )

        return len(keys) // key_multiplier

    def exists(self, key) -> bool:
        return self.manager.exists(key)

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        if self.is_mla_backend:
            query_keys = [f"{key}_{self.mla_suffix}_k" for key in keys]
            key_multiplier = 1
        else:
            query_keys = []
            for key in keys:
                query_keys.append(f"{key}_{self.mha_suffix}_k")
                query_keys.append(f"{key}_{self.mha_suffix}_v")
            key_multiplier = 2

        for i in range(len(query_keys)):
            if not self.manager.exists(query_keys[i]):
                if i == 0:
                    logger.info(
                        f"[PREFETCH-DEBUG] batch_exists: FIRST key miss: {query_keys[0]}, "
                        f"is_mla={self.is_mla_backend}, total_keys={len(query_keys)}, "
                        f"total_blocks={self.manager.get_stats()}"
                    )
                return i // key_multiplier
        return len(query_keys) // key_multiplier

    def close(self):
        self.manager.close()

    def clear(self) -> None:
        self.manager.close()
        # Re-create manager with same config
        self.manager = FlatMemoryManager(self.manager.config)
        logger.info("FlatMemoryStore cleared and re-initialized")

    def get_stats(self):
        storage_metrics = StorageMetrics()
        storage_metrics.prefetch_pgs.extend(self.prefetch_pgs)
        storage_metrics.backup_pgs.extend(self.backup_pgs)
        storage_metrics.prefetch_bandwidth.extend(self.prefetch_bandwidth)
        storage_metrics.backup_bandwidth.extend(self.backup_bandwidth)
        self.prefetch_pgs.clear()
        self.backup_pgs.clear()
        self.prefetch_bandwidth.clear()
        self.backup_bandwidth.clear()
        return storage_metrics
