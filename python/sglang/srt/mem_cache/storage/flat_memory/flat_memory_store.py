"""Flat Memory host-storage adapter; direct GPU transfers use the Flat linker."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.storage.flat_memory.payload import restorable_boundaries

logger = logging.getLogger(__name__)


def make_storage_metrics(manager, prefetch_samples, backup_samples):
    from sglang.srt.observability.metrics_collector import StorageMetrics

    metrics = StorageMetrics()
    for samples, pages, bandwidth in (
        (prefetch_samples, metrics.prefetch_pgs, metrics.prefetch_bandwidth),
        (backup_samples, metrics.backup_pgs, metrics.backup_bandwidth),
    ):
        for count, size, elapsed in samples:
            pages.append(count)
            if elapsed > 0:
                bandwidth.append(size / (1 << 30) / elapsed)
    report = manager.get_bandwidth_report()
    stats = manager.get_stats()
    report.update(
        {
            "dram_used_bytes": stats["dram_used"],
            "ssd_used_bytes": stats["ssd_used"],
            "total_blocks": stats["total_blocks"],
        }
    )
    report.update(manager.get_capacity_stats())
    metrics.flat_memory_bandwidth = report
    return metrics


class FlatMemoryStore(HiCacheStorage):
    def __init__(
        self, storage_config: HiCacheStorageConfig | None = None, mem_pool=None
    ):
        from flat_memory_system.manager import FlatMemoryConfig, FlatMemoryManager

        extra = storage_config.extra_config if storage_config is not None else None
        config = FlatMemoryConfig.from_dict(extra)
        if config.gds_mode == "compat":
            raise ValueError(
                "Flat gds_mode=compat must use the direct Flat radix backend, not a host pool"
            )
        if storage_config is not None and storage_config.should_split_heads:
            raise ValueError(
                "Flat host storage does not support heterogeneous-TP split heads"
            )
        self.is_mla_backend = storage_config.is_mla_model if storage_config else False
        self.local_rank = storage_config.tp_rank if storage_config else 0
        self.tp_size = storage_config.tp_size if storage_config else 1
        self.pp_rank = storage_config.pp_rank if storage_config else 0
        self.pp_size = storage_config.pp_size if storage_config else 1
        self.enable_pp = self.pp_size > 1
        self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"
        self.mla_suffix = self.mha_suffix
        identity = {
            "format": 1,
            "model": storage_config.model_name if storage_config else None,
            "tp_rank": self.local_rank,
            "tp_size": self.tp_size,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "cp_rank": storage_config.attn_cp_rank if storage_config else 0,
            "cp_size": storage_config.attn_cp_size if storage_config else 1,
            "tag": (extra or {}).get("extra_backend_tag"),
        }
        self._namespace = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        # FLAT_MEMORY: Namespaced keys do not isolate rank-local backing files.
        config.ssd_path = os.path.join(
            config.ssd_path,
            f"tp_rank_{self.local_rank}",
            f"pp_rank_{self.pp_rank}_cp_rank_{identity['cp_rank']}",
        )
        self._manager_factory = lambda: FlatMemoryManager(config)
        self.manager = self._manager_factory()
        self.mem_pool_host = None
        self.registered_pools = {}
        self._pool_sizes: dict[PoolName, tuple[int, ...]] = {}
        self._pool_tags: dict[PoolName, str] = {}
        self._metrics_lock = threading.Lock()
        self._prefetch_samples = []
        self._backup_samples = []
        self.gb_per_page = None
        if mem_pool is not None:
            self.register_mem_pool_host(mem_pool)

    @property
    def gpu_direct(self) -> bool:
        return False

    @staticmethod
    def _flatten_meta(pointers, sizes):
        flat_pointers, flat_sizes = [], []
        for pointer, size in zip(pointers, sizes):
            if isinstance(pointer, (list, tuple)):
                if not isinstance(size, (list, tuple)) or len(pointer) != len(size):
                    raise ValueError("Flat host buffer pointer/size vectors disagree")
                flat_pointers.extend(pointer)
                flat_sizes.extend(size)
            else:
                flat_pointers.append(pointer)
                flat_sizes.append(size)
        if len(pointers) != len(sizes):
            raise ValueError("Flat host buffer metadata lengths disagree")
        return flat_pointers, flat_sizes

    def register_mem_pool_host(self, mem_pool_host):
        super().register_mem_pool_host(mem_pool_host)
        self.register_mem_host_pool_v2(mem_pool_host, PoolName.KV)
        self.gb_per_page = sum(self._pool_sizes[PoolName.KV]) / (1 << 30)

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        name = PoolName(host_pool_name)
        if host_pool.kv_buffer is None:
            self.registered_pools[name] = host_pool
            self._pool_sizes[name] = ()
            self._pool_tags[name] = "logical"
            return
        pointers, sizes = host_pool.get_page_buffer_meta(
            torch.arange(host_pool.page_size, dtype=torch.int64)
        )
        pointers, sizes = self._flatten_meta(pointers, sizes)
        if not pointers or any(size <= 0 for size in sizes):
            raise ValueError(f"Flat host pool {name} has an empty physical page")
        self.registered_pools[name] = host_pool
        self._pool_sizes[name] = tuple(sizes)
        descriptor = (
            type(host_pool).__name__,
            host_pool.page_size,
            str(host_pool.dtype),
            host_pool.layout,
            sizes,
        )
        self._pool_tags[name] = hashlib.sha256(
            json.dumps(descriptor).encode()
        ).hexdigest()[:24]

    def _keys(self, *, keys: list[str], name: PoolName) -> list[str]:
        return [
            f"flat-host-v1:{self._namespace}:{name}:{self._pool_tags[name]}:b{component}:{key}"
            for key in keys
            for component in range(len(self._pool_sizes[name]))
        ]

    def _metadata(self, *, keys, indices, name):
        pool = self.registered_pools[name]
        if len(keys) * pool.page_size != len(indices):
            raise ValueError("Flat host page keys and token indices disagree")
        if not self._pool_sizes[name]:
            return [], [], []
        pointers, sizes = self._flatten_meta(*pool.get_page_buffer_meta(indices))
        expected = list(self._pool_sizes[name]) * len(keys)
        if sizes != expected or len(pointers) != len(expected):
            raise ValueError("Flat host pool layout changed after registration")
        return self._keys(keys=keys, name=name), pointers, sizes

    def _batch_io(self, *, keys, indices, name, write):
        if not keys:
            return []
        objects, pointers, sizes = self._metadata(keys=keys, indices=indices, name=name)
        if not objects:
            return [True] * len(keys)
        started = time.perf_counter()
        method = self.manager.batch_put if write else self.manager.batch_get
        results = method(objects, pointers, sizes)
        width = len(self._pool_sizes[name])
        if len(results) != len(objects):
            raise RuntimeError("Flat native host I/O returned an invalid result length")
        page_results = [
            all(results[index : index + width])
            for index in range(0, len(results), width)
        ]
        completed = next(
            (index for index, ok in enumerate(page_results) if not ok),
            len(page_results),
        )
        elapsed = time.perf_counter() - started
        with self._metrics_lock:
            samples = self._backup_samples if write else self._prefetch_samples
            samples.append(
                (completed, sum(self._pool_sizes[name]) * completed, elapsed)
            )
        return page_results

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        return self._batch_io(
            keys=keys, indices=host_indices, name=PoolName.KV, write=True
        )

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        return self._batch_io(
            keys=keys, indices=host_indices, name=PoolName.KV, write=False
        )

    def batch_set_v2(self, transfers: list[PoolTransfer], extra_info=None):
        return {
            transfer.name: self._batch_io(
                keys=transfer.keys,
                indices=transfer.host_indices,
                name=transfer.name,
                write=True,
            )
            for transfer in transfers
        }

    def batch_get_v2(self, transfers: list[PoolTransfer], extra_info=None):
        return {
            transfer.name: self._batch_io(
                keys=transfer.keys,
                indices=transfer.host_indices,
                name=transfer.name,
                write=False,
            )
            for transfer in transfers
        }

    def _page_exists(self, *, keys, name):
        width = len(self._pool_sizes[name])
        if not width:
            return [True] * len(keys)
        query = self._keys(keys=keys, name=name)
        addresses = self.manager.lookup_addresses(query)
        if len(addresses) != len(query):
            raise RuntimeError("Flat host lookup returned an invalid result length")
        return [
            all(addresses[index : index + width])
            for index in range(0, len(addresses), width)
        ]

    def batch_exists(self, keys, extra_info=None):
        exists = self._page_exists(keys=keys, name=PoolName.KV)
        return next(
            (index for index, present in enumerate(exists) if not present), len(keys)
        )

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        candidates = set(range(1, self.batch_exists(keys) + 1))
        hits = {}
        for transfer in pool_transfers or []:
            states = self._page_exists(keys=keys, name=transfer.name)
            boundaries = restorable_boundaries(
                states,
                policy=transfer.hit_policy,
                trailing_pages=max(1, len(transfer.keys or [])),
            )
            candidates &= boundaries
            hits[transfer.name] = max(boundaries, default=0)
        restorable = sorted(candidates)
        return PoolTransferResult(restorable[-1] if restorable else 0, hits, restorable)

    def batch_set_direct(self, keys, buffer_ptrs, buffer_sizes):
        # FLAT_MEMORY: Split packed host pages by real component sizes, not equal K/V halves.
        components = self._pool_sizes[PoolName.KV]
        expected = sum(components)
        if len(keys) != len(buffer_ptrs) or len(keys) != len(buffer_sizes):
            raise ValueError("Flat direct host page metadata lengths disagree")
        pointers, sizes = [], []
        for pointer, size in zip(buffer_ptrs, buffer_sizes):
            if size != expected:
                raise ValueError(
                    "Flat direct host page size differs from the registered layout"
                )
            offset = 0
            for component in components:
                pointers.append(pointer + offset)
                sizes.append(component)
                offset += component
        results = self.manager.batch_put(
            self._keys(keys=keys, name=PoolName.KV), pointers, sizes
        )
        width = len(components)
        if len(results) != len(pointers):
            raise RuntimeError("Flat direct host write returned invalid results")
        return [
            all(results[index : index + width])
            for index in range(0, len(results), width)
        ]

    def set(self, key, value=None, target_location=None, target_sizes=None):
        if target_location is None and isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("Flat ordinary writes require host tensors")
            target_location, target_sizes = value.data_ptr(), value.nbytes
        if target_location is None or target_sizes is None:
            raise ValueError("Flat set requires a source pointer and byte size")
        return self.manager.put(key, target_location, target_sizes)

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        if not keys:
            return False
        if target_locations is None and values is not None:
            if any(value.device.type != "cpu" for value in values):
                raise ValueError("Flat ordinary writes require host tensors")
            target_locations = [value.data_ptr() for value in values]
            target_sizes = [value.nbytes for value in values]
        if target_locations is None or target_sizes is None:
            raise ValueError("Flat batch_set requires source pointers and sizes")
        if len(keys) != len(target_locations) or len(keys) != len(target_sizes):
            raise ValueError("Flat batch_set vector lengths disagree")
        results = self.manager.batch_put(keys, target_locations, target_sizes)
        if len(results) != len(keys):
            raise RuntimeError("Flat batch_set returned an invalid result length")
        return all(results)

    def get(self, key, target_location=None, target_sizes=None):
        if target_location is None or target_sizes is None:
            raise ValueError("Flat get requires a destination pointer and byte size")
        return self.manager.get(key, target_location, target_sizes) >= 0

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        if target_locations is None or target_sizes is None:
            raise ValueError("Flat batch_get requires destination pointers and sizes")
        if len(keys) != len(target_locations) or len(keys) != len(target_sizes):
            raise ValueError("Flat batch_get vector lengths disagree")
        results = self.manager.batch_get(keys, target_locations, target_sizes)
        if len(results) != len(keys):
            raise RuntimeError("Flat batch_get returned an invalid result length")
        first_failure = next(
            (index for index, ok in enumerate(results) if not ok), len(results)
        )
        return first_failure // (1 if self.is_mla_backend else 2)

    def exists(self, key):
        return self.manager.exists(key)

    def close(self):
        if self.manager is not None:
            self.manager.close()
            self.manager = None

    def clear(self):
        self.close()
        self.manager = self._manager_factory()

    def get_stats(self):
        with self._metrics_lock:
            prefetch, self._prefetch_samples = self._prefetch_samples, []
            backup, self._backup_samples = self._backup_samples, []
        return make_storage_metrics(self.manager, prefetch, backup)

    def get_flat_memory_stats(self) -> dict[str, Any]:
        return {
            "storage": self.manager.get_stats(),
            "bandwidth": self.manager.get_bandwidth_report(),
            "capacity": self.manager.get_capacity_stats(),
            "cio": self.manager.get_cio_stats(),
            "io_window": self.manager.get_io_window(),
        }
