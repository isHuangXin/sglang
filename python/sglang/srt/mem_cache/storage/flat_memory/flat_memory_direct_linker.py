"""Rank-local Flat storage linked directly to unified cache device pools."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import replace
from typing import Any

import msgspec
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolGroup
from sglang.srt.mem_cache.storage.flat_memory.payload import GPUTransfer, PayloadLayout
from sglang.srt.mem_cache.storage.flat_memory.ssd_executor import SSDReadExecutor
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker

logger = logging.getLogger(__name__)


class FlatLoadResult(msgspec.Struct, frozen=True):
    rid: str
    error: str | None = None
    medium: str = "dram"
    elapsed_seconds: float = 0.0
    dram_pages: int = 0
    ssd_pages: int = 0
    mixed_pages: int = 0


class LayerLoadCounter:
    """Completion gate; Flat publishes all layers only after the full read succeeds."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, Future] = {}
        self._lock = threading.Lock()

    def update_producer(self) -> int:
        with self._lock:
            self.producer_index += 1
            self.futures[self.producer_index] = Future()
            return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, error: str | None) -> None:
        with self._lock:
            future = self.futures[index]
            if error is None:
                future.set_result(None)
            else:
                future.set_exception(RuntimeError(error))

    def wait_until(self, threshold: int) -> None:
        with self._lock:
            future = self.futures.get(self.consumer_index)
        if future is not None:
            future.result()
            if threshold == self.num_layers - 1:
                with self._lock:
                    self.futures.pop(self.consumer_index, None)

    def forget(self, index: int) -> None:
        with self._lock:
            future = self.futures.get(index)
            if future is not None and future.done():
                self.futures.pop(index)

    def reset(self) -> None:
        with self._lock:
            self.futures.clear()
            self.producer_index = self.consumer_index = -1


class FlatMemoryDirectLinker(UnifiedCacheLinker):
    def __init__(
        self,
        *,
        pool_group: DevicePoolGroup,
        config: dict,
        model_namespace: str,
        tp_rank: int,
        tp_size: int,
        device: torch.device | str,
        manager: Any = None,
    ):
        if config.get("gds_mode", "compat") != "compat":
            raise ValueError("Flat direct linker requires gds_mode=compat")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            # FLAT_MEMORY: A fresh worker thread otherwise resolves plain 'cuda' as device 0.
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type != "cuda" and manager is None:
            raise ValueError("Flat direct linker requires a CUDA device")
        self.pool_group = pool_group
        self.pools = pool_group.entry_map
        self.page_size = pool_group.page_size
        self.num_layers = pool_group.num_layers
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self._config = dict(config)
        self._closed = False
        self._owns_manager = manager is None
        if manager is None:
            from flat_memory_system.manager import FlatMemoryConfig, FlatMemoryManager

            native_config = FlatMemoryConfig.from_dict(config)
            native_config.gds_mode = "compat"
            native_config.gpu_id = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            # FLAT_MEMORY: Replicated MLA still has a private store on every TP rank.
            native_config.ssd_path = os.path.join(
                native_config.ssd_path, f"tp_rank_{tp_rank}"
            )
            self._manager_factory = lambda: FlatMemoryManager(native_config)
            manager = self._manager_factory()
        else:
            self._manager_factory = None
        self.manager = manager
        self.layout = PayloadLayout(
            pool_group=pool_group,
            model_namespace=model_namespace,
            tp_rank=tp_rank,
            tp_size=tp_size,
            max_io_bytes=int(config.get("gds_max_io_bytes", 16 * 1024 * 1024)),
        )
        self.transfer = GPUTransfer(
            layout=self.layout,
            manager=self.manager,
            device=self.device,
            staging_bytes=int(config.get("gpu_stage_bytes", 128 * 1024 * 1024)),
            batch_pages=int(config.get("gpu_batch_pages", 128)),
        )
        self.backup_wait_seconds = float(config.get("backup_wait_seconds", 10.0))
        self.drain_timeout = float(config.get("drain_timeout", 30.0))
        if self.backup_wait_seconds < 0 or self.drain_timeout <= 0:
            raise ValueError(
                "Flat backup wait must be nonnegative and drain timeout positive"
            )
        self.layer_done_counter = LayerLoadCounter(self.num_layers)
        self._lookup_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="flat-query"
        )
        self._dispatch_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="flat-dispatch"
        )
        self._dram_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="flat-dram"
        )
        self._ssd_executor = SSDReadExecutor(workers=2)
        self._write_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="flat-write"
        )
        self._pending_loads: dict[str, list[PoolTransfer]] = {}
        self._loads: dict[str, Future] = {}
        self._lookups: dict[str, Future] = {}
        self._query_jobs: set[Future] = set()
        self._load_batches: deque[tuple[int, dict[str, Future]]] = deque()
        self._offloads: deque[Future] = deque()
        self._prefetch_samples = []
        self._backup_samples = []
        self._lock = threading.RLock()

    @property
    def storage(self):
        return self

    @property
    def gpu_direct(self) -> bool:
        return True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Flat direct linker is closed")

    @staticmethod
    def _snapshot(transfers: list[PoolTransfer]) -> list[PoolTransfer]:
        # FLAT_MEMORY: Worker metadata must not alias a scheduler's mutable key lists.
        return [
            replace(transfer, keys=list(transfer.keys or [])) for transfer in transfers
        ]

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        self._check_open()
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        return self.transfer.lookup(keys=list(kv.keys or []), transfers=expanded)

    def submit_lookup(self, rid: str, transfers: list[PoolTransfer]) -> Future:
        self._check_open()
        snapshot = self._snapshot(transfers)
        with self._lock:
            if rid in self._lookups:
                return self._lookups[rid]
            offloads = tuple(self._offloads)
            future = self._lookup_executor.submit(
                self._lookup_after_backup, rid, snapshot, offloads
            )
            self._lookups[rid] = future
            self._query_jobs.add(future)
            future.add_done_callback(self._query_finished)
            return future

    def _query_finished(self, future: Future) -> None:
        with self._lock:
            self._query_jobs.discard(future)

    def _lookup_after_backup(self, rid: str, transfers, offloads) -> list[int]:
        result = self.lookup(rid, transfers)
        kv = next((item for item in transfers if item.name == PoolName.KV), None)
        complete = kv is not None and result and result[-1] == len(kv.keys or [])
        pending = [future for future in offloads if not future.done()]
        if not complete and pending and self.backup_wait_seconds:
            wait(pending, timeout=self.backup_wait_seconds)
            result = self.lookup(rid, transfers)
        return result

    def release_lookup(self, rid: str) -> None:
        with self._lock:
            future = self._lookups.pop(rid, None)
        if future is not None:
            future.cancel()

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        self._check_open()
        expanded = self.pool_group.resolve_transfers(
            self._snapshot(transfers), allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        with self._lock:
            if rid in self._pending_loads or rid in self._loads:
                raise RuntimeError(f"Flat load for {rid!r} is already queued")
            self._pending_loads[rid] = expanded
        return True

    def cancel_private_load(self, rid: str) -> bool:
        with self._lock:
            return self._pending_loads.pop(rid, None) is not None

    def cancel_queued_load(self, rid: str) -> bool:
        # FLAT_MEMORY: The generic wrapper has already published slots at this point.
        return False

    def _ready_event(self):
        if self.device.type != "cuda":
            return None
        with torch.cuda.device(self.device):
            event = torch.cuda.Event()
            event.record()
        return event

    def start_preparing_loads(self) -> int:
        self._check_open()
        with self._lock:
            if not self._pending_loads:
                return -1
            pending, self._pending_loads = self._pending_loads, {}
            counter_index = self.layer_done_counter.update_producer()
            futures = {rid: Future() for rid in pending}
            self._loads.update(futures)
            self._load_batches.append((counter_index, dict(futures)))
            ready_event = self._ready_event()
            for rid, transfers in pending.items():
                self._dispatch_executor.submit(
                    self._dispatch_read,
                    rid,
                    transfers,
                    ready_event,
                    futures[rid],
                    counter_index,
                    futures,
                )
            return counter_index

    def start_layer_wise_loading(self) -> int:
        return self.start_preparing_loads()

    def _media_counts(self, transfers) -> tuple[str, int, int, int]:
        media: dict[str, set[int]] = {}
        for transfer in transfers:
            keys = transfer.keys or []
            addresses = self.transfer.page_addresses(name=transfer.name, keys=keys)
            for key, page in zip(keys, addresses):
                if not page or not all(page):
                    raise RuntimeError(
                        "Flat GPU restore lost a required payload fragment"
                    )
                media.setdefault(key, set()).update(address >> 60 for address in page)
        dram = sum(kinds == {0} for kinds in media.values())
        ssd = sum(1 in kinds for kinds in media.values())
        mixed = sum(0 in kinds and 1 in kinds for kinds in media.values())
        if any(kinds - {0, 1} for kinds in media.values()):
            raise RuntimeError("Flat compat cannot read remote/unknown storage media")
        return ("ssd" if ssd else "dram", dram, ssd, mixed)

    def _dispatch_read(self, rid, transfers, ready_event, result, counter_index, batch):
        try:
            medium, dram, ssd, mixed = self._media_counts(transfers)
            args = (rid, transfers, ready_event, medium, dram, ssd, mixed)
            if medium == "ssd":
                read = self._ssd_executor.submit(
                    self._read, *args, io_bytes=self._payload_bytes(transfers)
                )
            else:
                read = self._dram_executor.submit(self._read, *args)
            read.add_done_callback(
                lambda done: self._read_finished(
                    done, rid, result, counter_index, batch
                )
            )
        except Exception as error:
            self._finish_read(
                FlatLoadResult(rid, str(error)), result, counter_index, batch
            )

    def _read_finished(self, done, rid, result, counter_index, batch):
        try:
            value = done.result()
        except BaseException as error:
            value = FlatLoadResult(rid, f"{type(error).__name__}: {error}")
        self._finish_read(value, result, counter_index, batch)

    def _read(
        self, rid, transfers, ready_event, medium, dram, ssd, mixed
    ) -> FlatLoadResult:
        started = time.perf_counter()
        failure = None
        try:
            if ready_event is not None:
                ready_event.synchronize()
            self.transfer.read(transfers)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        elapsed = time.perf_counter() - started
        if failure is None:
            with self._lock:
                self._prefetch_samples.append(
                    (dram + ssd, self._payload_bytes(transfers), elapsed)
                )
        return FlatLoadResult(rid, failure, medium, elapsed, dram, ssd, mixed)

    def _payload_bytes(self, transfers) -> int:
        return sum(
            len(transfer.keys or [])
            * sum(spec.length for spec in self.layout.specs[transfer.name])
            for transfer in transfers
        )

    def _finish_read(self, value, result, counter_index, batch) -> None:
        with self._lock:
            result.set_result(value)
            if all(future.done() for future in batch.values()):
                errors = [
                    future.result().error
                    for future in batch.values()
                    if future.result().error
                ]
                self.layer_done_counter.complete(
                    counter_index, "; ".join(errors) if errors else None
                )

    def load_ready(self, rid: str) -> bool:
        with self._lock:
            future = self._loads.get(rid)
        if future is None or not future.done():
            return False
        result = future.result()
        if result.error:
            raise RuntimeError(result.error)
        return True

    def get_load_error(self, rid: str) -> str | None:
        with self._lock:
            future = self._loads.get(rid)
        return future.result().error if future is not None and future.done() else None

    def take_load_result(self, rid: str) -> FlatLoadResult | None:
        with self._lock:
            future = self._loads.get(rid)
            if future is None or not future.done():
                return None
            result = future.result()
            self._loads.pop(rid)
            kept = deque()
            for counter_index, batch in self._load_batches:
                batch.pop(rid, None)
                if batch:
                    kept.append((counter_index, batch))
                else:
                    self.layer_done_counter.forget(counter_index)
            self._load_batches = kept
        return result

    def forget_request(self, rid: str) -> None:
        self.release_lookup(rid)
        self.take_load_result(rid)

    def num_completed_loads(self) -> int:
        with self._lock:
            count = 0
            for _, batch in self._load_batches:
                if not all(future.done() for future in batch.values()):
                    break
                count += 1
            return count

    def pop_completed_load_result(self) -> list[FlatLoadResult]:
        with self._lock:
            if not self._load_batches or not all(
                future.done() for future in self._load_batches[0][1].values()
            ):
                raise RuntimeError("No Flat load batch has completed")
            _, batch = self._load_batches.popleft()
            for rid in batch:
                self._loads.pop(rid, None)
            return [future.result() for future in batch.values()]

    def pop_completed_load(self) -> list[str]:
        results = self.pop_completed_load_result()
        errors = [result.error for result in results if result.error]
        if errors:
            raise RuntimeError("; ".join(errors))
        return [result.rid for result in results]

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        self._check_open()
        expanded = self.pool_group.resolve_transfers(
            self._snapshot(transfers), allow_partial=True
        )
        if not expanded:
            return False
        ready_event = self._ready_event()
        with self._lock:
            self._offloads.append(
                self._write_executor.submit(self._write, expanded, ready_event)
            )
        return True

    def _write(self, transfers, ready_event) -> bool:
        started = time.perf_counter()
        try:
            if ready_event is not None:
                ready_event.synchronize()
            self.transfer.write(transfers)
            pages = len({key for transfer in transfers for key in transfer.keys or []})
            with self._lock:
                self._backup_samples.append(
                    (
                        pages,
                        self._payload_bytes(transfers),
                        time.perf_counter() - started,
                    )
                )
            return True
        except Exception:
            logger.exception("Flat GPU offload failed")
            return False

    def num_completed_offloads(self) -> int:
        with self._lock:
            count = 0
            for future in self._offloads:
                if not future.done():
                    break
                count += 1
            return count

    def pop_completed_offload(self) -> bool:
        with self._lock:
            if not self._offloads or not self._offloads[0].done():
                raise RuntimeError("No Flat offload has completed")
            return self._offloads.popleft().result()

    def has_unfinished_io(self) -> bool:
        # FLAT_MEMORY: Ready or failed futures need retirement, not an idle delay.
        with self._lock:
            return any(
                not future.done()
                for jobs in (self._query_jobs, self._loads.values(), self._offloads)
                for future in jobs
            )

    def drain(self, timeout: float | None = None) -> bool:
        self.start_preparing_loads()
        with self._lock:
            futures = (
                list(self._loads.values())
                + list(self._offloads)
                + list(self._query_jobs)
            )
        _, incomplete = wait(
            futures, timeout=self.drain_timeout if timeout is None else timeout
        )
        return not incomplete

    def reset(self) -> None:
        if not self.drain():
            raise TimeoutError("Flat GPU I/O did not drain; resources remain owned")
        with self._lock:
            self._loads.clear()
            self._lookups.clear()
            self._offloads.clear()
            self._load_batches.clear()
            self.layer_done_counter.reset()

    def close(self) -> None:
        if self._closed:
            return
        self.reset()
        for executor in (
            self._lookup_executor,
            self._dispatch_executor,
            self._dram_executor,
            self._ssd_executor,
            self._write_executor,
        ):
            executor.shutdown(wait=True)
        self._closed = True
        self.manager.close()
        # FLAT_MEMORY: The Python manager's close() only logs; drop both native owners.
        if self._owns_manager:
            self.transfer.manager = None
            self.manager = None

    def clear(self) -> None:
        self.reset()
        if self._manager_factory is None:
            raise RuntimeError("An injected Flat manager cannot be recreated")
        self.manager.close()
        self.transfer.manager = None
        self.manager = None
        self.manager = self._manager_factory()
        self.transfer.manager = self.manager

    def get_prefetch_stats(self, rid: str) -> dict | None:
        with self._lock:
            future = self._loads.get(rid)
        if future is None or not future.done():
            return None
        result = future.result()
        if result.error:
            raise RuntimeError(result.error)
        return {
            "dram_tokens": result.dram_pages * self.page_size,
            "ssd_tokens": result.ssd_pages * self.page_size,
            "mixed_tokens": result.mixed_pages * self.page_size,
            "dram_elapsed_seconds": (
                result.elapsed_seconds if result.medium == "dram" else 0.0
            ),
            "ssd_elapsed_seconds": (
                result.elapsed_seconds if result.medium == "ssd" else 0.0
            ),
            "dram_ops": int(result.medium == "dram"),
            "ssd_ops": int(result.medium == "ssd"),
        }

    def get_stats(self):
        from sglang.srt.mem_cache.storage.flat_memory.flat_memory_store import (
            make_storage_metrics,
        )

        with self._lock:
            prefetch, self._prefetch_samples = self._prefetch_samples, []
            backup, self._backup_samples = self._backup_samples, []
        return make_storage_metrics(self.manager, prefetch, backup)

    def get_flat_memory_stats(self) -> dict:
        self._check_open()
        return {
            "storage": self.manager.get_stats(),
            "bandwidth": self.manager.get_bandwidth_report(),
            "capacity": self.manager.get_capacity_stats(),
            "cio": self.manager.get_cio_stats(),
            "io_window": self.manager.get_io_window(),
            "transfer": self.transfer.get_stats(),
        }
