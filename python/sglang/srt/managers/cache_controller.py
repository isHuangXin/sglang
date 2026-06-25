from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import os
import threading
import time
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)

device_module = get_device_module()


class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()  # start event on controller stream

    def complete(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        return self.load_events[-1]


class LayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self):
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ].finish_event.query(), (
            "Producer finish event should be ready before being reused."
        )
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1


class CacheOperation:

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.data = None

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        assert len(ops) > 0
        if len(ops) == 1:
            return ops[0]

        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged_op = CacheOperation(host_indices, device_indices, -1, priority)
        merged_op.node_ids = node_ids
        return merged_op

    def __lt__(self, other: CacheOperation):
        return self.priority < other.priority


class HiCacheAck(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    node_ids: List[int]


class TransferBuffer:
    """
    Overlapping buffer preparation and transfer operations to improve throughput.
    """

    def __init__(
        self, stop_event, buffer_count: int = 3, max_buffer_size: int = 1024
    ) -> None:
        self.stop_event = stop_event
        self.buffers = Queue(maxsize=buffer_count)
        # todo: adjust the buffer size based on throughput profile of the system
        self.max_buffer_size = max_buffer_size

    def full(self) -> bool:
        return self.buffers.full()

    def empty(self) -> bool:
        return self.buffers.empty()

    def put(self, item, block=True, timeout=1) -> None:
        while not self.stop_event.is_set():
            try:
                self.buffers.put(item, block=block, timeout=timeout)
                break
            except Full:
                if not block:
                    break
                continue
            except Exception as e:
                logger.error(e)

    def get(self, block=True, timeout=1) -> Optional[CacheOperation]:
        try:
            return self.buffers.get(block=block, timeout=timeout)
        except Empty:
            return None
        except Exception as e:
            logger.error(e)

    def clear(self):
        self.buffers.queue.clear()


class StorageOperation:
    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        full_token_ids: Optional[List[int]] = None,
    ):
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.last_hash = last_hash
        self.completed_tokens = 0
        self.hash_value = hash_value if hash_value is not None else []
        self.prefix_keys = prefix_keys
        # Complete token sequence from the start of the request, used for
        # fallback hash-chain reconstruction when the suffix-only query misses.
        self.full_token_ids = full_token_ids

        self.id = StorageOperation.counter
        StorageOperation.counter += 1

    def __lt__(self, other: "StorageOperation"):
        return self.id < other.id


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        full_token_ids: Optional[List[int]] = None,
    ):
        self.request_id = request_id

        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()
        self.start_time_perf = time.perf_counter()  # FLAT_MEMORY: high-res timer for per-phase timing

        super().__init__(host_indices, token_ids, last_hash, prefix_keys=prefix_keys,
                         full_token_ids=full_token_ids)

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


class HiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = int(os.environ.get("SGLANG_PREFETCH_THRESHOLD", "256")),
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        pp_rank: int = 0,
        pp_size: int = 1,
    ):
        self.tp_group = tp_group
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend
        self.enable_storage = False
        self.storage_backend = None
        self.storage_backend_type = None
        self.pp_rank = pp_rank
        self.pp_size = pp_size

        # Default storage page IO functions (may be overridden by attach).
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set

        # Dedicated stop event for storage background threads (prefetch/backup).
        # NOTE: Do NOT reuse `self.stop_event` here since it also guards core HiCache
        # transfer buffers (CPU<->GPU). We want to allow runtime attach/detach of
        # storage without stopping the whole controller.
        self.storage_stop_event = threading.Event()

        self.device = self.mem_pool_device.device
        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        self.ack_load_queue: List[HiCacheAck] = []
        self.ack_write_queue: List[HiCacheAck] = []

        self.stop_event = threading.Event()
        self.write_buffer = TransferBuffer(self.stop_event)
        self.load_buffer = TransferBuffer(
            self.stop_event, buffer_count=10, max_buffer_size=100
        )

        self.write_stream = device_module.Stream()
        self.load_stream = device_module.Stream()

        # If a storage backend is provided at startup, treat it as an implicit attach,
        # so init/runtime share the same lifecycle semantics and code paths.
        if storage_backend is not None:
            try:
                self.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=model_name,
                    storage_backend_extra_config=storage_backend_extra_config,
                )
            except ValueError as e:
                # Preserve the historical error shape on init for unknown backends.
                raise ValueError(f"Failed to create storage backend: {e}") from e

    def _start_storage_threads(self):
        """Start storage prefetch/backup threads and their queues.

        This is used by runtime attach, and also by reset when storage is enabled.
        """
        assert self.enable_storage
        assert not self.storage_stop_event.is_set()

        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_queue = Queue()
        self.backup_queue = Queue()
        self.backup_idle_event = threading.Event()
        self.backup_idle_event.set()  # initially idle

        # Track the full GPU→Host→Storage pipeline, not just backup_queue
        self.pending_backup_count = 0
        self.pending_backup_lock = threading.Lock()

        self.prefetch_revoke_queue = Queue()
        self.ack_backup_queue = Queue()
        self.host_mem_release_queue = Queue()

        self.prefetch_thread.start()
        self.backup_thread.start()

    def _stop_storage_threads(self):
        """Stop storage prefetch/backup threads and drain internal queues.

        Caller should ensure no in-flight requests.
        """
        # Always request stop. This is safe even when storage is already disabled,
        # and makes detach truly idempotent (previous partial detach may have left
        # threads alive).
        # NOTE: do NOT clear stop_event unless threads have fully stopped; otherwise
        # a still-alive thread may resume and touch released state.
        self.storage_stop_event.set()

        # Best-effort wakeups so threads exit promptly even if blocked on queues.
        try:
            if hasattr(self, "prefetch_queue"):
                self.prefetch_queue.put_nowait(None)
            if hasattr(self, "backup_queue"):
                self.backup_queue.put_nowait(None)
            # FLAT_MEMORY: wake all IO workers (each may be blocked on prefetch_buffer)
            if hasattr(self, "prefetch_buffer"):
                num_io_workers = len(getattr(self, "prefetch_io_aux_threads", []))
                for _ in range(max(num_io_workers, 1)):
                    self.prefetch_buffer.put_nowait(None)
        except Exception:
            pass

        # Best-effort joins (threads are daemon, but join keeps state clean).
        threads = []
        if hasattr(self, "prefetch_thread"):
            threads.append(self.prefetch_thread)
        if hasattr(self, "backup_thread"):
            threads.append(self.backup_thread)
        # FLAT_MEMORY: join all prefetch IO aux threads (may be 1 or N)
        if hasattr(self, "prefetch_io_aux_threads"):
            threads.extend(self.prefetch_io_aux_threads)
        elif hasattr(self, "prefetch_io_aux_thread"):
            threads.append(self.prefetch_io_aux_thread)

        for t in threads:
            try:
                t.join(timeout=10)
            except Exception:
                pass

        alive = [t for t in threads if getattr(t, "is_alive", lambda: False)()]
        if alive:
            logger.error(
                "Failed to stop HiCache storage threads cleanly: %s",
                [getattr(t, "name", repr(t)) for t in alive],
            )
            raise RuntimeError("Failed to stop HiCache storage threads cleanly.")

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = int(os.environ.get("SGLANG_PREFETCH_THRESHOLD", "256")),
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        """Attach (enable) storage backend at runtime.

        Requirement: no in-flight requests. This call is expected to run on the scheduler
        thread (control path), not concurrently with prefetch/backup.
        """
        if self.enable_storage:
            raise RuntimeError("Storage backend already attached.")

        # Defensive: a previous partial detach may have flipped `enable_storage` but
        # left background threads alive. Attaching on top of them is unsafe.
        try:
            self._stop_storage_threads()
        except Exception as e:
            raise RuntimeError(
                "Cannot attach storage backend: previous detach did not stop storage threads cleanly."
            ) from e

        # Rollback-safe init: if creation fails, keep controller state consistent
        # for future attach attempts.
        self.storage_backend_type = storage_backend
        from sglang.srt.mem_cache.hicache_storage import get_hash_str

        self.get_hash_str = get_hash_str
        self.storage_config = self._generate_storage_config(
            model_name, storage_backend_extra_config
        )
        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = (
            self.storage_config.is_mla_model
            # todo: load balancing
            and self.storage_config.tp_rank != 0
        )

        # Use storage backend factory for dynamic backend creation
        from sglang.srt.mem_cache.storage import StorageBackendFactory

        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, self.mem_pool_host
            )
            self.storage_backend.register_mem_pool_host(self.mem_pool_host)

            self.enable_storage = True
            # todo: threshold policy for prefetching
            # Allow runtime override via env var (useful for testing storage cache hits)
            env_threshold = int(os.environ.get("SGLANG_PREFETCH_THRESHOLD", str(prefetch_threshold)))
            self.prefetch_threshold = max(env_threshold, self.page_size)
            logger.info(f"Storage prefetch threshold set to {self.prefetch_threshold} tokens (env={env_threshold}, arg={prefetch_threshold}, page_size={self.page_size})")
            self.prefetch_capacity_limit = max(
                0, int(0.8 * (self.mem_pool_host.size - self.mem_pool_device.size))
            )
            # FLAT_MEMORY: Separate read and write batch sizes.
            # - READ batch size: large (8192) so C++ BatchReadByKeys sees all ~15 buckets
            #   in a single call, enabling parallel pread via io_pool_ (4-5 GB/s).
            # - WRITE batch size: small (512) so each batch_set_v1 call creates a small
            #   bucket file (~32-64MB), matching Mooncake's design of many small files.
            # Other backends use the same batch size for both read and write.
            if self.storage_backend_type == "flat_memory":
                self.storage_batch_size = int(
                    os.environ.get("SGLANG_STORAGE_READ_BATCH_SIZE", "8192")
                )
                self.storage_write_batch_size = int(
                    os.environ.get("SGLANG_STORAGE_WRITE_BATCH_SIZE", "512")
                )
            else:
                self.storage_batch_size = int(
                    os.environ.get("SGLANG_STORAGE_BATCH_SIZE", "512")
                )
                self.storage_write_batch_size = self.storage_batch_size
            logger.info(
                f"Storage batch size: read={self.storage_batch_size}, write={self.storage_write_batch_size} pages "
                f"(backend={self.storage_backend_type})"
            )
            # tracking the number of tokens locked in prefetching, updated by the main scheduler thread
            self.prefetch_tokens_occupied = 0

            # create a new communication group for synchronizing storage operations across TP workers
            self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
            if self.tp_world_size > 1:
                from sglang.srt.distributed.parallel_state import (
                    create_custom_parallel_group,
                )

                group_ranks = torch.distributed.get_process_group_ranks(self.tp_group)
                self.prefetch_tp_group = create_custom_parallel_group(
                    group_ranks=group_ranks, backend="gloo"
                )

            # Select the get and set functions
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set

            if (self.storage_backend_type in ["hf3fs", "mooncake", "eic", "nixl", "flat_memory"]) or (
                self.storage_backend_type == "dynamic"
                and bool(self.storage_config.extra_config.get("interface_v1", 0))
            ):
                self.page_get_func = self._page_get_zero_copy
                self.page_set_func = self._page_set_zero_copy

            # Ensure stop_event is clear before starting threads.
            self.storage_stop_event.clear()
            self._start_storage_threads()
        except Exception:
            # Best-effort cleanup for partial init.
            try:
                self._stop_storage_threads()
            except Exception:
                pass
            try:
                if hasattr(self, "prefetch_tp_group"):
                    try:
                        torch.distributed.destroy_process_group(self.prefetch_tp_group)
                    except Exception:
                        pass
                    self.prefetch_tp_group = None
            except Exception:
                pass
            try:
                if (
                    hasattr(self, "storage_backend")
                    and self.storage_backend is not None
                ):
                    if hasattr(self.storage_backend, "close"):
                        self.storage_backend.close()
            except Exception:
                pass
            self.storage_backend = None
            self.storage_backend_type = None
            self.enable_storage = False
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set
            raise

    def detach_storage_backend(self):
        """Detach (disable) storage backend at runtime.

        Requirement: no in-flight requests. This will stop storage threads and release
        the backend instance (best-effort close).
        """
        # Idempotent cleanup: even if `enable_storage` is already False,
        # we may still have leftover resources (threads/backend/process group) from a
        # previous partial detach. We attempt cleanup whenever possible.
        try:
            self._stop_storage_threads()
        except Exception as e:
            # Do not proceed tearing down backend/process group if threads are not
            # fully stopped; otherwise still-alive threads may touch released state.
            # Caller can retry detach.
            logger.exception("Stop storage threads failed: %s", e)
            # IMPORTANT: Do not silently succeed. Upper layers rely on exceptions here
            # to avoid flipping `enable_storage` flags while threads are still alive.
            raise RuntimeError("Stop storage threads failed; detach aborted.") from e

        # Best-effort destroy process group created for storage ops.
        try:
            if (
                hasattr(self, "prefetch_tp_group")
                and self.prefetch_tp_group is not None
            ):
                try:
                    torch.distributed.destroy_process_group(self.prefetch_tp_group)
                except Exception:
                    pass
                self.prefetch_tp_group = None
        except Exception:
            pass

        # Best-effort close (some backends rely on GC/destructor).
        try:
            if (
                hasattr(self, "storage_backend")
                and self.storage_backend is not None
                and hasattr(self.storage_backend, "close")
            ):
                self.storage_backend.close()
        except Exception:
            logger.exception("Failed to close storage backend cleanly.")

        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage = False
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set
        # Now it's safe to clear the stop event for future re-attach.
        self.storage_stop_event.clear()

    def _generate_storage_config(
        self,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):

        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.dp_rank = 0

        # Currently, NPUMLATokenToKVPool is the subclass of MLATokenToKVPool.
        is_mla_backend = isinstance(self.mem_pool_device, MLATokenToKVPool)

        return HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            is_mla_model=is_mla_backend,
            is_page_first_layout=self.mem_pool_host.layout == "page_first",
            model_name=model_name,
            extra_config=storage_backend_extra_config,
        )

    def reset(self):
        self.stop_event.set()
        self.storage_stop_event.set()

        self.write_queue.clear()
        self.load_queue.clear()
        self.write_buffer.clear()
        self.load_buffer.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        if self.enable_storage:
            self.prefetch_thread.join()
            self.backup_thread.join()
            # FLAT_MEMORY: join all prefetch IO aux threads
            if hasattr(self, "prefetch_io_aux_threads"):
                for t in self.prefetch_io_aux_threads:
                    t.join(timeout=10)
            self.prefetch_queue.queue.clear()
            self.backup_queue.queue.clear()
            self.prefetch_revoke_queue.queue.clear()
            self.ack_backup_queue.queue.clear()

        self.stop_event.clear()
        self.storage_stop_event.clear()

        if self.enable_storage:
            self.prefetch_thread = threading.Thread(
                target=self.prefetch_thread_func, daemon=True
            )
            self.backup_thread = threading.Thread(
                target=self.backup_thread_func, daemon=True
            )
            self.prefetch_thread.start()
            self.backup_thread.start()

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(op)
        self.write_queue.clear()

        start_event = device_module.Event()
        finish_event = device_module.Event()

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )
            finish_event.record()
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices

    def move_indices(self, op: CacheOperation):
        host_indices, device_indices = op.host_indices, op.device_indices
        # move indices to GPU if using kernels, to host if using direct indexing
        if self.io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(self.device, non_blocking=True)
            return host_indices, device_indices
        elif self.io_backend == "direct":
            if self.mem_pool_host.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)
            elif self.mem_pool_host.layout == "page_first_direct":
                return host_indices, device_indices.cpu()
        elif self.io_backend == "kernel_ascend":
            return host_indices, device_indices.cpu()
        else:
            raise ValueError(f"Unsupported io backend")

    def start_loading(self) -> int:
        if len(self.load_queue) == 0:
            return -1

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(op)
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                )
                producer_event.complete(i)
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the load stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.load_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.load_stream)

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=producer_event.start_event,
                finish_event=producer_event.finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id

    def evict_device(self, device_indices: torch.Tensor) -> int:
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices)

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        self.mem_pool_host.free(host_indices)
        return len(host_indices)

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        full_token_ids: Optional[List[int]] = None,
    ) -> PrefetchOperation:
        """
        Prefetch KV caches from storage backend to host memory.
        """
        operation = PrefetchOperation(
            request_id, host_indices, new_input_tokens, last_hash, prefix_keys,
            full_token_ids=full_token_ids
        )
        self.prefetch_queue.put(operation)
        return operation

    def terminate_prefetch(self, operation):
        operation.mark_terminate()
        return operation.completed_tokens, operation.hash_value

    def append_host_mem_release(self, host_indices: torch.Tensor):
        if host_indices.numel() == 0:
            return
        pages = host_indices.split(self.mem_pool_host.page_size)
        for page in pages:
            self.host_mem_release_queue.put(page)

    def _page_get_zero_copy(
        self, operation, hash_values, host_indices, extra_info=None
    ):
        results = self.storage_backend.batch_get_v1(
            hash_values, host_indices, extra_info
        )
        inc = 0
        for i in range(len(hash_values)):
            if not results[i]:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            inc += self.page_size
        operation.increment(inc)

    # todo: deprecate
    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):
        dummy_page_dst = [
            self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values
        ]
        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
        if page_data is None:
            return
        for i in range(len(hash_values)):
            if page_data[i] is None:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            # Must set the data before increasing the completed tokens.
            # Otherwise this page may be read before being set.
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * self.page_size],
                page_data[i],
            )
            if not operation.increment(self.page_size):
                break  # Operation terminated by controller

    def _page_transfer(self, operation):
        # Transfer batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), self.storage_batch_size):
            batch_hashes = operation.hash_value[i : i + self.storage_batch_size]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            prev_completed_tokens = operation.completed_tokens
            # Get one batch token, and update the completed_tokens if succeed
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)
            # Check termination
            if (
                operation.completed_tokens
                != prev_completed_tokens + len(batch_hashes) * self.page_size
            ):
                operation.mark_terminate()
                break  # Some operations fail or operation terminated by controller

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

    def prefetch_io_aux_func(self):
        """
        Auxiliary function conducting IO operations for prefetching.
        """
        while not self.storage_stop_event.is_set():
            try:
                t_io_dequeue = time.perf_counter()
                operation = self.prefetch_buffer.get(block=True, timeout=1)
                if operation is None:
                    continue
                # FLAT_MEMORY: measure IO buffer queue wait
                io_queue_wait_ms = (t_io_dequeue - operation.start_time_perf) * 1000 if hasattr(operation, 'start_time_perf') else -1

                t_io_start = time.perf_counter()
                self._page_transfer(operation)
                t_io_end = time.perf_counter()
                io_transfer_ms = (t_io_end - t_io_start) * 1000
                total_elapsed_ms = (t_io_end - operation.start_time_perf) * 1000 if hasattr(operation, 'start_time_perf') else -1

                # FLAT_MEMORY: log IO phase timing
                if len(operation.hash_value) > 10:
                    logger.info(
                        f"[PREFETCH-TIMING] io_aux: req={operation.request_id[:8]}, "
                        f"total_elapsed={total_elapsed_ms:.1f}ms, "
                        f"io_queue_wait={io_queue_wait_ms:.1f}ms, "
                        f"io_transfer={io_transfer_ms:.1f}ms, "
                        f"pages={len(operation.hash_value)}, "
                        f"completed={operation.completed_tokens}"
                    )

                # operation terminated by controller, release pre-allocated memory
                self.append_host_mem_release(
                    operation.host_indices[operation.completed_tokens :]
                )
            except Empty:
                continue

    def prefetch_rate_limited(self) -> bool:
        """
        Rate limit the prefetching operations to avoid overwhelming the storage backend.
        """
        # cancel prefetch if too much memory is occupied
        if self.prefetch_tokens_occupied >= self.prefetch_capacity_limit:
            return True
        # todo: more sophisticated rate limiting based on storage backend performance
        return False

    def _do_storage_query(
        self, last_hash, tokens_to_fetch, prefix_keys
    ) -> tuple[list[str], int]:
        """Core storage query: compute hash chain and check batch_exists."""
        storage_query_count = 0
        hash_value = []

        # FLAT_MEMORY: per-phase timing instrumentation
        t_hash_total = 0.0
        t_exists_total = 0.0
        n_hash_calls = 0
        n_exists_calls = 0

        for start in range(
            0, len(tokens_to_fetch), self.page_size * self.storage_batch_size
        ):
            end = min(
                start + self.page_size * self.storage_batch_size, len(tokens_to_fetch)
            )
            batch_tokens = tokens_to_fetch[start:end]
            batch_hashes = []
            t0_hash = time.perf_counter()
            for i in range(0, len(batch_tokens), self.page_size):
                last_hash = self.get_hash_str(
                    batch_tokens[i : i + self.page_size], last_hash
                )
                batch_hashes.append(last_hash)
            t1_hash = time.perf_counter()
            t_hash_total += (t1_hash - t0_hash)
            n_hash_calls += len(batch_hashes)

            t0_exists = time.perf_counter()
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)
            t1_exists = time.perf_counter()
            t_exists_total += (t1_exists - t0_exists)
            n_exists_calls += len(batch_hashes)

            hash_value.extend(batch_hashes[:hit_page_num])
            storage_query_count += hit_page_num * self.page_size
            if hit_page_num < len(batch_hashes):
                break
            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

        # FLAT_MEMORY: log per-phase timing breakdown
        if len(tokens_to_fetch) > 256:
            logger.info(
                f"[PREFETCH-TIMING] _do_storage_query: n_tokens={len(tokens_to_fetch)}, "
                f"hash_compute={t_hash_total*1000:.1f}ms ({n_hash_calls} calls), "
                f"batch_exists={t_exists_total*1000:.1f}ms ({n_exists_calls} keys), "
                f"hit_count={storage_query_count}"
            )

        return hash_value, storage_query_count

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        last_hash = operation.last_hash
        tokens_to_fetch = operation.token_ids
        prefix_keys = operation.prefix_keys.copy() if operation.prefix_keys else None

        if tokens_to_fetch:
            first_hash_preview = self.get_hash_str(
                tokens_to_fetch[: self.page_size], last_hash
            )
            logger.info(
                f"[PREFETCH-DEBUG] _storage_hit_query: first_hash={first_hash_preview[:16]}, "
                f"n_tokens={len(tokens_to_fetch)}, last_hash_in={last_hash[:16] if last_hash else 'None'}, "
                f"first_tokens={tokens_to_fetch[:8]}"
            )

        # Primary query: from the radix cache match point (suffix only)
        hash_value, storage_hit_count = self._do_storage_query(
            last_hash, tokens_to_fetch, prefix_keys
        )

        # Fallback: if suffix query missed and we have the full token sequence,
        # retry from the beginning of the sequence (last_hash=None).
        # This handles the case where Round 1 backed up with a different radix
        # tree structure (different node split points / request ordering).
        if (
            storage_hit_count < self.page_size
            and operation.full_token_ids is not None
            and last_hash is not None  # only fallback when we started mid-chain
        ):
            prefix_len = len(operation.full_token_ids) - len(tokens_to_fetch)
            logger.info(
                f"[PREFETCH-FALLBACK] Suffix query missed, retrying from sequence start. "
                f"prefix_len={prefix_len}, full_len={len(operation.full_token_ids)}"
            )
            full_hash_value, full_hit_count = self._do_storage_query(
                None, operation.full_token_ids, None
            )
            if full_hit_count > prefix_len:
                # Storage has data beyond the radix-cached prefix.
                # Return only the suffix portion (skip prefix pages).
                skip_pages = prefix_len // self.page_size
                hash_value = full_hash_value[skip_pages:]
                storage_hit_count = full_hit_count - prefix_len
                logger.info(
                    f"[PREFETCH-FALLBACK] Hit! full_hit={full_hit_count} tokens, "
                    f"suffix_hit={storage_hit_count} tokens (skipped {prefix_len} prefix tokens)"
                )

        return hash_value, storage_hit_count

    def prefetch_thread_func(self):
        """
        Coordinator: sets up shared state, spawns query workers and IO workers.

        FLAT_MEMORY: The original single-thread design is the #1 bottleneck.
        Each storage query takes ~1s (hash + exists), so with concurrency=4,
        requests queue behind each other adding 0-11s of wait.
        Fix: spawn N query workers that pull from prefetch_queue in parallel.
        """
        self.prefetch_buffer = Queue()
        self.storage_prefetch_queries = 0
        self.storage_prefetch_hits = 0
        self.storage_prefetch_tokens_hit = 0
        # FLAT_MEMORY: lock for shared stats counters across query workers
        self._prefetch_stats_lock = threading.Lock()

        # FLAT_MEMORY: spawn multiple IO workers for parallel SSD prefetch reads.
        num_io_workers = int(os.environ.get("SGLANG_PREFETCH_IO_WORKERS", "8"))
        self.prefetch_io_aux_threads = []
        for i in range(num_io_workers):
            t = threading.Thread(
                target=self.prefetch_io_aux_func,
                daemon=True,
                name=f"prefetch-io-{i}",
            )
            t.start()
            self.prefetch_io_aux_threads.append(t)

        # FLAT_MEMORY: spawn N query workers to parallelize hash+exists queries.
        # Each query takes ~1s (SHA256 hash chain + batch_exists Python→C++ loop).
        # With 1 worker and concurrency=4, average queue wait is 2.2s (max 11s).
        # With 4 workers, queue wait drops to ~0.3s, cutting SSD→Host latency 5x.
        num_query_workers = int(os.environ.get("SGLANG_PREFETCH_QUERY_WORKERS", "4"))
        self.prefetch_query_threads = []
        for i in range(num_query_workers):
            t = threading.Thread(
                target=self._prefetch_query_worker,
                daemon=True,
                name=f"prefetch-query-{i}",
            )
            t.start()
            self.prefetch_query_threads.append(t)

        logger.info(
            f"[PREFETCH] Started {num_query_workers} prefetch query workers "
            f"(env SGLANG_PREFETCH_QUERY_WORKERS) + "
            f"{num_io_workers} IO workers (env SGLANG_PREFETCH_IO_WORKERS)"
        )

        # Coordinator sleeps until stop; actual work is in query workers.
        while not self.storage_stop_event.is_set():
            self.storage_stop_event.wait(timeout=1)

    def _prefetch_query_worker(self):
        """
        Worker thread: dequeue from prefetch_queue → hash+exists → prefetch_buffer.

        FLAT_MEMORY: Multiple instances run in parallel to eliminate the
        single-thread queuing bottleneck that dominated SSD→Host latency.
        """
        worker_name = threading.current_thread().name
        while (not self.storage_stop_event.is_set()) or not self.prefetch_queue.empty():
            try:
                t_dequeue_start = time.perf_counter()
                operation = self.prefetch_queue.get(block=True, timeout=1)
                t_dequeue_end = time.perf_counter()
                if operation is None:
                    continue
                # FLAT_MEMORY: measure queue wait time (from PrefetchOperation creation to dequeue)
                queue_wait_ms = (t_dequeue_end - operation.start_time_perf) * 1000 if hasattr(operation, 'start_time_perf') else -1
                # FLAT_MEMORY: Reduced backup wait timeout from 10s to 0.1s.
                # Original 10s wait blocks ALL prefetch reads until ALL backup writes
                # finish, adding ~2000ms to SSD→Host latency when concurrent writes
                # are in flight. With Flat Memory's C++ BatchPutCoalesced(), data is
                # visible in the block_index_ immediately after put() returns, and
                # SSDBucketBackend's pending_ buffer handles reads of not-yet-flushed
                # data. A short 100ms wait is sufficient to let the backup thread pick
                # up new operations without starving prefetch reads.
                t_backup_wait_start = time.perf_counter()
                self.backup_idle_event.wait(timeout=0.1)
                t_backup_wait_end = time.perf_counter()
                backup_wait_ms = (t_backup_wait_end - t_backup_wait_start) * 1000

                t_query_start = time.perf_counter()
                hash_value, storage_hit_count = self._storage_hit_query(operation)
                t_query_end = time.perf_counter()
                query_ms = (t_query_end - t_query_start) * 1000

                # FLAT_MEMORY: log per-phase timing for prefetch query worker
                if len(operation.token_ids) > 256:
                    logger.info(
                        f"[PREFETCH-TIMING] {worker_name}: req={operation.request_id[:8]}, "
                        f"queue_wait={queue_wait_ms:.1f}ms, "
                        f"backup_wait={backup_wait_ms:.1f}ms, "
                        f"storage_query={query_ms:.1f}ms, "
                        f"hit_count={storage_hit_count}, n_tokens={len(operation.token_ids)}"
                    )
                logger.info(
                    f"[PREFETCH-DEBUG] storage_hit_query: req={operation.request_id[:8]}, "
                    f"hit_count={storage_hit_count}, threshold={self.prefetch_threshold}, "
                    f"total_tokens={len(operation.token_ids)}, hash_count={len(hash_value)}"
                )
                if self.tp_world_size > 1:
                    storage_hit_count_tensor = torch.tensor(
                        storage_hit_count, dtype=torch.int
                    )
                    torch.distributed.all_reduce(
                        storage_hit_count_tensor,
                        op=torch.distributed.ReduceOp.MIN,
                        group=self.prefetch_tp_group,
                    )
                    storage_hit_count = storage_hit_count_tensor.item()

                # FLAT_MEMORY: thread-safe stats update
                with self._prefetch_stats_lock:
                    self.storage_prefetch_queries += 1
                    local_queries = self.storage_prefetch_queries
                    if storage_hit_count < self.prefetch_threshold:
                        pass  # stats only, decision below
                    else:
                        self.storage_prefetch_hits += 1
                        self.storage_prefetch_tokens_hit += storage_hit_count
                        local_hits = self.storage_prefetch_hits
                        local_tokens_hit = self.storage_prefetch_tokens_hit

                if storage_hit_count < self.prefetch_threshold:
                    # not to prefetch if not enough benefits
                    self.prefetch_revoke_queue.put(operation.request_id)
                    self.append_host_mem_release(operation.host_indices)
                    logger.info(
                        f"Revoking prefetch for request {operation.request_id} due to insufficient hits ({storage_hit_count})."
                    )
                else:
                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]
                    # free the pre-allocated memory for pages that are not hit
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                    operation.host_indices = operation.host_indices[:storage_hit_count]
                    logger.info(
                        f"[STORAGE-HIT] Prefetching {len(operation.hash_value)} pages "
                        f"({storage_hit_count} tokens) from storage for req {operation.request_id[:8]}. "
                        f"Total: {local_hits}/{local_queries} queries hit, "
                        f"{local_tokens_hit} tokens prefetched from storage."
                    )
                    self.prefetch_buffer.put(operation)

            except Empty:
                continue

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        """
        Write KV caches from host memory to storage backend.
        """
        operation = StorageOperation(
            host_indices, token_ids, hash_value=hash_value, prefix_keys=prefix_keys
        )
        self.backup_queue.put(operation)
        return operation.id

    # todo: deprecate
    def _generic_page_set(self, hash_values, host_indices, extra_info=None) -> bool:
        data = [
            self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        return self.storage_backend.batch_set(hash_values, data)

    def _page_set_zero_copy(self, hash_values, host_indices, extra_info=None) -> bool:
        return all(
            self.storage_backend.batch_set_v1(hash_values, host_indices, extra_info)
        )

    # Backup batch by batch
    def _page_backup(self, operation):
        # Backup batch by batch
        # FLAT_MEMORY: Use storage_write_batch_size (512) for writes to keep bucket
        # files small (~32-64MB each), matching Mooncake's design. The read path uses
        # the larger storage_batch_size (8192) for parallel SSD reads.
        prefix_keys = operation.prefix_keys
        write_batch_size = self.storage_write_batch_size
        logger.info(
            f"[PREFETCH-DEBUG] _page_backup: n_hashes={len(operation.hash_value)}, "
            f"first_hash={operation.hash_value[0][:16] if operation.hash_value else 'EMPTY'}, "
            f"host_indices_len={len(operation.host_indices)}, write_batch_size={write_batch_size}"
        )
        for i in range(0, len(operation.hash_value), write_batch_size):
            batch_hashes = operation.hash_value[i : i + write_batch_size]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            # Set one batch token, and record if success.
            # todo: allow partial success
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            success = self.page_set_func(batch_hashes, batch_host_indices, extra_info)
            if not success:
                logger.warning(
                    f"Write page to storage: {len(batch_hashes)} pages failed."
                )
                break

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * len(batch_hashes)

    def backup_thread_func(self):
        """
        Manage backup operations from host memory to storage backend.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue

                self.backup_idle_event.clear()
                if not self.backup_skip:
                    self._page_backup(operation)
                self.ack_backup_queue.put(operation)
                # Decrement pipeline counter and signal idle only when fully drained
                with self.pending_backup_lock:
                    self.pending_backup_count -= 1
                    if self.pending_backup_count == 0 and self.backup_queue.empty():
                        self.backup_idle_event.set()

            except Empty:
                # Only set idle if no pending operations in the full pipeline
                with self.pending_backup_lock:
                    if self.pending_backup_count == 0:
                        self.backup_idle_event.set()
                continue
