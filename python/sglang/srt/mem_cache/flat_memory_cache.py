# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned transactions for Flat Memory device-prefix reuse."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.unified_cache.components import LinkerTransferPhase
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    PreparedLinkerLoad,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.observability.flat_memory_metrics import initialize_flat_request

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixKey

logger = logging.getLogger(__name__)


def request_storage_namespace(key: RadixKey) -> str:
    # FLAT_MEMORY: Token hash chains alone do not isolate cache salts or extra keys.
    identity = json.dumps([key.extra_key, key.cache_salt], separators=(",", ":"))
    return "request-v1-" + hashlib.sha256(identity.encode()).hexdigest() + ":"


class _Prefetch(msgspec.Struct, kw_only=True):
    req: Any
    key: Any
    result: Any
    tail_hashes: list[str]
    future: Future | None = None
    prepared: PreparedLinkerLoad | None = None
    phase: str = "query"
    hold_node: Any = None
    hold_params: Any = None
    cancelled: bool = False
    loaded_tokens: int = 0
    started: float = 0.0
    swa_evicted_before: int = 0


class FlatMemoryCache:
    """Compose the upstream tree linker with asynchronous, unpublished loads.

    All allocation, tree mutation and collectives run on the scheduler thread.
    Workers may only query storage or copy into private device allocations.
    """

    def __init__(self, *, cache, cache_linker, config: dict, ready_fcfs: bool = True):
        self.cache = cache
        self.ready_fcfs = ready_fcfs
        self.cache_linker = cache_linker
        self.tree_linker = UnifiedCacheLinkerWrapper(cache, cache_linker)
        self.prefetches: dict[str, _Prefetch] = {}
        self.arrivals: dict[str, int] = {}
        self._next_arrival = 0
        self.io_errors = 0
        self._last_metrics_log = 0.0
        threshold_override = envs.SGLANG_PREFETCH_THRESHOLD.get()
        self.prefetch_threshold = int(
            config.get("prefetch_threshold", 256)
            if threshold_override is None
            else threshold_override
        )
        self.max_prefetch_tokens = int(config.get("prefetch_max_tokens", 65536))
        self.drain_timeout = float(config.get("drain_timeout", 30.0))
        if self.prefetch_threshold < 0 or self.max_prefetch_tokens < cache.page_size:
            raise ValueError("Invalid Flat prefetch threshold or token budget")
        if self.drain_timeout <= 0:
            raise ValueError("Flat drain_timeout must be positive")

    @property
    def manager(self):
        return self.cache_linker.manager

    @property
    def layer_done_counter(self):
        return self.cache_linker.layer_done_counter

    @property
    def pending_offloads(self):
        return self.tree_linker.pending_offloads

    def register_request(self, req: Req) -> None:
        if req.rid not in self.arrivals:
            self.arrivals[req.rid] = self._next_arrival
            self._next_arrival += 1
        if not req.flat_storage_backend:
            initialize_flat_request(
                req, page_size=self.cache.page_size, bytes_per_page=None
            )

    def order_ready_requests(self, requests: list) -> list:
        # FLAT_MEMORY: I/O completion order must not replace request arrival order.
        if not self.ready_fcfs:
            return requests
        return sorted(requests, key=lambda req: self.arrivals[req.rid])

    def has_hit(self, rid: str) -> bool:
        # Completed Flat reads are already device hits; admission never queues DMA.
        return False

    def match(self, key: RadixKey, req: Req, result: MatchResult) -> MatchResult:
        self.register_request(req)
        if (
            req.rid in self.prefetches
            or (req.kv is not None and req.kv.req_pool_idx is not None)
            or envs.SGLANG_RADIX_FORCE_MISS.get()
        ):
            return result
        device_hit_len = int(result.device_indices.numel())
        namespace = request_storage_namespace(key)
        tail_hashes = [
            namespace + value
            for value in self.tree_linker._tail_hashes(key, result, device_hit_len)
        ]
        state = _Prefetch(
            req=req,
            key=key,
            result=result,
            tail_hashes=tail_hashes,
            started=time.monotonic(),
            swa_evicted_before=req.kv.swa_evicted_seqlen if req.kv is not None else 0,
        )
        self.prefetches[req.rid] = state
        if len(tail_hashes) * self.cache.page_size < self.prefetch_threshold:
            state.phase = "ready"
            return result
        transfers = self._lookup_transfers(tail_hashes)
        if not transfers:
            state.phase = "ready"
            return result
        state.hold_node = result.last_device_node
        state.hold_params = self.cache.inc_lock_ref(state.hold_node).to_dec_params()
        try:
            state.future = self.cache_linker.submit_lookup(req.rid, transfers)
        except BaseException:
            self._release_hold(state)
            self.prefetches.pop(req.rid)
            raise
        return result

    def _lookup_transfers(self, keys: list[str]) -> list:
        transfers = []
        for component in self.cache._components_tuple:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOOKUP, None, keys
            )
            if transfer is None:
                return []
            transfers.append(transfer)
        return transfers

    def _reduce(self, values: list[int], op) -> list[int]:
        tensor = torch.tensor(values, dtype=torch.int, device="cpu")
        self.cache._all_reduce_attn_groups(tensor, op)
        return tensor.tolist()

    def poll(self) -> None:
        self._poll_offloads()
        self._log_metrics()
        # FLAT_MEMORY: Every rank visits the same request list, never worker order.
        states = list(self.prefetches.values())
        if not states:
            return
        query_ready = self._reduce(
            [
                int(s.phase == "query" and s.future is not None and s.future.done())
                for s in states
            ],
            torch.distributed.ReduceOp.MIN,
        )
        for state, ready in zip(states, query_ready):
            if ready:
                self._finish_query(state)
        occupied = sum(s.loaded_tokens for s in states if s.prepared is not None)
        for state in states:
            if state.phase == "allocate":
                self._prepare(state, occupied=occupied)
                if state.prepared is not None:
                    occupied += state.loaded_tokens
        self._poll_reads(states)
        for state in states:
            if state.cancelled and state.phase == "ready":
                self._forget(state)

    def _finish_query(self, state: _Prefetch) -> None:
        if state.cancelled:
            state.phase = "ready"
            return
        try:
            restorable = state.future.result()
        except Exception:
            self.io_errors += 1
            logger.exception("Flat storage lookup failed for request %s", state.req.rid)
            restorable = []
        pages = self.tree_linker._sync_restorable_prefix(
            restorable, num_pages=len(state.tail_hashes), device_hit_pages=0
        )
        if pages == 0:
            state.phase = "ready"
            self._release_hold(state)
            return
        state.loaded_tokens = pages * self.cache.page_size
        state.tail_hashes = state.tail_hashes[:pages]
        state.phase = "allocate"

    def _prepare(self, state: _Prefetch, *, occupied: int) -> None:
        if state.cancelled:
            state.phase = "ready"
            return
        # One large prefix is allowed; simultaneous prefixes share the fixed budget.
        if occupied and occupied + state.loaded_tokens > self.max_prefetch_tokens:
            return
        rid = state.req.rid
        self.tree_linker.hit_markers[rid] = ExternalCacheHitMarker(
            prefix_key=state.key[
                : len(state.result.device_indices) + state.loaded_tokens
            ],
            tail_hashes=state.tail_hashes,
            device_hit_len=len(state.result.device_indices),
        )
        error = None
        try:
            state.prepared = self.tree_linker.prepare_load(
                state.req,
                prefix_indices=state.result.device_indices,
                anchor=state.result.last_device_node,
            )
        except Exception as exc:
            error = exc
        allocated = self._reduce(
            [int(state.prepared is not None)], torch.distributed.ReduceOp.MIN
        )[0]
        if not allocated:
            if state.prepared is not None:
                self.tree_linker.abort_prepared_load(state.prepared)
                state.prepared = None
            if error is not None:
                logger.warning("Flat private allocation failed: %s", error)
            self._restore_request(state)
            self._release_hold(state)
            state.phase = "ready"
            return
        try:
            queued = self.cache_linker.load(
                rid, [transfer for _, transfer in state.prepared.component_transfers]
            )
        except Exception:
            logger.exception("Flat failed to queue private read for %s", rid)
            queued = False
        queued_everywhere = self._reduce([int(queued)], torch.distributed.ReduceOp.MIN)[
            0
        ]
        if not queued_everywhere:
            if queued:
                self.cache_linker.cancel_private_load(rid)
            self.tree_linker.abort_prepared_load(state.prepared)
            state.prepared = None
            self._restore_request(state)
            self._release_hold(state)
            state.phase = "ready"
            return
        self.cache_linker.start_preparing_loads()
        state.phase = "read"

    def _poll_reads(self, states: list[_Prefetch]) -> None:
        done, errors = [], []
        for state in states:
            ready, failed = False, False
            if state.phase == "read":
                try:
                    ready = self.cache_linker.load_ready(state.req.rid)
                except Exception:
                    ready, failed = True, True
            done.append(int(ready))
            errors.append(int(failed))
        done = self._reduce(done, torch.distributed.ReduceOp.MIN)
        errors = self._reduce(errors, torch.distributed.ReduceOp.MAX)
        for state, ready, failed in zip(states, done, errors):
            if not ready:
                continue
            if failed or state.cancelled:
                self.tree_linker.abort_prepared_load(state.prepared)
                state.prepared = None
                self._restore_request(state)
                self._release_hold(state)
                if failed:
                    self.io_errors += 1
                    logger.warning(
                        "Flat read failed; recomputing request %s", state.req.rid
                    )
            else:
                self._commit(state)
                self._record_read_metrics(state)
            self.cache_linker.forget_request(state.req.rid)
            state.phase = "ready"

    def _commit(self, state: _Prefetch) -> None:
        # FLAT_MEMORY: All required components and ranks completed before tree insertion.
        _, node = self.tree_linker.commit_prepared_load(state.prepared, queue_io=False)
        state.prepared = None
        hold = self.cache.inc_lock_ref(node).to_dec_params()
        self._release_hold(state)
        state.hold_node, state.hold_params = node, hold
        state.req.storage_hit_length = state.loaded_tokens
        state.req.storage_read_tokens = state.loaded_tokens
        state.req.storage_read_latency_ms = (time.monotonic() - state.started) * 1000

    def _record_read_metrics(self, state: _Prefetch) -> None:
        result = self.cache_linker.take_load_result(state.req.rid)
        counts = [result.dram_pages, result.ssd_pages, result.mixed_pages]
        minima = self._reduce(counts, torch.distributed.ReduceOp.MIN)
        maxima = self._reduce(counts, torch.distributed.ReduceOp.MAX)
        elapsed_us = self._reduce(
            [int(result.elapsed_seconds * 1_000_000)], torch.distributed.ReduceOp.MAX
        )[0]
        stats = state.req.flat_prefetch_stats
        for name, count in zip(("flat_dram", "flat_ssd", "flat_mixed"), minima):
            stats[name] = count * self.cache.page_size if minima == maxima else None
        # A mixed read belongs to SSD; it is not a third, additive cache tier.
        medium = "ssd" if maxima[1] else "dram"
        stats[f"flat_prefetch_{medium}_ms"] += elapsed_us / 1000.0
        stats[f"flat_prefetch_{medium}_ops"] += 1
        state.req.storage_read_latency_ms = elapsed_us / 1000.0
        collector = self.cache.storage_metrics_collector
        if collector is not None:
            collector.log_prefetched_tokens(state.loaded_tokens)
            collector.log_prefetch_latency_ms(elapsed_us / 1000.0)

    def _restore_request(self, state: _Prefetch) -> None:
        if state.req.kv is not None:
            state.req.kv.swa_evicted_seqlen = state.swa_evicted_before
        state.loaded_tokens = 0

    def is_ready(self, rid: str) -> bool:
        state = self.prefetches.get(rid)
        return state is None or state.phase == "ready"

    def release_ready_holds(self) -> None:
        # FLAT_MEMORY: Ready prefixes must remain evictable so a suffix can be admitted.
        # Admission rematches and takes its own request lock before allocating tokens.
        for state in self.prefetches.values():
            if state.phase == "ready":
                self._release_hold(state)

    def on_admitted(self, requests: list) -> None:
        for req in requests:
            state = self.prefetches.get(req.rid)
            if state is not None:
                if state.phase != "ready":
                    raise RuntimeError("Flat admitted an incomplete storage read")
                start = len(state.result.device_indices)
                reused = max(
                    0, min(len(req.prefix_indices) - start, state.loaded_tokens)
                )
                req.storage_hit_length = reused
                if reused != state.loaded_tokens:
                    for name in ("flat_dram", "flat_ssd", "flat_mixed"):
                        req.flat_prefetch_stats[name] = 0 if reused == 0 else None
                self._forget(state, keep_arrival=True)

    def release_request(self, rid: str) -> None:
        state = self.prefetches.get(rid)
        if state is None:
            self.arrivals.pop(rid, None)
            return
        state.cancelled = True
        if state.phase == "ready":
            self._forget(state)
        elif state.phase == "query":
            state.future.cancel()
        elif state.phase == "allocate":
            state.phase = "ready"

    def _release_hold(self, state: _Prefetch) -> None:
        if state.hold_params is not None:
            self.cache.dec_lock_ref(state.hold_node, state.hold_params)
            state.hold_node = state.hold_params = None

    def _forget(self, state: _Prefetch, *, keep_arrival: bool = False) -> None:
        self._release_hold(state)
        self.cache_linker.forget_request(state.req.rid)
        self.tree_linker.hit_markers.pop(state.req.rid, None)
        self.prefetches.pop(state.req.rid, None)
        if not keep_arrival:
            self.arrivals.pop(state.req.rid, None)

    def offload_nodes(self, node_ids) -> None:
        for node_id in node_ids:
            node = self.cache.resolve_node_handle(node_id)
            if node.external_cache_stored or node.write_through_pending_id is not None:
                continue
            self.tree_linker._offload_node(
                node_id, key_namespace=request_storage_namespace(node.key)
            )
            # FLAT_MEMORY: Pending deduplicates writes; it is not durable storage state.
            node.external_cache_stored = False

    def replace_pending_offload_node(self, ack_id, old_node_id, new_node_ids) -> None:
        self.tree_linker.replace_pending_offload_node(ack_id, old_node_id, new_node_ids)

    def _poll_offloads(self) -> None:
        count = self._reduce(
            [self.tree_linker.num_completed_offloads()], torch.distributed.ReduceOp.MIN
        )[0]
        if count:
            successes = self.tree_linker.take_completed_offloads(count)
            successes = self._reduce(
                [int(x) for x in successes], torch.distributed.ReduceOp.MIN
            )
            self.io_errors += sum(not success for success in successes)
            collector = self.cache.storage_metrics_collector
            if collector is not None:
                for success, pending in zip(successes, self.pending_offloads):
                    if success:
                        tokens = sum(
                            len(self.cache.resolve_node_handle(node_id).key)
                            for node_id in pending.publish_node_ids
                        )
                        collector.log_backuped_tokens(tokens)
                        collector.log_storage_write_tokens(tokens)
            self.tree_linker.commit_completed_offloads([bool(x) for x in successes])

    def _log_metrics(self) -> None:
        collector = self.cache.storage_metrics_collector
        now = time.monotonic()
        if collector is not None and now - self._last_metrics_log >= 1.0:
            collector.log_storage_metrics(self.cache_linker.get_stats())
            self._last_metrics_log = now

    def is_idle(self) -> bool:
        return not self.prefetches and not self.pending_offloads

    def drain(self, timeout: float | None = None) -> bool:
        deadline = time.monotonic() + (
            self.drain_timeout if timeout is None else timeout
        )
        while not self.is_idle():
            self.poll()
            expired = self._reduce(
                [int(time.monotonic() >= deadline)], torch.distributed.ReduceOp.MAX
            )[0]
            if expired:
                return False
            time.sleep(0.001)
        return bool(self.cache_linker.drain(max(0.0, deadline - time.monotonic())))

    def reset(self) -> None:
        for rid in list(self.prefetches):
            self.release_request(rid)
        if not self.drain():
            raise TimeoutError(
                "Flat cache reset timed out; device allocations remain owned"
            )
        self.tree_linker.reset()
        self.arrivals.clear()

    def close(self) -> None:
        self.reset()
        self.tree_linker.close()

    def start_layer_wise_loading(self) -> int:
        # Flat loads complete before admission, so no forward-stream layer wait is needed.
        return -1

    def snapshot(self) -> dict:
        result = self.cache_linker.get_flat_memory_stats()
        result.update(
            pending_backups=len(self.pending_offloads),
            pending_prefetches=len(self.prefetches),
            flat_io_errors=self.io_errors,
            application_host_staging_bytes=0,
            host_pool_bytes=0,
            gds_mode="compat",
        )
        return result

    def gather(self, value) -> list:
        if self.cache.tp_world_size == 1:
            return [value]
        values = [None] * self.cache.tp_world_size
        torch.distributed.all_gather_object(values, value, group=self.cache.tp_group)
        return values

    def can_clear(self, *, requests_idle: bool) -> bool:
        try:
            allowed = requests_idle and not self.manager.get_io_window()["active"]
        except Exception:
            allowed = False
        return bool(self._reduce([int(allowed)], torch.distributed.ReduceOp.MIN)[0])

    def validate_mutation(
        self, operation: str, *, flush_requested: bool, requests_idle: bool
    ) -> None:
        if operation == "memory":
            raise ValueError(
                "Detach Flat direct storage before releasing model or KV memory."
            )
        if not flush_requested:
            raise ValueError("Flat direct weight updates require flush_cache=True.")
        if not self.can_clear(requests_idle=requests_idle):
            raise ValueError(
                "Drain Flat requests and close the I/O window before updating weights."
            )
        if not self.drain():
            raise TimeoutError("Flat I/O did not drain before the weight update.")

    def flush_after_weight_update(self, *, flush_gpu) -> bool:
        if not flush_gpu():
            return False
        self.clear()
        return True

    def clear(self) -> None:
        self.reset()
        self.cache_linker.clear()


def get_flat_memory_cache(cache) -> FlatMemoryCache | None:
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    return cache.flat_memory if isinstance(cache, UnifiedRadixCache) else None
