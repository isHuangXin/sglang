# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned transactions for Flat Memory device-prefix reuse."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.storage.flat_memory.io_result import FlatUnsafeIOError
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
    original_device_tokens: int = 0
    query_start: int = 0
    queried_endpoint: int = 0
    query_generation: int = -1
    refresh_attempts: int = 0
    refresh_deadline: float = 0.0
    read_anchor_tokens: int = 0
    read_endpoint: int = 0
    lease_started: float = 0.0
    ready_started: float = 0.0
    full_reservation: int = 0
    swa_reservation: int = 0
    bypass: bool = False
    revoke_reason: str = ""
    admission_attempted: bool = False
    admitted: bool = False


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
        self.capacity_pressure_offloads = 0
        self.lease_fallbacks = 0
        self.restore_deferrals = 0
        self.ready_dropped_tokens = 0
        self.admission_dropped_tokens = 0
        self._lease_rid: str | None = None
        self._admission_candidate: str | None = None
        self._admission_states: list[_Prefetch] = []
        self._last_metrics_log = 0.0
        threshold_override = envs.SGLANG_PREFETCH_THRESHOLD.get()
        self.prefetch_threshold = int(
            config.get("prefetch_threshold", 256)
            if threshold_override is None
            else threshold_override
        )
        # FLAT_MEMORY: The legacy budget permits one oversized prefix; leases serialize it.
        self.max_prefetch_tokens = int(config.get("prefetch_max_tokens", 65536))
        self.drain_timeout = float(config.get("drain_timeout", 30.0))
        self.lease_timeout = float(
            config.get("restore_lease_timeout", self.drain_timeout)
        )
        if self.prefetch_threshold < 0 or self.max_prefetch_tokens < cache.page_size:
            raise ValueError("Invalid Flat prefetch threshold or token budget")
        if not math.isfinite(self.drain_timeout) or self.drain_timeout <= 0:
            raise ValueError("Flat drain_timeout must be finite and positive")
        if not math.isfinite(self.lease_timeout) or self.lease_timeout <= 0:
            raise ValueError("Flat restore_lease_timeout must be finite and positive")

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
        # FLAT_MEMORY: A ready lease gets the first real admission attempt.
        ordered = (
            sorted(requests, key=lambda req: self.arrivals[req.rid])
            if self.ready_fcfs
            else requests
        )
        lease = self.prefetches.get(self._lease_rid)
        if lease is not None and lease.phase == "ready":
            return [req for req in ordered if req.rid == self._lease_rid] + [
                req for req in ordered if req.rid != self._lease_rid
            ]
        return ordered

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
            original_device_tokens=device_hit_len,
            query_start=device_hit_len,
            queried_endpoint=device_hit_len,
        )
        self.prefetches[req.rid] = state
        self._diagnose(state, original_device_tokens=device_hit_len)
        if (
            not tail_hashes
            or len(tail_hashes) * self.cache.page_size < self.prefetch_threshold
        ):
            state.phase = "ready"
            return result
        transfers = self._lookup_transfers(tail_hashes)
        if not transfers:
            state.phase, state.bypass = "ready", True
            return result
        # FLAT_MEMORY: Metadata probes never retain an evictable device anchor.
        try:
            state.future = self.cache_linker.submit_lookup(req.rid, transfers)
        except BaseException:
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
        self.cache_linker.poll_queries()
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
        self._expire_lease()
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
        state.query_generation = self.cache_linker.get_lookup_generation(state.req.rid)
        state.queried_endpoint = state.query_start + pages * self.cache.page_size
        state.phase = "allocate" if pages else "ready"
        self._diagnose(
            state,
            queried_endpoint=state.queried_endpoint,
            query_ms=(time.monotonic() - state.started) * 1000,
        )

    def carry_reservation(self, adder) -> None:
        lease = self.prefetches.get(self._lease_rid)
        active = lease is not None and not lease.admitted
        adder.set_flat_restore_reservation(
            full_tokens=lease.full_reservation if active else 0,
            swa_tokens=lease.swa_reservation if active else 0,
        )

    def prepare_admission(
        self, requests: list, adder, *, has_chunked_req: bool
    ) -> list:
        # FLAT_MEMORY: This stage is shared by all ranks, before the local add loop.
        requests = self.order_ready_requests(requests)
        self._agree_requests(requests)
        self._admission_states = [
            self.prefetches[req.rid] for req in requests if req.rid in self.prefetches
        ]
        self._admission_candidate = None
        for state in self._admission_states:
            state.admission_attempted = False
        self._refresh_stale_queries(self._admission_states)
        lease = self.prefetches.get(self._lease_rid)
        if lease is not None:
            if lease.phase == "ready":
                self._revalidate_lease(lease, adder)
        elif not has_chunked_req:
            candidate = next(
                (s for s in self._admission_states if s.phase == "allocate"), None
            )
            if candidate is not None:
                self._admission_candidate = candidate.req.rid
                self._select_restore(candidate, adder)
        self.carry_reservation(adder)
        return self.order_ready_requests(requests)

    def _agree_requests(self, requests: list) -> None:
        identity = json.dumps([req.rid for req in requests], separators=(",", ":"))
        digest = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
        signature = [len(requests), digest & 0x7FFFFFFF]
        minima = self._reduce(signature, torch.distributed.ReduceOp.MIN)
        maxima = self._reduce(signature, torch.distributed.ReduceOp.MAX)
        if minima != maxima:
            raise RuntimeError("Flat restore admission queue differs across TP ranks")

    def _refresh_stale_queries(self, states: list[_Prefetch]) -> None:
        if not states:
            return
        generation = self.cache_linker.storage_generation
        stale = self._reduce(
            [
                int(
                    s.phase in ("ready", "allocate")
                    and not s.bypass
                    and not s.lease_started
                    and s.future is not None
                    and s.refresh_attempts < 2
                    and s.queried_endpoint
                    < len(s.key) // self.cache.page_size * self.cache.page_size
                    and s.query_generation < generation
                )
                for s in states
            ],
            torch.distributed.ReduceOp.MAX,
        )
        for state, refresh in zip(states, stale):
            check_gap = (
                state.phase == "ready"
                and not state.bypass
                and not state.lease_started
                and state.query_start > 0
            )
            if not refresh and not check_gap:
                continue
            result = self._rematch(state)
            gap = len(result.device_indices) < state.query_start
            if not refresh and not gap:
                continue
            tail_tokens = (
                (len(state.key) - len(result.device_indices))
                // self.cache.page_size
                * self.cache.page_size
            )
            if state.future is None and tail_tokens < self.prefetch_threshold:
                continue
            if self._can_refresh(state):
                self._refresh_query(state, result)
            elif gap:
                self._fallback(state, "anchor_gap_refresh_exhausted")

    def _can_refresh(self, state: _Prefetch) -> bool:
        now = time.monotonic()
        if not state.refresh_deadline:
            state.refresh_deadline = now + self.lease_timeout
        exhausted = self._reduce(
            [int(state.refresh_attempts >= 2 or now >= state.refresh_deadline)],
            torch.distributed.ReduceOp.MAX,
        )[0]
        return not exhausted

    def _rematch(self, state: _Prefetch) -> MatchResult:
        # FLAT_MEMORY: A metadata-only query's original node/indices may be evicted.
        result = self.cache.match_prefix(MatchPrefixParams(key=state.key))
        size = [len(result.device_indices)]
        minima = self._reduce(size, torch.distributed.ReduceOp.MIN)
        maxima = self._reduce(size, torch.distributed.ReduceOp.MAX)
        if minima != maxima:
            raise RuntimeError("Flat restore device anchor differs across TP ranks")
        return result

    def _refresh_query(self, state: _Prefetch, result: MatchResult) -> None:
        state.refresh_attempts += 1
        state.query_start = len(result.device_indices)
        namespace = request_storage_namespace(state.key)
        state.tail_hashes = [
            namespace + value
            for value in self.tree_linker._tail_hashes(
                state.key, result, state.query_start
            )
        ]
        state.queried_endpoint = state.query_start
        if not state.tail_hashes:
            state.phase = "ready"
            self.cache_linker.release_lookup(state.req.rid)
            state.future = None
            return
        transfers = self._lookup_transfers(state.tail_hashes)
        if not transfers:
            state.phase, state.bypass = "ready", True
            self.cache_linker.release_lookup(state.req.rid)
            return
        state.future = self.cache_linker.refresh_lookup(state.req.rid, transfers)
        state.phase = "query"
        self._diagnose(state, refresh_attempts=state.refresh_attempts)

    def _select_restore(self, state: _Prefetch, adder) -> None:
        result = self._rematch(state)
        device_tokens = len(result.device_indices)
        if device_tokens >= state.queried_endpoint:
            self._fallback(state, "device_covers_restore", count=False)
            return
        if device_tokens < state.query_start:
            if self._can_refresh(state):
                self._refresh_query(state, result)
            else:
                self._fallback(state, "anchor_gap_refresh_exhausted")
            return
        page = self.cache.page_size
        keys = state.tail_hashes[
            (device_tokens - state.query_start)
            // page : (state.queried_endpoint - state.query_start)
            // page
        ]
        footprint = self.tree_linker.plan_load_tokens(keys)
        supported = self._reduce(
            [int(footprint is not None)], torch.distributed.ReduceOp.MIN
        )[0]
        if not supported:
            self._fallback(state, "unsupported_restore_footprint")
            return
        state.result = result
        state.hold_node = result.last_device_node
        state.hold_params = self.cache.inc_lock_ref(state.hold_node).to_dec_params()
        plan = adder.plan_flat_restore(
            state.req,
            restored_prefix_len=state.queried_endpoint,
            full_tokens=footprint.get(PoolName.KV, 0),
            swa_tokens=footprint.get(PoolName.SWA, 0),
        )
        can_restore, never_fits = self._sync_plan(plan)
        if not can_restore:
            self._release_hold(state)
            self.restore_deferrals += 1
            self._diagnose(state, reason=plan.reason or "restore_headroom")
            if never_fits:
                self._fallback(state, "restore_never_fits")
            return
        state.full_reservation, state.swa_reservation = self._sync_reservation(plan)
        state.read_anchor_tokens = device_tokens
        state.read_endpoint = state.queried_endpoint
        state.loaded_tokens = state.read_endpoint - device_tokens
        state.lease_started = time.monotonic()
        self._lease_rid = state.req.rid
        self._diagnose(
            state,
            read_anchor_tokens=device_tokens,
            read_endpoint=state.read_endpoint,
            prepared_tokens=state.loaded_tokens,
            prepare_ms=(state.lease_started - state.started) * 1000,
        )
        self._prepare(state, keys=keys)

    def _sync_plan(self, plan) -> tuple[bool, bool]:
        can_restore = self._reduce(
            [int(plan.can_restore)], torch.distributed.ReduceOp.MIN
        )[0]
        never_fits = self._reduce(
            [int(plan.never_fits)], torch.distributed.ReduceOp.MAX
        )[0]
        return bool(can_restore), bool(never_fits)

    def _sync_reservation(self, plan) -> tuple[int, int]:
        full, swa = self._reduce(
            [plan.full_reservation, plan.swa_reservation],
            torch.distributed.ReduceOp.MAX,
        )
        return full, swa

    def _revalidate_lease(self, state: _Prefetch, adder) -> None:
        adder.set_flat_restore_reservation(full_tokens=0, swa_tokens=0)
        plan = adder.plan_flat_restore(
            state.req,
            restored_prefix_len=state.read_endpoint,
            full_tokens=0,
            swa_tokens=0,
        )
        can_restore, never_fits = self._sync_plan(plan)
        if never_fits:
            self._fallback(state, "ready_never_fits")
        elif can_restore:
            state.full_reservation, state.swa_reservation = self._sync_reservation(plan)
        else:
            self._diagnose(state, reason=plan.reason or "ready_headroom")

    def _prepare(self, state: _Prefetch, *, keys: list[str]) -> None:
        rid = state.req.rid
        self.tree_linker.hit_markers[rid] = ExternalCacheHitMarker(
            prefix_key=state.key[: state.read_endpoint],
            tail_hashes=keys,
            device_hit_len=state.read_anchor_tokens,
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
            self._fallback(state, "private_allocation_failed")
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
            self._fallback(state, "private_read_queue_failed")
            return
        # FLAT_MEMORY: A dispatch failure must not make possibly active DMA evictable.
        state.phase = "read"
        self.cache_linker.start_preparing_loads()

    def _poll_reads(self, states: list[_Prefetch]) -> None:
        done, errors = [], []
        for state in states:
            ready, failed = False, False
            if state.phase == "read":
                try:
                    ready = self.cache_linker.load_ready(state.req.rid)
                except FlatUnsafeIOError:
                    ready, failed = True, 2
                except Exception:
                    ready, failed = True, 1
            done.append(int(ready))
            errors.append(int(failed))
        done = self._reduce(done, torch.distributed.ReduceOp.MIN)
        errors = self._reduce(errors, torch.distributed.ReduceOp.MAX)
        for state, ready, failed in zip(states, done, errors):
            if not ready:
                continue
            if failed == 2:
                raise FlatUnsafeIOError(
                    "Flat restore did not quiesce on every TP rank; "
                    "private GPU pages remain owned"
                )
            if failed or state.cancelled or state.revoke_reason:
                self.tree_linker.abort_prepared_load(state.prepared)
                state.prepared = None
                self._restore_request(state)
                state.phase = "ready"
                self._fallback(
                    state,
                    "read_failed" if failed else state.revoke_reason or "cancelled",
                    count=not failed and not state.cancelled,
                )
                if failed:
                    self.io_errors += 1
                    logger.warning(
                        "Flat read failed; recomputing request %s", state.req.rid
                    )
            else:
                self._commit(state)
                self._record_read_metrics(state)
                state.phase = "ready"
            self.cache_linker.forget_request(state.req.rid)

    def _commit(self, state: _Prefetch) -> None:
        # FLAT_MEMORY: All required components and ranks completed before tree insertion.
        _, node = self.tree_linker.commit_prepared_load(state.prepared, queue_io=False)
        state.prepared = None
        # FLAT_MEMORY: Published SWA belongs to the tree; admission may rematch less.
        if state.req.kv is not None:
            state.req.kv.swa_evicted_seqlen = state.swa_evicted_before
        hold = self.cache.inc_lock_ref(node).to_dec_params()
        self._release_hold(state)
        state.hold_node, state.hold_params = node, hold
        state.ready_started = time.monotonic()
        state.req.storage_read_tokens = state.loaded_tokens
        state.req.storage_read_latency_ms = (state.ready_started - state.started) * 1000
        self._diagnose(
            state,
            committed_tokens=state.loaded_tokens,
            ready_endpoint=state.read_endpoint,
            ready_ms=(state.ready_started - state.started) * 1000,
        )

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

    def before_admission(self, req: Req, adder) -> None:
        state = self.prefetches.get(req.rid)
        if state is None:
            return
        if state.phase != "ready":
            raise RuntimeError("Flat attempted admission before storage readiness")
        if req.rid == self._lease_rid:
            # FLAT_MEMORY: Ordinary admission now charges the owner's suffix/decode.
            adder.set_flat_restore_reservation(full_tokens=0, swa_tokens=0)
        req.storage_hit_length = self._reused_tokens(state)
        self._diagnose(
            state,
            admission_endpoint=len(req.prefix_indices),
            admission_storage_tokens=req.storage_hit_length,
        )

    def after_admission(self, req: Req, adder) -> None:
        state = self.prefetches.get(req.rid)
        if state is not None:
            state.admission_attempted = True
            # FLAT_MEMORY: CONTINUE is not acceptance; NO_TOKEN may follow an append.
            if any(candidate is req for candidate in adder.can_run_list):
                self._handoff(state)
        self.carry_reservation(adder)

    def finish_admission(self, adder, *, can_progress: bool) -> None:
        states = self._admission_states
        if states:
            admitted = [int(state.admitted) for state in states]
            minima = self._reduce(admitted, torch.distributed.ReduceOp.MIN)
            maxima = self._reduce(admitted, torch.distributed.ReduceOp.MAX)
            if minima != maxima:
                raise RuntimeError("Flat request admission differs across TP ranks")
            for state, accepted in zip(states, minima):
                if accepted:
                    self._forget(state, keep_arrival=True)
        state = self.prefetches.get(self._lease_rid or self._admission_candidate)
        # FLAT_MEMORY: A rank-local gate may skip admission without rejecting the KV.
        can_progress, admission_attempted = self._reduce(
            [int(can_progress), int(state is not None and state.admission_attempted)],
            torch.distributed.ReduceOp.MIN,
        )
        if (
            state is not None
            and not state.bypass
            and not can_progress
            and (
                state.phase == "allocate"
                or (state.phase == "ready" and admission_attempted)
            )
        ):
            self._fallback(state, "no_admission_progress")
        self._admission_states = []
        self._admission_candidate = None
        self.carry_reservation(adder)

    def _reused_tokens(self, state: _Prefetch) -> int:
        return max(
            0,
            min(
                len(state.req.prefix_indices) - state.read_anchor_tokens,
                state.loaded_tokens,
            ),
        )

    def _handoff(self, state: _Prefetch) -> None:
        if state.admitted:
            return
        if state.phase != "ready":
            raise RuntimeError("Flat admitted an incomplete storage read")
        req = state.req
        reused = self._reused_tokens(state)
        req.storage_hit_length = reused
        dropped = state.loaded_tokens - reused
        self.admission_dropped_tokens += dropped
        if dropped:
            for name in ("flat_dram", "flat_ssd", "flat_mixed"):
                req.flat_prefetch_stats[name] = 0 if reused == 0 else None
        now = time.monotonic()
        self._diagnose(
            state,
            admission_endpoint=len(req.prefix_indices),
            admission_storage_tokens=reused,
            admission_dropped_tokens=dropped,
            admitted_ms=(now - state.started) * 1000,
            ready_wait_ms=(
                (now - state.ready_started) * 1000 if state.ready_started else 0
            ),
        )
        # FLAT_MEMORY: Release the saved endpoint/params, not req.last_node after rematch.
        self._release_hold(state)
        state.admitted = True

    def on_admitted(self, requests: list) -> None:
        for req in requests:
            state = self.prefetches.get(req.rid)
            if state is not None:
                self._handoff(state)
                self._forget(state, keep_arrival=True)

    def _expire_lease(self) -> None:
        state = self.prefetches.get(self._lease_rid)
        if state is None:
            return
        expired = self._reduce(
            [int(time.monotonic() - state.lease_started >= self.lease_timeout)],
            torch.distributed.ReduceOp.MAX,
        )[0]
        if expired and not state.revoke_reason:
            self._fallback(state, "lease_expired")

    def _fallback(self, state: _Prefetch, reason: str, *, count: bool = True) -> None:
        if state.phase == "read":
            # FLAT_MEMORY: Revocation is logical until all-rank I/O completion.
            state.revoke_reason = reason
            return
        if state.prepared is not None:
            raise RuntimeError("Flat cannot revoke a private allocation before drain")
        if count:
            self.lease_fallbacks += 1
            # Tokens abandoned at ready are separate from actual admission loss.
            if state.hold_params is not None and state.ready_started:
                self.ready_dropped_tokens += state.loaded_tokens
        self._release_hold(state)
        if self._lease_rid == state.req.rid:
            self._lease_rid = None
        state.full_reservation = state.swa_reservation = 0
        state.phase, state.bypass = "ready", True
        state.revoke_reason = reason
        self.tree_linker.hit_markers.pop(state.req.rid, None)
        self.cache_linker.release_lookup(state.req.rid)
        self._diagnose(
            state,
            reason=reason,
            fallback_ms=(time.monotonic() - state.started) * 1000,
        )

    def _diagnose(self, state: _Prefetch, **values) -> None:
        state.req.flat_prefetch_stats.update(
            (f"flat_restore_{name}", value) for name, value in values.items()
        )

    def release_request(self, rid: str) -> None:
        state = self.prefetches.get(rid)
        if state is None:
            self.arrivals.pop(rid, None)
            return
        state.cancelled = True
        if state.phase != "read":
            self._forget(state)

    def _release_hold(self, state: _Prefetch) -> None:
        if state.hold_params is not None:
            self.cache.dec_lock_ref(state.hold_node, state.hold_params)
            state.hold_node = state.hold_params = None

    def _forget(self, state: _Prefetch, *, keep_arrival: bool = False) -> None:
        if state.prepared is not None or state.phase == "read":
            raise RuntimeError("Flat cannot forget a restore before I/O drain")
        self._release_hold(state)
        self.cache_linker.forget_request(state.req.rid)
        self.tree_linker.hit_markers.pop(state.req.rid, None)
        self.prefetches.pop(state.req.rid, None)
        if self._lease_rid == state.req.rid:
            self._lease_rid = None
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
            results = [
                self.cache_linker.pop_completed_offload_result() for _ in range(count)
            ]
            # FLAT_MEMORY: Pressure on one rank must not mask another rank's I/O error.
            flags = self._reduce(
                [
                    flag
                    for result in results
                    for flag in (
                        int(not result.success),
                        int(result.capacity_rejected),
                        int(not result.success and not result.capacity_rejected),
                        int(result.unsafe),
                    )
                ],
                torch.distributed.ReduceOp.MAX,
            )
            if any(flags[3::4]):
                raise FlatUnsafeIOError(
                    "Flat offload did not quiesce on every TP rank; "
                    "source GPU pages remain owned"
                )
            successes = [not failed for failed in flags[0::4]]
            self.capacity_pressure_offloads += sum(flags[1::4])
            self.io_errors += sum(flags[2::4])
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

    def has_unfinished_io(self) -> bool:
        return self.cache_linker.has_unfinished_io()

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
        self._admission_states = []
        self._admission_candidate = self._lease_rid = None

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
            flat_capacity_pressure_offloads=self.capacity_pressure_offloads,
            flat_restore_lease_active=int(self._lease_rid is not None),
            flat_restore_lease_fallbacks=self.lease_fallbacks,
            flat_restore_deferrals=self.restore_deferrals,
            flat_restore_ready_dropped_tokens=self.ready_dropped_tokens,
            flat_restore_admission_dropped_tokens=self.admission_dropped_tokens,
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
