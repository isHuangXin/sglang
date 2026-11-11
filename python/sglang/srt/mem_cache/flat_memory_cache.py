# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned transactions for Flat Memory device-prefix reuse."""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import math
import time
from concurrent.futures import Future
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.flat_memory_control_reduce import (
    ShmControlReducer,
    control_reduce_group,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.storage.flat_memory.io_result import FlatUnsafeIOError
from sglang.srt.mem_cache.storage.flat_memory.payload import (
    MEDIA_HINT_ANY,
    MEDIA_HINT_DRAM,
    MEDIA_HINT_SSD,
)
from sglang.srt.mem_cache.unified_cache.components import LinkerTransferPhase
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    PreparedLinkerLoad,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.observability.flat_memory_metrics import (
    finalize_flat_cache_accounting,
    initialize_flat_request,
)
from sglang.srt.observability.storage_restore import RestoreEpisode, RestoreSource

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixKey

logger = logging.getLogger(__name__)

# FLAT_MEMORY: These are observed residence categories, not exclusive physical causes.
_SCHEDULER_GATES = (
    "engine_paused",
    "prefill_cadence",
    "min_free_slots",
    "request_slots",
    "lora",
    "admission_token_budget",
    "other_admission",
    "admission_tail",
    "test_gate",
)
_RESTORE_GATES = (
    "restore_prefill_pass_budget",
    "restore_prefill_request_limit",
    "restore_full_pool_capacity",
    "restore_full_decode_headroom",
    "restore_swa_restore_and_chunk_capacity",
    "restore_swa_decode_headroom",
    "restore_peer_veto",
    "restore_other",
)
_PREPARE_GATES = (
    "existing_lease",
    "chunked_prefill",
    "candidate_queue",
) + _RESTORE_GATES
_WAIT_BUCKETS = (
    (
        "unknown",
        "query_pending",
        "read_pending",
        "ready_admission",
    )
    + _SCHEDULER_GATES
    + _PREPARE_GATES
)
# FLAT_MEMORY: Reclaim reports are reconciled every this many polls, since each
# check costs a TP collective. Arbitrary.
_RECLAIM_SYNC_POLLS = 64
# Arbitrary; caps durable offloads retired per poll, extra ones wait a poll.
_OFFLOAD_FLAG_SLOTS = 64
# Arbitrary; bounds the per-pass scan for prefetch candidates in a long queue.
_PREFETCH_SCAN_REQUESTS = 512
# Bound for the page -> node map; arbitrary, rebuilt from live nodes past it.
_MAX_PAGE_NODES = 1 << 22

_CONTROL_NAMES = (
    "scheduler",
    "poll",
    "prepare_admission",
    "finish_admission",
    "refresh_stale_queries",
    "rematch",
    "reduce",
    "reduce_collective",
    "gather",
)


def _timed_control(method):
    name = method.__name__.lstrip("_")

    @wraps(method)
    def timed(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return method(self, *args, **kwargs)
        finally:
            self.record_control_time(name, time.monotonic() - started)

    return timed


def request_storage_namespace(key: RadixKey) -> str:
    # FLAT_MEMORY: Token hash chains alone do not isolate cache salts or extra keys.
    identity = json.dumps([key.extra_key, key.cache_salt], separators=(",", ":"))
    return "request-v1-" + hashlib.sha256(identity.encode()).hexdigest() + ":"


class _Prefetch(msgspec.Struct, kw_only=True):
    req: Any
    key: Any
    result: Any
    tail_hashes: list[str]
    restore_episode: RestoreEpisode | None = None
    restore_source: RestoreSource = RestoreSource()
    future: Future | None = None
    prepared: PreparedLinkerLoad | None = None
    phase: str = "query"
    # Rank-agreed eviction epoch `result` was last matched in; -1 = never.
    validated_epoch: int = -1
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
    storage_probe_attempts: int = 0
    anchor_refresh_attempts: int = 0
    refresh_deadline: float = 0.0
    device_cover_reactivations: int = 0
    read_anchor_tokens: int = 0
    read_endpoint: int = 0
    lease_started: float = 0.0
    ready_started: float = 0.0
    full_reservation: int = 0
    swa_reservation: int = 0
    ssd_pages: int = 0
    bypass: bool = False
    revoke_reason: str = ""
    # FLAT_MEMORY: FC-CIO layer-wise restore: committed before the read ends;
    # stream_hold keeps the published pages from eviction until it does.
    streaming: bool = False
    stream_hold: Any = None
    # Leaf of the early-published restore, dropped if the stream fails.
    stream_leaf_id: Any = None
    # A forward pass of the owner waited on this stream's layers.
    stream_consumed: bool = False
    admission_attempted: bool = False
    admitted: bool = False
    allocation_wait_started: float = 0.0
    allocation_wait_deadline: float = 0.0
    allocation_wait_eligible: bool = False
    allocation_plan_attempts: int = 0
    plan_rejection_recorded: bool = False
    wait_bucket: str = "unknown"
    wait_since: float = 0.0
    wait_seconds: dict[str, float] = msgspec.field(default_factory=dict)
    pre_prepare_seconds: dict[str, float] | None = None
    wait_finalized: bool = False


class FlatMemoryCache:
    """Compose the upstream tree linker with asynchronous, unpublished loads.

    All allocation, tree mutation and collectives run on the scheduler thread.
    Workers may only query storage or copy into private device allocations.
    """

    def __init__(self, *, cache, cache_linker, config: dict, ready_fcfs: bool = True):
        self.cache = cache
        self.ready_fcfs = ready_fcfs
        self.cache_linker = cache_linker
        self.tree_linker = UnifiedCacheLinkerWrapper(
            cache,
            cache_linker,
            resolve_pending_node=cache.tree_core.try_node_by_id,
        )
        self.prefetches: dict[str, _Prefetch] = {}
        self.arrivals: dict[str, int] = {}
        self._next_arrival = 0
        self.io_errors = 0
        self.capacity_pressure_offloads = 0
        self.lease_fallbacks = 0
        self.restore_deferrals = 0
        self.ready_dropped_tokens = 0
        self.admission_dropped_tokens = 0
        # FLAT_MEMORY: Rank-agreed lease order; collectives iterate it identically.
        self._leases: list[str] = []
        self._prefill_busy = False
        self._admission_candidate: str | None = None
        self._admission_states: list[_Prefetch] = []
        # None outside a pass; inside one, only these requests may be admitted.
        self._admission_rids: set[str] | None = None
        self._last_metrics_log = 0.0
        self._control_seconds = dict.fromkeys(_CONTROL_NAMES, 0.0)
        self._control_calls = dict.fromkeys(_CONTROL_NAMES, 0)
        self._revalidated_requests = 0
        threshold_override = envs.SGLANG_PREFETCH_THRESHOLD.get()
        self.prefetch_threshold = int(
            config.get("prefetch_threshold", 256)
            if threshold_override is None
            else threshold_override
        )
        self.max_prefetch_tokens = int(config.get("prefetch_max_tokens", 65536))
        # FLAT_MEMORY: FC-MAS queues; GPU pages are still gated by plan_flat_restore.
        # In-flight restores, not I/O threads (restore_ssd/dram_io_workers): with
        # 4 leases at 4 SSD readers the drive idled while 100+ restores waited.
        # 16 is arbitrary; each lease pins one request's restored pages.
        self.max_ssd_leases = int(config.get("restore_max_ssd_leases", 16))
        self.max_dram_leases = int(config.get("restore_max_dram_leases", 16))
        if self.max_ssd_leases < 1 or self.max_dram_leases < 1:
            raise ValueError("Flat restore lease limits must be positive")
        # FLAT_MEMORY: Unleased requests revalidated and admitted per pass, so a
        # long waiting queue costs O(window) per pass. 0 disables the bound.
        # One pass admits at most chunked_prefill_size / page_size requests (32
        # at 8192 / 256); each windowed request costs
        # ~0.1 ms of radix rematch per pass, so 128 spent ~12 ms on spares.
        self.admission_window = int(config.get("restore_admission_window", 32))
        if self.admission_window < 0:
            raise ValueError("Flat restore admission window must be non-negative")
        # FLAT_MEMORY: Queued requests past the window also start reads, into free
        # GPU pages only; this share of the full pool caps their unadmitted pages.
        # 0 disables it. 0.1 is arbitrary; the pool ran ~30% used under load.
        self.prefetch_pool_fraction = float(
            config.get("restore_prefetch_pool_fraction", 0.1)
        )
        if not 0.0 <= self.prefetch_pool_fraction <= 1.0:
            raise ValueError("Flat restore_prefetch_pool_fraction must be in [0, 1]")
        # FLAT_MEMORY: FC-GVA Table 2 write-time scope classification.
        self.scope_media_hints = bool(config.get("scope_media_hints", True))
        # Paper tau has no published value. 8192 is about 2.5x the ~3K mean per-turn
        # growth estimated from the CC replay statistics; not tuned on a real run.
        self.scope_short_turn_tokens = int(config.get("scope_short_turn_tokens", 8192))
        if self.scope_short_turn_tokens < 0:
            raise ValueError("Flat scope_short_turn_tokens must be nonnegative")
        # FLAT_MEMORY: Write-through stores a prefix before anyone shares it, so
        # refcount > 1 rarely fires; a positive value sends every prompt's first
        # tokens (the system prompt region) to DRAM instead. Off: replays that
        # isolate sessions would only waste DRAM on unshared copies.
        self.scope_shared_head_tokens = int(config.get("scope_shared_head_tokens", 0))
        if self.scope_shared_head_tokens < 0:
            raise ValueError("Flat scope_shared_head_tokens must be nonnegative")
        # FLAT_MEMORY: Table 2 L_agent. Positive makes a long context outrank the
        # short-turn rule, so an agentic session's history stays on SSD; 0 keeps
        # short-turn first. The paper gives no value.
        self.scope_agent_context_tokens = int(
            config.get("scope_agent_context_tokens", 0)
        )
        if self.scope_agent_context_tokens < 0:
            raise ValueError("Flat scope_agent_context_tokens must be nonnegative")
        self._write_scope_hint = MEDIA_HINT_ANY
        self.scope_hint_tokens = {"dram": 0, "ssd": 0, "any": 0}
        # FLAT_MEMORY: FC-CIO layer-pipelined restore. Early-published pages are
        # private to their owner until the read ends; a failed read retracts the
        # owner and recomputes it. The lead bounds a forward stall; it is arbitrary.
        self.layer_wise_restore = bool(config.get("layer_wise_restore", False))
        self.layer_wise_admit_lead = (
            float(config.get("layer_wise_admit_lead_ms", 200.0)) / 1000
        )
        if self.layer_wise_admit_lead < 0:
            raise ValueError("Flat layer_wise_admit_lead_ms must be nonnegative")
        # Streams only a restore whose remaining input is one prefill chunk, so the
        # owner's whole forward ends before its result is processed; must not
        # exceed --chunked-prefill-size.
        self.layer_wise_max_extend_tokens = int(
            config.get("layer_wise_max_extend_tokens", 8192)
        )
        # Prefill rate that sizes an owner's compute window; 0 uses the lead only.
        # ~35K tok/s was measured HBM-only on V4-Flash (README_hbm.md); 30K is slack.
        self.layer_wise_prefill_tokens_per_s = float(
            config.get("layer_wise_prefill_tokens_per_s", 30000.0)
        )
        if self.layer_wise_prefill_tokens_per_s < 0:
            raise ValueError("Flat layer_wise_prefill_tokens_per_s must be nonnegative")
        self._streaming: dict[str, _Prefetch] = {}
        self.layer_wise_commits = 0
        self.layer_wise_fallbacks = 0
        # Admitted owners of failed streams, retracted by the scheduler.
        self._failed_streams: dict[str, _Prefetch] = {}
        # Requests whose stream failed recompute without storage once.
        self._no_restore_rids: set[str] = set()
        # FLAT_MEMORY: page key -> (node id, page hash) of the node that backed it up.
        self._page_nodes: dict[str, tuple[int, str]] = {}
        self._polls = 0
        self.reclaim_invalidated_nodes = 0
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
        self._control_reducer = self._create_control_reducer(config)
        # FLAT_MEMORY: only an eviction can drop a matched anchor, so a state
        # matched in the current epoch needs no rematch.
        self._evict_epoch = 0
        self._evictions_seen = cache.evicted_tokens_total
        # FLAT_MEMORY: periodic rank-0 summary of the control counters above, so a
        # serving run shows where scheduler time goes. 0 disables it.
        self.control_log_seconds = float(config.get("control_log_seconds", 30.0))
        if self.control_log_seconds < 0:
            raise ValueError("Flat control_log_seconds must be nonnegative")
        self._logs_control = self.control_log_seconds > 0 and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        self._control_log_at = time.monotonic()
        self._control_logged = self._control_counters()

    def _create_control_reducer(self, config: dict) -> ShmControlReducer | None:
        # FLAT_MEMORY: Collective at construction, which every TP rank reaches in
        # the same order; any rank that cannot share memory keeps all on gloo.
        transport = config.get("tp_control_transport", "shm")
        if transport not in ("shm", "gloo"):
            raise ValueError(f"Flat tp_control_transport must be shm or gloo: {transport}")
        group = control_reduce_group(self.cache) if transport == "shm" else None
        if group is None:
            return None
        # Arbitrary; a rank this late has hung, and gloo would hang too.
        timeout = float(config.get("tp_control_timeout", 1800.0))
        reducer = ShmControlReducer.create(group=group, timeout=timeout)
        logger.info(
            "Flat TP control reduce transport: %s", "shm" if reducer else "gloo"
        )
        return reducer

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

    def begin_scheduler_pass(self) -> None:
        # FLAT_MEMORY: Close prior observations before making new scheduling decisions.
        now = time.monotonic()
        if self._logs_control and now - self._control_log_at >= self.control_log_seconds:
            self._log_control(now)
        for state in self.prefetches.values():
            if state.admitted or state.cancelled or state.wait_finalized:
                continue
            bucket = (
                f"{state.phase}_pending"
                if state.phase in ("query", "read")
                else "unknown"
            )
            # Same bucket: the open interval keeps accruing, nothing to close.
            if state.wait_bucket != bucket:
                self._set_wait(state, bucket, now=now)

    def observe_scheduler_gate(self, reason: str, requests=None) -> None:
        # FLAT_MEMORY: Query/read ownership outranks gates; admission_tail must not
        # erase a specific prepare gate. Other observed gates replace one another.
        if reason not in _SCHEDULER_GATES:
            reason = "other_admission"
        states = (
            self.prefetches.values()
            if requests is None
            else (self.prefetches.get(req.rid) for req in requests)
        )
        now = time.monotonic()
        for state in states:
            if (
                state is None
                or state.wait_bucket == reason
                or state.admitted
                or state.cancelled
                or state.wait_finalized
                or state.phase in ("query", "read")
                or (reason == "admission_tail" and state.wait_bucket in _PREPARE_GATES)
            ):
                continue
            self._set_wait(state, reason, now=now)

    def _control_counters(self) -> tuple[dict, dict, int]:
        return (
            dict(self._control_seconds),
            dict(self._control_calls),
            self._revalidated_requests,
        )

    def _log_control(self, now: float) -> None:
        seconds, calls, revalidated = self._control_counters()
        old_seconds, old_calls, old_revalidated = self._control_logged
        passes = calls["scheduler"] - old_calls["scheduler"]

        def per_pass(name: str) -> str:
            spent = seconds.get(name, 0.0) - old_seconds.get(name, 0.0)
            return f"{spent * 1000 / max(passes, 1):.3f}"

        reduces = calls["reduce"] - old_calls["reduce"]
        logger.info(
            "Flat control %.0fs: %d passes, ms/pass scheduler=%s poll=%s "
            "prepare_admission=%s finish_admission=%s rematch=%s reduce=%s "
            "(%.2f reduces/pass, %s); rematched %d requests; evict epoch %d; "
            "prefetches %d, leases %d",
            now - self._control_log_at,
            passes,
            per_pass("scheduler"),
            per_pass("poll"),
            per_pass("prepare_admission"),
            per_pass("finish_admission"),
            per_pass("rematch"),
            per_pass("reduce_collective"),
            reduces / max(passes, 1),
            "shm" if self._control_reducer is not None else "gloo",
            revalidated - old_revalidated,
            self._evict_epoch,
            len(self.prefetches),
            len(self._leases),
        )
        self._control_log_at = now
        self._control_logged = (seconds, calls, revalidated)

    def record_control_time(self, name: str, elapsed_seconds: float) -> None:
        # FLAT_MEMORY: Rank-local inclusive times overlap; never sum parent and child.
        self._control_seconds[name] = self._control_seconds.get(name, 0.0) + max(
            0.0, elapsed_seconds
        )
        self._control_calls[name] = self._control_calls.get(name, 0) + 1

    def _set_wait(self, state: _Prefetch, bucket: str, *, now=None) -> None:
        if state.wait_finalized:
            return
        now = max(state.wait_since, time.monotonic() if now is None else now)
        state.wait_seconds[state.wait_bucket] = state.wait_seconds.get(
            state.wait_bucket, 0.0
        ) + (now - state.wait_since)
        state.wait_since, state.wait_bucket = now, bucket

    def _publish_wait(self, state: _Prefetch, *, scope: str, seconds: dict) -> None:
        # FLAT_MEMORY: Overlapping request windows are not additive serving cost.
        self._diagnose(
            state,
            **{f"{scope}_total_ms": (state.wait_since - state.started) * 1000},
            **{
                f"{scope}_{bucket}_ms": seconds.get(bucket, 0.0) * 1000
                for bucket in _WAIT_BUCKETS
            },
        )

    def _freeze_pre_prepare(self, state: _Prefetch) -> None:
        if state.pre_prepare_seconds is None and not state.wait_finalized:
            self._set_wait(state, "unknown", now=state.lease_started)
            state.pre_prepare_seconds = state.wait_seconds.copy()
            self._publish_wait(
                state, scope="pre_prepare", seconds=state.pre_prepare_seconds
            )

    def _finalize_wait(self, state: _Prefetch, outcome: str, *, now=None) -> None:
        if state.wait_finalized:
            return
        self._set_wait(state, state.wait_bucket, now=now)
        state.wait_finalized = True
        self._publish_wait(state, scope="wait", seconds=state.wait_seconds)
        self._diagnose(state, wait_outcome=outcome, wait_finalized=1)

    def order_ready_requests(self, requests: list) -> list:
        # FLAT_MEMORY: A ready lease gets the first real admission attempt; the add
        # loop stops at the first NO_TOKEN, and a lease behind it would keep its
        # pinned pages, the very pages that made the earlier request not fit.
        ordered = (
            sorted(requests, key=lambda req: self.arrivals[req.rid])
            if self.ready_fcfs
            else requests
        )
        ready = {
            state.req.rid for state in self._active_leases() if state.phase == "ready"
        }
        if ready:
            return [req for req in ordered if req.rid in ready] + [
                req for req in ordered if req.rid not in ready
            ]
        return ordered

    def _active_leases(self) -> list[_Prefetch]:
        return [self.prefetches[rid] for rid in self._leases]

    def _drop_lease(self, rid: str) -> None:
        if rid in self._leases:
            self._leases.remove(rid)

    def has_hit(self, rid: str) -> bool:
        # Completed Flat reads are already device hits; admission never queues DMA.
        return False

    def match(self, key: RadixKey, req: Req, result: MatchResult) -> MatchResult:
        self.register_request(req)
        result = self._clip_streams(key, result, owner=req.rid)
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
        started = time.monotonic()
        state = _Prefetch(
            req=req,
            key=key,
            result=result,
            tail_hashes=tail_hashes,
            restore_episode=req.storage_restore_timing.active,
            started=started,
            wait_since=started,
            validated_epoch=self._evict_epoch,
            swa_evicted_before=req.kv.swa_evicted_seqlen if req.kv is not None else 0,
            original_device_tokens=device_hit_len,
            query_start=device_hit_len,
            queried_endpoint=device_hit_len,
        )
        self.prefetches[req.rid] = state
        # FLAT_MEMORY: The ledger describes the latest match episode, not all reentries.
        # Legacy request counters survive retraction; only this ledger starts anew.
        stats = req.flat_prefetch_stats
        episode = stats.get("flat_restore_wait_episode", 0) + 1
        for name in tuple(stats):
            if name.startswith(("flat_restore_wait_", "flat_restore_pre_prepare_")):
                del stats[name]
        self._diagnose(
            state,
            original_device_tokens=device_hit_len,
            wait_episode=episode,
            wait_schema_version=1,
            wait_outcome="open",
            wait_finalized=0,
        )
        if (
            not tail_hashes
            or len(tail_hashes) * self.cache.page_size < self.prefetch_threshold
        ):
            state.phase = "ready"
            self._set_wait(state, "ready_admission")
            return result
        if state.restore_episode is not None:
            state.restore_episode.outcome = "miss"
        transfers = self._lookup_transfers(tail_hashes)
        if not transfers or req.rid in self._no_restore_rids:
            # A request whose layer-wise restore failed recomputes once instead.
            self._no_restore_rids.discard(req.rid)
            state.phase, state.bypass = "ready", True
            self._set_wait(state, "ready_admission")
            return result
        # FLAT_MEMORY: Metadata probes never retain an evictable device anchor.
        self._set_wait(state, "query_pending")
        try:
            state.future = self.cache_linker.submit_lookup(req.rid, transfers)
        except BaseException:
            if state.restore_episode is not None:
                state.restore_episode.close("failed")
            self._finalize_wait(state, "lookup_failed")
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

    @_timed_control
    def _reduce(self, values: list[int], op) -> list[int]:
        started = time.monotonic()
        try:
            if self._control_reducer is not None:
                reduced = self._control_reducer.all_reduce(values, op)
                if reduced is not None:
                    return reduced
            tensor = torch.tensor(values, dtype=torch.int, device="cpu")
            self.cache._all_reduce_attn_groups(tensor, op)
            return tensor.tolist()
        finally:
            # FLAT_MEMORY: Includes waiting for peers, not just transport execution.
            self.record_control_time("reduce_collective", time.monotonic() - started)

    def _reduce_fused(self, segments: list[tuple[list[int], Any]]) -> list[list[int]]:
        # FLAT_MEMORY: One gloo MIN collective; MAX segments travel negated.
        # A collective costs ~0.3-1.1 ms at TP8 regardless of length.
        packed = []
        for values, op in segments:
            if op == torch.distributed.ReduceOp.MIN:
                packed.extend(values)
            elif op == torch.distributed.ReduceOp.MAX:
                packed.extend(-value for value in values)
            else:
                raise ValueError(f"Unsupported fused reduce op: {op}")
        reduced = self._reduce(packed, torch.distributed.ReduceOp.MIN) if packed else []
        parts, offset = [], 0
        for values, op in segments:
            part = reduced[offset : offset + len(values)]
            offset += len(values)
            parts.append(
                part
                if op == torch.distributed.ReduceOp.MIN
                else [-value for value in part]
            )
        return parts

    @_timed_control
    def poll(self) -> None:
        self.cache_linker.poll_queries()
        self._log_metrics()
        # FLAT_MEMORY: Every rank visits the same request list, never worker order.
        states = list(self.prefetches.values())
        streams = [
            state
            for rid, state in self._streaming.items()
            if rid not in self.prefetches and rid not in self._failed_streams
        ]
        for state in streams:
            if state.stream_consumed:
                # FLAT_MEMORY: The owner's forward already waited every layer; settle
                # the read so a failure retracts it before its result is processed.
                self.cache_linker.wait_load(state.req.rid, timeout=self.drain_timeout)
        if not states and not streams and not self.pending_offloads:
            # FLAT_MEMORY: All three are rank-consistent, so every rank skips alike.
            self._sync_reclaimed()
            return
        leases = self._active_leases() if states else []
        now = time.monotonic()
        offload_counts, offload_flags = self._offload_flags()
        # FLAT_MEMORY: Phases are rank-agreed; a queued request past its query
        # flags only no-ops, so poll stays O(in-flight), not O(waiting queue).
        queries = [s for s in states if s.phase in ("query", "allocate")]
        query_flags, query_failures = self._query_flags(queries, now=now)
        reads = [s for s in states if s.phase == "read" or s.streaming] + streams
        read_done, read_errors, read_nearly = self._read_flags(reads)
        # FLAT_MEMORY: All inputs are sampled before any transition, as the
        # sequential collectives did; transitions below keep their old order.
        (
            offload_counts,
            offload_flags,
            query_flags,
            query_failures,
            lease_expired,
            read_done,
            read_errors,
            read_nearly,
        ) = self._reduce_fused(
            [
                (offload_counts, torch.distributed.ReduceOp.MIN),
                (offload_flags, torch.distributed.ReduceOp.MAX),
                (query_flags, torch.distributed.ReduceOp.MIN),
                (query_failures, torch.distributed.ReduceOp.MAX),
                (self._lease_flags(leases, now=now), torch.distributed.ReduceOp.MAX),
                (read_done, torch.distributed.ReduceOp.MIN),
                (read_errors, torch.distributed.ReduceOp.MAX),
                (read_nearly, torch.distributed.ReduceOp.MIN),
            ]
        )
        self._apply_offloads(offload_counts, offload_flags)
        self._sync_reclaimed()
        if not states and not streams:
            return
        finished = []
        for state, ready, unexpired, refresh_unexpired, failed in zip(
            queries,
            query_flags[::3],
            query_flags[1::3],
            query_flags[2::3],
            query_failures,
        ):
            if not unexpired:
                self._fallback(state, "allocation_wait_timeout")
            elif not refresh_unexpired:
                self._fallback(state, "query_refresh_timeout")
            elif ready:
                finished.append((state, bool(failed)))
        self._finish_queries(finished)
        self._apply_lease_expiry(leases, lease_expired)
        self._apply_reads(reads, read_done, read_errors, read_nearly)
        for state in states:
            if state.cancelled and state.phase == "ready":
                self._forget(state)

    def _query_flags(
        self, states: list[_Prefetch], *, now: float
    ) -> tuple[list[int], list[int]]:
        flags, failures = [], []
        for s in states:
            done = s.phase == "query" and s.future is not None and s.future.done()
            flags += (
                int(done),
                int(
                    not s.allocation_wait_deadline
                    or s.phase not in ("query", "allocate")
                    or now < s.allocation_wait_deadline
                ),
                int(not s.refresh_deadline or s.phase != "query" or now < s.refresh_deadline),
            )
            # Only read where every rank agrees the query is done.
            failures.append(int(done and s.future.exception() is not None))
        return flags, failures

    def _finish_queries(self, finished: list[tuple[_Prefetch, bool]]) -> None:
        # FLAT_MEMORY: One collective for every query this poll finished, in rank-agreed
        # order; a per-query gloo reduce made poll cost grow with the arrival rate.
        pending = []
        for state, failed in finished:
            if state.cancelled:
                state.phase = "ready"
                continue
            restorable = []
            try:
                restorable = state.future.result()
            except Exception:
                self.io_errors += 1
                logger.exception(
                    "Flat storage lookup failed for request %s", state.req.rid
                )
            if failed:
                self._fallback(state, "query_failed", count=False)
                continue
            pending.append((state, restorable))
        if not pending:
            return
        # A 0/1 mask per query; MIN is AND, so the reduction intersects the
        # per-rank sets of restorable prefix lengths.
        masks = []
        for state, restorable in pending:
            mask = [0] * (len(state.tail_hashes) + 1)
            for pages in restorable:
                if 0 < pages < len(mask):
                    mask[pages] = 1
            masks.extend(mask)
        common = self._reduce(masks, torch.distributed.ReduceOp.MIN)
        offset = 0
        for state, _ in pending:
            width = len(state.tail_hashes) + 1
            segment = common[offset : offset + width]
            offset += width
            pages = max(
                (length for length, shared in enumerate(segment) if shared), default=0
            )
            self._finish_query(state, pages=pages)

    def _finish_query(self, state: _Prefetch, *, pages: int) -> None:
        state.query_generation = self.cache_linker.get_lookup_generation(state.req.rid)
        state.queried_endpoint = state.query_start + pages * self.cache.page_size
        state.phase = "allocate" if pages else "ready"
        self._set_wait(state, "unknown" if pages else "ready_admission")
        self._diagnose(
            state,
            queried_endpoint=state.queried_endpoint,
            query_ms=(time.monotonic() - state.started) * 1000,
        )

    def carry_reservation(self, adder, *, exclude: str | None = None) -> None:
        leases = [
            state
            for state in self._active_leases()
            if not state.admitted and state.req.rid != exclude
        ]
        adder.set_flat_restore_reservation(
            full_tokens=sum(state.full_reservation for state in leases),
            swa_tokens=sum(state.swa_reservation for state in leases),
        )

    @_timed_control
    def prepare_admission(
        self, requests: list, adder, *, has_chunked_req: bool
    ) -> list:
        # FLAT_MEMORY: This stage is shared by all ranks, before the local add loop.
        requests = self.order_ready_requests(requests)
        self._agree_requests(requests)
        self._admission_states = self._admission_window_states(requests)
        self._admission_rids = {state.req.rid for state in self._admission_states}
        self._admission_candidate = None
        for state in self._admission_states:
            state.admission_attempted = False
            if state.phase == "ready":
                self._set_wait(state, "ready_admission")
        self._note_evictions()
        self._refresh_stale_queries(self._admission_states)
        for lease in self._active_leases():
            if lease.phase == "ready":
                self._revalidate_lease(lease, adder)
        # FLAT_MEMORY: Reads start under chunked prefill too; plan_flat_restore
        # charges the admitted chunk's remaining input as running headroom.
        # A chunk the adder refused (SWA budget) is uncharged, so wait instead.
        self._prefill_busy = has_chunked_req
        if not has_chunked_req or adder.can_run_list:
            self._select_restores(adder)
            self._prefetch_ahead(requests, adder)
        else:
            for state in self._admission_states:
                if state.phase == "allocate":
                    self._set_wait(state, "chunked_prefill")
        self.carry_reservation(adder)
        return self.order_ready_requests(requests)

    def _admission_window_states(self, requests: list) -> list[_Prefetch]:
        # FLAT_MEMORY: Leases are always kept; requests is rank-agreed, so every
        # rank picks the same window.
        states, unleased = [], 0
        for req in requests:
            state = self.prefetches.get(req.rid)
            if state is None:
                continue
            if req.rid not in self._leases:
                if self.admission_window and unleased >= self.admission_window:
                    continue
                unleased += 1
            states.append(state)
        return states

    def _select_restores(self, adder) -> None:
        # FLAT_MEMORY: FC-MAS. DRAM and SSD restores hold separate lease slots,
        # so a DRAM hit never queues behind a multi-GB SSD read.
        candidates = [s for s in self._admission_states if s.phase == "allocate"]
        if not candidates:
            return
        leases = self._active_leases()
        slots = {
            True: self.max_ssd_leases - sum(bool(s.ssd_pages) for s in leases),
            False: self.max_dram_leases - sum(not s.ssd_pages for s in leases),
        }
        selected = blocked = False
        for state in self._rank_restores(candidates):
            if state.phase != "allocate":
                continue
            if blocked:
                self._set_wait(state, "candidate_queue")
                continue
            ssd = bool(state.ssd_pages)
            if slots[ssd] <= 0:
                self._set_wait(state, "existing_lease")
                continue
            if selected:
                # FLAT_MEMORY: The previous private allocation may have evicted this anchor.
                self._revalidate_unleased(state)
                if state.phase != "allocate":
                    continue
            self.carry_reservation(adder)
            if self._select_restore(state, adder):
                slots[ssd] -= 1
                selected = True
                # A whole-window rematch costs ~0.1 ms per request; only an
                # eviction by this private allocation can move another anchor.
                if self._note_evictions():
                    self._revalidate_ready_unleased()
            elif state.phase == "allocate":
                # FLAT_MEMORY: A budget veto stops lower-priority candidates from
                # overtaking it, so a large old restore cannot starve.
                self._admission_candidate = state.req.rid
                blocked = True

    def _prefetch_ahead(self, requests: list, adder) -> None:
        # FLAT_MEMORY: Arrival-order prefetch past the admission window, storage
        # straight into private GPU pages; a vetoed window restore is never overtaken.
        if not self.prefetch_pool_fraction or self._admission_candidate is not None:
            return
        leases = self._active_leases()
        slots = {
            True: self.max_ssd_leases - sum(bool(s.ssd_pages) for s in leases),
            False: self.max_dram_leases - sum(not s.ssd_pages for s in leases),
        }
        candidates = self._prefetch_candidates(
            requests, limit=max(0, slots[True]) + max(0, slots[False])
        )
        if not candidates:
            return
        free = self._prefetch_free_tokens(adder, leases=leases)
        if free is None:
            return
        self._revalidate_unleased_many(candidates)
        candidates = [state for state in candidates if state.phase == "allocate"]
        if not candidates:
            return
        for state in self._rank_restores(candidates):
            ssd = bool(state.ssd_pages)
            if slots[ssd] <= 0:
                continue
            self.carry_reservation(adder)
            if not self._select_restore(state, adder, prefetch_free=free):
                return
            slots[ssd] -= 1
            if self._note_evictions():
                # Remaining anchors may have moved; they rematch on a later pass.
                self._revalidate_ready_unleased()
                return

    def _prefetch_candidates(self, requests: list, *, limit: int) -> list[_Prefetch]:
        # requests is rank-agreed, so every rank picks the same states in order.
        candidates = []
        for req in requests[:_PREFETCH_SCAN_REQUESTS]:
            if len(candidates) >= limit:
                break
            if req.rid in self._admission_rids:
                continue
            state = self.prefetches.get(req.rid)
            if (
                state is not None
                and state.phase == "allocate"
                and not state.bypass
                and not state.lease_started
            ):
                candidates.append(state)
        return candidates

    def _prefetch_free_tokens(self, adder, *, leases: list[_Prefetch]) -> list[int] | None:
        # FLAT_MEMORY: Free pages only, net of what unadmitted leases still need,
        # so a prefetch never evicts the HBM prefix cache. MIN-agreed across ranks.
        allocator = adder.token_to_kv_pool_allocator
        if adder.is_hybrid_swa:
            full_free = allocator.full_available_size()
            swa_free = allocator.swa_available_size()
        else:
            full_free, swa_free = allocator.available_size(), 0
        unadmitted = [state for state in leases if not state.admitted]
        held = sum(state.loaded_tokens for state in unadmitted)
        cap = int(self.prefetch_pool_fraction * allocator.size_full) - held
        full = min(full_free - sum(s.full_reservation for s in unadmitted), cap)
        swa = swa_free - sum(s.swa_reservation for s in unadmitted)
        full, swa = self._reduce([full, swa], torch.distributed.ReduceOp.MIN)
        return [full, swa] if full > 0 else None

    def _note_evictions(self) -> bool:
        # FLAT_MEMORY: rank-agreed, so every rank skips or rematches the same states.
        current = self.cache.evicted_tokens_total
        evicted = self._reduce(
            [int(current != self._evictions_seen)], torch.distributed.ReduceOp.MAX
        )[0]
        self._evictions_seen = current
        if evicted:
            self._evict_epoch += 1
        return bool(evicted)

    def _revalidate_ready_unleased(self) -> None:
        # FLAT_MEMORY: Private allocation may evict another unleased ready hit.
        targets = [
            state
            for state in self._admission_states
            if self._is_unleased(state) and state.phase == "ready"
        ]
        self._revalidate_unleased_many(targets)
        for state in targets:
            if state.phase == "allocate":
                self._set_wait(state, "candidate_queue")

    def _rank_restores(self, candidates: list[_Prefetch]) -> list[_Prefetch]:
        # FLAT_MEMORY: Eq. 1 HRRN on SSD bytes; values are MAX-reduced so every
        # rank sorts identical integers.
        page = self.cache.page_size
        now = time.monotonic()
        local = []
        for state in candidates:
            local.append(
                self.cache_linker.restore_ssd_pages(
                    state.req.rid,
                    start=(len(state.result.device_indices) - state.query_start)
                    // page,
                    end=(state.queried_endpoint - state.query_start) // page,
                )
            )
            local.append(int((now - state.started) * 1000))
        agreed = self._reduce(local, torch.distributed.ReduceOp.MAX)
        wait_ms = {}
        for state, ssd_pages, waited in zip(candidates, agreed[::2], agreed[1::2]):
            state.ssd_pages = ssd_pages
            wait_ms[state.req.rid] = waited
        dram = [state for state in candidates if not state.ssd_pages]
        ssd = sorted(
            (state for state in candidates if state.ssd_pages),
            key=lambda state: (
                -wait_ms[state.req.rid] / state.ssd_pages,
                self.arrivals[state.req.rid],
            ),
        )
        return dram + ssd

    def _agree_requests(self, requests: list) -> None:
        # A rid containing NUL could alias another list; that only weakens this
        # cross-rank check, and json.dumps cost ~2x as much at a 6K queue.
        identity = "\0".join([req.rid for req in requests])
        digest = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
        signature = [len(requests), digest & 0x7FFFFFFF]
        minima, maxima = self._reduce_fused(
            [
                (signature, torch.distributed.ReduceOp.MIN),
                (signature, torch.distributed.ReduceOp.MAX),
            ]
        )
        if minima != maxima:
            raise RuntimeError("Flat restore admission queue differs across TP ranks")

    @staticmethod
    def _is_unleased(state: _Prefetch) -> bool:
        return (
            state.phase in ("ready", "allocate")
            and not state.bypass
            and not state.lease_started
        )

    @_timed_control
    def _refresh_stale_queries(self, states: list[_Prefetch]) -> None:
        if not states:
            return
        generation = self.cache_linker.storage_generation
        stale = [
            int(
                self._is_unleased(s)
                and s.future is not None
                and s.storage_probe_attempts < 2
                and s.queried_endpoint == s.query_start
                and s.queried_endpoint
                < len(s.key) // self.cache.page_size * self.cache.page_size
                and s.query_generation < generation
            )
            for s in states
        ]
        was_unleased = [self._is_unleased(state) for state in states]
        (stale,) = self._revalidate_unleased_many(
            [state for state, keep in zip(states, was_unleased) if keep],
            extra=[(stale, torch.distributed.ReduceOp.MAX)],
        )
        for state, refresh, keep in zip(states, stale, was_unleased):
            if not keep:
                continue
            if refresh and self._is_unleased(state) and state.phase == "ready":
                self._refresh_query(state, state.result, reason="storage_generation")

    def _revalidate_unleased(self, state: _Prefetch) -> None:
        self._revalidate_unleased_many([state])

    def _revalidate_unleased_many(
        self, states: list[_Prefetch], *, extra: list | None = None
    ) -> list[list[int]]:
        # FLAT_MEMORY: One collective per batch, independent of its size; extra
        # segments sampled before revalidation share it and are returned reduced.
        # Per-state handling only edits that state, so batching keeps the order.
        extra = extra or []
        if not states and not extra:
            return []
        # A state matched in this epoch keeps its result: no anchor it names was
        # evicted, and inserts only extend device coverage. Its checks still run.
        stale = [state for state in states if state.validated_epoch != self._evict_epoch]
        fresh = iter(self._rematch(stale))
        results = [
            next(fresh) if state.validated_epoch != self._evict_epoch else state.result
            for state in states
        ]
        sizes = [len(result.device_indices) for result in results]
        now = time.monotonic()
        timed_out = [
            int(bool(state.allocation_wait_deadline) and now >= state.allocation_wait_deadline)
            for state in states
        ]
        minima, maxima, timed_out, *reduced_extra = self._reduce_fused(
            [
                (sizes, torch.distributed.ReduceOp.MIN),
                (sizes, torch.distributed.ReduceOp.MAX),
                (timed_out, torch.distributed.ReduceOp.MAX),
                *extra,
            ]
        )
        if minima != maxima:
            raise RuntimeError("Flat restore device anchor differs across TP ranks")
        for state, result, expired in zip(states, results, timed_out):
            state.validated_epoch = self._evict_epoch
            self._apply_revalidation(state, result, timed_out=bool(expired))
        return reduced_extra

    def _apply_revalidation(
        self, state: _Prefetch, result: MatchResult, *, timed_out: bool
    ) -> None:
        state.result = result
        device_tokens = len(result.device_indices)
        if device_tokens >= state.queried_endpoint:
            state.phase = "ready"
            self._set_wait(state, "ready_admission")
            if state.queried_endpoint > state.query_start:
                self._diagnose(state, reason="device_covers_restore")
            return
        if timed_out:
            self._fallback(state, "allocation_wait_timeout")
            return
        was_ready = state.phase == "ready"
        if device_tokens < state.query_start:
            tail_tokens = (
                (len(state.key) - device_tokens)
                // self.cache.page_size
                * self.cache.page_size
            )
            if state.future is None and tail_tokens < self.prefetch_threshold:
                return
            if not self._refresh_query(state, result):
                self._fallback(state, "anchor_gap_refresh_exhausted")
        else:
            # FLAT_MEMORY: Receipt coordinates stay fixed; selection slices its hashes.
            state.phase = "allocate"
            self._set_wait(state, "unknown")
            self._diagnose(state, reason="receipt_reuse")
        if was_ready and not state.bypass:
            state.device_cover_reactivations += 1
            self._diagnose(
                state, device_cover_reactivations=state.device_cover_reactivations
            )

    def _can_refresh(self, state: _Prefetch, *, reason: str) -> bool:
        now = time.monotonic()
        if not state.refresh_deadline:
            state.refresh_deadline = now + self.lease_timeout
        attempts = (
            state.storage_probe_attempts
            if reason == "storage_generation"
            else state.anchor_refresh_attempts
        )
        # FLAT_MEMORY: Each purpose gets two submissions, not a per-generation budget.
        exhausted = self._reduce(
            [int(attempts >= 2 or now >= state.refresh_deadline)],
            torch.distributed.ReduceOp.MAX,
        )[0]
        return not exhausted

    @_timed_control
    def _rematch(self, states: list[_Prefetch]) -> list[MatchResult]:
        self._revalidated_requests += len(states)
        # FLAT_MEMORY: A metadata-only query's original node/indices may be evicted.
        # The caller agrees the anchor sizes across ranks.
        return [
            self._clip_streams(
                state.key,
                self.cache.match_prefix(MatchPrefixParams(key=state.key)),
                owner=state.req.rid,
            )
            for state in states
        ]

    def _clip_streams(self, key, result: MatchResult, *, owner: str) -> MatchResult:
        # FLAT_MEMORY: A still-streaming restore's pages are private to its owner;
        # another match stops at that restore's anchor, so a failed read never
        # reaches a second request. Rank-agreed: streams and matches are.
        if not self._streaming:
            return result
        matched = len(result.device_indices)
        anchor = matched
        for state in self._streaming.values():
            if state.req.rid == owner or matched <= state.read_anchor_tokens:
                continue
            if self._shares_path(key, state.key, state.read_anchor_tokens):
                anchor = min(anchor, state.read_anchor_tokens)
        if anchor == matched:
            return result
        from sglang.srt.mem_cache.radix_cache import RadixKey

        clipped = RadixKey(
            key.token_ids,
            extra_key=key.extra_key,
            is_bigram=key.is_bigram,
            limit=anchor,
            cache_salt=key.cache_salt,
        )
        return self.cache.match_prefix(MatchPrefixParams(key=clipped))

    def _shares_path(self, key, other, anchor: int) -> bool:
        # Equal through the first restored page means the match walked that restore.
        end = anchor + self.cache.page_size
        if (
            key.extra_key != other.extra_key
            or key.cache_salt != other.cache_salt
            or len(key) < end
            or len(other) < end
        ):
            return False
        return key.token_ids[anchor:end] == other.token_ids[anchor:end] and (
            key.token_ids[:anchor] == other.token_ids[:anchor]
        )

    def _refresh_query(
        self, state: _Prefetch, result: MatchResult, *, reason: str = "anchor_gap"
    ) -> bool:
        # FLAT_MEMORY: Direct linkers return the old future while its query is unfinished.
        finished = self._reduce(
            [int(state.future is None or state.future.done())],
            torch.distributed.ReduceOp.MIN,
        )[0]
        if not finished:
            return False
        query_start = len(result.device_indices)
        namespace = request_storage_namespace(state.key)
        tail_hashes = [
            namespace + value
            for value in self.tree_linker._tail_hashes(state.key, result, query_start)
        ]
        if not tail_hashes:
            state.phase = "ready"
            self._set_wait(state, "ready_admission")
            state.query_start = state.queried_endpoint = query_start
            state.tail_hashes = []
            self.cache_linker.release_lookup(state.req.rid)
            state.future = None
            return True
        transfers = self._lookup_transfers(tail_hashes)
        if not transfers:
            self._fallback(state, "unsupported_lookup", count=False)
            return True
        if not self._can_refresh(state, reason=reason):
            return False
        self._set_wait(state, "query_pending")
        try:
            future = self.cache_linker.refresh_lookup(state.req.rid, transfers)
        except BaseException:
            if state.restore_episode is not None:
                state.restore_episode.close("failed")
            self._finalize_wait(state, "lookup_failed")
            raise
        state.future = future
        state.query_start = state.queried_endpoint = query_start
        state.tail_hashes = tail_hashes
        state.refresh_attempts += 1
        if reason == "storage_generation":
            state.storage_probe_attempts += 1
        else:
            state.anchor_refresh_attempts += 1
        state.phase = "query"
        self._diagnose(
            state,
            refresh_attempts=state.refresh_attempts,
            storage_probe_attempts=state.storage_probe_attempts,
            anchor_refresh_attempts=state.anchor_refresh_attempts,
            refresh_reason=reason,
        )
        return True

    def _select_restore(
        self, state: _Prefetch, adder, *, prefetch_free: list[int] | None = None
    ) -> bool:
        # FLAT_MEMORY: The shared revalidation pass already refreshed this match.
        # prefetch_free is the agreed [full, swa] free-page budget of a prefetch,
        # spent on success; a prefetch veto leaves the state for the window.
        result = state.result
        device_tokens = len(result.device_indices)
        page = self.cache.page_size
        keys = state.tail_hashes[
            (device_tokens - state.query_start)
            // page : (state.queried_endpoint - state.query_start)
            // page
        ]
        footprint = self.tree_linker.plan_load_tokens(keys)
        fits = footprint is not None and (
            prefetch_free is None
            or (
                footprint.get(PoolName.KV, 0) <= prefetch_free[0]
                and footprint.get(PoolName.SWA, 0) <= prefetch_free[1]
            )
        )
        supported, fits = self._reduce(
            [int(footprint is not None), int(fits)], torch.distributed.ReduceOp.MIN
        )
        if not supported:
            self._fallback(state, "unsupported_restore_footprint")
            return False
        if not fits:
            return False
        state.result = result
        state.hold_node = result.last_device_node
        state.hold_params = self.cache.inc_lock_ref(state.hold_node).to_dec_params()
        plan = adder.plan_flat_restore(
            state.req,
            restored_prefix_len=state.queried_endpoint,
            full_tokens=footprint.get(PoolName.KV, 0),
            swa_tokens=footprint.get(PoolName.SWA, 0),
            check_pass_budget=False,
        )
        state.allocation_plan_attempts += 1
        self._diagnose(state, allocation_plan_attempts=state.allocation_plan_attempts)
        can_restore, never_fits, wait_eligible, reservation = self._sync_plan(plan)
        if not can_restore and prefetch_free is not None:
            # A prefetch veto is not a deferral; the window path owns waits and fallbacks.
            self._release_hold(state)
            return False
        state.allocation_wait_eligible = wait_eligible
        if not can_restore:
            self._observe_plan_gate(state, plan)
            self._record_plan_rejection(state, plan)
            self._release_hold(state)
            self.restore_deferrals += 1
            self._diagnose(state, reason=plan.reason or "restore_headroom")
            if never_fits:
                self._fallback(state, "restore_never_fits")
            elif wait_eligible and not state.allocation_wait_deadline:
                state.allocation_wait_started = time.monotonic()
                state.allocation_wait_deadline = (
                    state.allocation_wait_started + self.lease_timeout
                )
                self._diagnose(
                    state, allocation_wait_budget_ms=self.lease_timeout * 1000
                )
            return False
        self._finish_allocation_wait(state, "budget_available")
        state.full_reservation, state.swa_reservation = reservation
        if prefetch_free is not None:
            prefetch_free[0] -= footprint.get(PoolName.KV, 0) + state.full_reservation
            prefetch_free[1] -= footprint.get(PoolName.SWA, 0) + state.swa_reservation
        state.read_anchor_tokens = device_tokens
        state.read_endpoint = state.queried_endpoint
        state.loaded_tokens = state.read_endpoint - device_tokens
        state.lease_started = time.monotonic()
        self._freeze_pre_prepare(state)
        self._leases.append(state.req.rid)
        self._diagnose(
            state,
            lease_medium="ssd" if state.ssd_pages else "dram",
            lease_ssd_pages=state.ssd_pages,
            lease_concurrency=len(self._leases),
            read_anchor_tokens=device_tokens,
            read_endpoint=state.read_endpoint,
            prepared_tokens=state.loaded_tokens,
            prepare_ms=(state.lease_started - state.started) * 1000,
        )
        self._prepare(state, keys=keys)
        return True

    def _sync_plan(self, plan) -> tuple[bool, bool, bool, tuple[int, int]]:
        # FLAT_MEMORY: Durability-only backups cannot release additional GPU pages.
        # Source ownership is a retry opportunity, never speculative capacity credit.
        wait_eligible = plan.can_restore or (
            not plan.never_fits
            and plan.reason in ("full_decode_headroom", "swa_decode_headroom")
            and self.tree_linker.source_lock_count > 0
        )
        # The reservation rides along; callers use it only when can_restore.
        (can_restore, wait_eligible), (never_fits, full, swa) = self._reduce_fused(
            [
                (
                    [int(plan.can_restore), int(wait_eligible)],
                    torch.distributed.ReduceOp.MIN,
                ),
                (
                    [int(plan.never_fits), plan.full_reservation, plan.swa_reservation],
                    torch.distributed.ReduceOp.MAX,
                ),
            ]
        )
        return (
            bool(can_restore),
            bool(never_fits),
            bool(wait_eligible and not never_fits),
            (full, swa),
        )

    def _observe_plan_gate(self, state: _Prefetch, plan) -> None:
        # FLAT_MEMORY: The first rejection snapshot cannot explain a later peer veto.
        bucket = "restore_peer_veto" if plan.can_restore else f"restore_{plan.reason}"
        self._set_wait(state, bucket if bucket in _RESTORE_GATES else "restore_other")

    def _record_plan_rejection(self, state: _Prefetch, plan) -> None:
        if state.plan_rejection_recorded:
            return
        # FLAT_MEMORY: Gather once after a common veto, while the anchor is pinned.
        # Keep one real rank's comparison instead of combining unrelated extrema.
        snapshots = self.gather(
            {
                "can_restore": plan.can_restore,
                "reason": plan.reason,
                "full_required_tokens": plan.full_required_tokens,
                "full_available_tokens": plan.full_available_tokens,
                "swa_required_tokens": plan.swa_required_tokens,
                "swa_available_tokens": plan.swa_available_tokens,
                "pending_backups": len(self.pending_offloads),
                "source_locked_backups": self.tree_linker.source_lock_count,
            }
        )
        rank, snapshot = next(
            (rank, value)
            for rank, value in enumerate(snapshots)
            if not value["can_restore"]
        )
        self._diagnose(
            state,
            plan_reject_rank=rank,
            plan_reason=snapshot["reason"],
            **{
                f"plan_{key}": value
                for key, value in snapshot.items()
                if key not in ("can_restore", "reason")
            },
        )
        state.plan_rejection_recorded = True

    def _finish_allocation_wait(self, state: _Prefetch, outcome: str) -> None:
        if state.allocation_wait_started:
            self._diagnose(
                state,
                allocation_wait_ms=(time.monotonic() - state.allocation_wait_started)
                * 1000,
                allocation_wait_outcome=outcome,
            )
            state.allocation_wait_started = 0.0

    def _revalidate_lease(self, state: _Prefetch, adder) -> None:
        self.carry_reservation(adder, exclude=state.req.rid)
        plan = adder.plan_flat_restore(
            state.req,
            restored_prefix_len=state.read_endpoint,
            full_tokens=0,
            swa_tokens=0,
        )
        can_restore, never_fits, _, reservation = self._sync_plan(plan)
        if never_fits:
            self._fallback(state, "ready_never_fits")
        elif can_restore:
            state.full_reservation, state.swa_reservation = reservation
            self._set_wait(state, "ready_admission")
        else:
            self._observe_plan_gate(state, plan)
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
        self._set_wait(state, "read_pending")
        self.cache_linker.start_preparing_loads()

    def _poll_reads(self, states: list[_Prefetch]) -> None:
        done, errors, nearly = self._read_flags(states)
        done, errors, nearly = self._reduce_fused(
            [
                (done, torch.distributed.ReduceOp.MIN),
                (errors, torch.distributed.ReduceOp.MAX),
                (nearly, torch.distributed.ReduceOp.MIN),
            ]
        )
        self._apply_reads(states, done, errors, nearly)

    def _read_flags(
        self, states: list[_Prefetch]
    ) -> tuple[list[int], list[int], list[int]]:
        done, errors, nearly = [], [], []
        for state in states:
            ready, failed = False, False
            if state.phase == "read" or state.streaming:
                try:
                    ready = self.cache_linker.load_ready(state.req.rid)
                except FlatUnsafeIOError:
                    ready, failed = True, 2
                except Exception:
                    ready, failed = True, 1
            done.append(int(ready))
            errors.append(int(failed))
            nearly.append(int(not ready and self._near_done(state)))
        # _near_done is False without layer-wise restore, so no extra segment then.
        return done, errors, nearly if self.layer_wise_restore else []

    def _apply_reads(
        self,
        states: list[_Prefetch],
        done: list[int],
        errors: list[int],
        nearly: list[int],
    ) -> None:
        if not nearly:
            nearly = [0] * len(states)
        retired = []
        for state, ready, failed, near in zip(states, done, errors, nearly):
            if state.streaming:
                if ready and failed == 2:
                    raise FlatUnsafeIOError(
                        "Flat layer-wise restore did not quiesce on every TP rank; "
                        "its published pages may still be written"
                    )
                if ready and failed:
                    self._fail_stream(state)
                elif ready:
                    self._finish_stream(state)
                continue
            if not ready:
                if near and not state.cancelled and not state.revoke_reason:
                    self._stream_commit(state)
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
                ready_ns = time.monotonic_ns()
                self._commit(state)
                if state.restore_episode is not None:
                    state.restore_episode.restored_tokens = state.loaded_tokens
                    state.restore_episode.observe_ready(
                        ready_ns=ready_ns, endpoint="flat_all_tp_ready"
                    )
                # The result must be taken before forget_request drops it.
                retired.append((state, self.cache_linker.take_load_result(state.req.rid)))
                state.phase = "ready"
            self.cache_linker.forget_request(state.req.rid)
        self._record_read_metrics_many(retired)

    def _near_done(self, state: _Prefetch) -> bool:
        # FLAT_MEMORY: Admit once the rest of the read takes no longer than the
        # owner's own prefill, so layer-major I/O stays ahead of the forward and
        # no batch member waits on storage; the lead caps a wrong estimate.
        extend = self._owner_extend(state)
        if (
            not self.layer_wise_restore
            or state.phase != "read"
            or extend > self.layer_wise_max_extend_tokens
        ):
            return False
        done, total = self.cache_linker.load_progress(state.req.rid)
        bandwidth = self.cache_linker.read_bandwidth()
        if total <= 0 or bandwidth is None:
            return False
        budget = self.layer_wise_admit_lead
        if self.layer_wise_prefill_tokens_per_s > 0:
            budget = min(budget, extend / self.layer_wise_prefill_tokens_per_s)
        return (total - done) / bandwidth <= budget

    @staticmethod
    def _owner_extend(state: _Prefetch) -> int:
        # Tokens the owner prefills after its restore; the radix key omits the last.
        req = state.req
        return len(req.origin_input_ids) + len(req.output_ids) - state.read_endpoint

    def _stream_commit(self, state: _Prefetch) -> None:
        done, total = self.cache_linker.load_progress(state.req.rid)
        self._commit(state)
        node = state.hold_node
        state.stream_hold = (node, self.cache.inc_lock_ref(node).to_dec_params())
        state.stream_leaf_id = self.cache.resolve_node_handle(node).id
        state.streaming = True
        state.phase = "ready"
        self._streaming[state.req.rid] = state
        self.layer_wise_commits += 1
        self._diagnose(
            state, layer_wise_commit=1, layer_wise_remaining_bytes=total - done
        )

    def _finish_stream(self, state: _Prefetch) -> None:
        ready_ns = time.monotonic_ns()
        if state.restore_episode is not None:
            state.restore_episode.restored_tokens = state.loaded_tokens
            state.restore_episode.observe_ready(
                ready_ns=ready_ns, endpoint="flat_all_tp_ready"
            )
        self._record_read_metrics(state)
        req = state.req
        if req._cache_breakdown_computed:
            # FLAT_MEMORY: The owner's first extend settled its cache sources before
            # this stream's DRAM/SSD counts existed; settle again on the same prefix.
            finalize_flat_cache_accounting(
                req, prefix_len=req.cached_tokens_device + req.cached_tokens_storage
            )
        self._release_stream_hold(state)
        state.streaming = False
        self._streaming.pop(state.req.rid, None)
        self.cache_linker.forget_request(state.req.rid)

    def _release_stream_hold(self, state: _Prefetch) -> None:
        if state.stream_hold is not None:
            node, params = state.stream_hold
            self.cache.dec_lock_ref(node, params)
            state.stream_hold = None

    def _fail_stream(self, state: _Prefetch) -> None:
        # FLAT_MEMORY: Rank-agreed (read errors are MAX-reduced). Only the owner
        # can hold these pages (_clip_streams), so dropping them and recomputing
        # the owner serves nothing computed from a failed read.
        rid = state.req.rid
        self.io_errors += 1
        self.layer_wise_fallbacks += 1
        logger.warning("Flat layer-wise restore failed; recomputing request %s", rid)
        self._release_stream_hold(state)
        if rid not in self.prefetches:
            # Admitted: the scheduler retracts the owner, then finish_failed_streams.
            if state.restore_episode is not None:
                state.restore_episode.close("failed")
            self._failed_streams[rid] = state
            return
        self._release_hold(state)
        self._drop_stream_pages(state)
        self._end_stream(state)
        self._restore_request(state)
        self._fallback(state, "stream_read_failed", count=False)
        # The match named the dropped pages; admission must rematch.
        state.validated_epoch = -1

    def take_failed_streams(self) -> set[str]:
        return set(self._failed_streams)

    def finish_failed_streams(self, rids) -> None:
        # FLAT_MEMORY: After the owners are retracted their locks are gone, so the
        # pages are unshared device leaves again. Called in rank-agreed order.
        for rid in sorted(rids):
            state = self._failed_streams.pop(rid)
            self._drop_stream_pages(state)
            self._end_stream(state)
            self._no_restore_rids.add(rid)

    def _end_stream(self, state: _Prefetch) -> None:
        state.streaming = False
        self._streaming.pop(state.req.rid, None)
        self.cache_linker.forget_request(state.req.rid)

    def _drop_stream_pages(self, state: _Prefetch) -> None:
        # Restored nodes sit wholly below the anchor; a split only cuts them further.
        tree = self.cache.tree_core
        node = tree.try_node_by_id(state.stream_leaf_id)
        tracker = collections.defaultdict(int)
        while (
            node is not None
            and node is not self.cache.root_node
            and self._node_depth(node) > state.read_anchor_tokens
        ):
            if not tree._is_device_leaf(node):
                raise FlatUnsafeIOError(
                    "Flat layer-wise restore failed but another request holds its "
                    "pages; refusing to serve them"
                )
            parent = node.parent
            self.cache._evict_device_leaf(node.id, tracker)
            node = parent

    def _node_depth(self, node) -> int:
        depth = 0
        while node is not None and node is not self.cache.root_node:
            depth += len(node.key)
            node = node.parent
        return depth

    def layer_wise_consumer(self, rids: list[str]) -> int:
        # FLAT_MEMORY: Only a stream's owner reads its pages (_clip_streams), so a
        # batch waits on its own members' streams and no one else's.
        owners = [rid for rid in rids if rid in self._streaming]
        if not owners:
            return -1
        for rid in owners:
            self._streaming[rid].stream_consumed = True
        return self.cache_linker.layer_counter_for(owners)

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
        self._set_wait(state, "ready_admission", now=state.ready_started)
        state.req.storage_read_tokens = state.loaded_tokens
        state.req.storage_read_latency_ms = (state.ready_started - state.started) * 1000
        self._diagnose(
            state,
            committed_tokens=state.loaded_tokens,
            ready_endpoint=state.read_endpoint,
            ready_ms=(state.ready_started - state.started) * 1000,
        )

    def _record_read_metrics(self, state: _Prefetch) -> None:
        self._record_read_metrics_many(
            [(state, self.cache_linker.take_load_result(state.req.rid))]
        )

    def _record_read_metrics_many(self, reads: list[tuple[_Prefetch, Any]]) -> None:
        # FLAT_MEMORY: One fused collective for every read retired this poll,
        # in rank-agreed order; it used to be three per read.
        if not reads:
            return
        local, elapsed = [], []
        for state, result in reads:
            # FLAT_MEMORY: Source validity rides the TP reduction, independently of timing.
            valid = int(
                (result.dram_pages + result.ssd_pages) * self.cache.page_size
                == state.loaded_tokens
                and 0 <= result.mixed_pages <= result.ssd_pages
            )
            local += [result.dram_pages, result.ssd_pages, result.mixed_pages, valid]
            elapsed.append(int(result.elapsed_seconds * 1_000_000))
        minima, maxima, elapsed = self._reduce_fused(
            [
                (local, torch.distributed.ReduceOp.MIN),
                (local, torch.distributed.ReduceOp.MAX),
                (elapsed, torch.distributed.ReduceOp.MAX),
            ]
        )
        for index, (state, _) in enumerate(reads):
            self._apply_read_metrics(
                state,
                minima=minima[4 * index : 4 * index + 4],
                maxima=maxima[4 * index : 4 * index + 4],
                elapsed_us=elapsed[index],
            )

    def _apply_read_metrics(
        self, state: _Prefetch, *, minima: list[int], maxima: list[int], elapsed_us: int
    ) -> None:
        state.restore_source = RestoreSource(
            int(bool(maxima[0] or maxima[2])) | (int(bool(maxima[1])) << 1),
            bool(minima[3]),
        )
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

    def is_ready(self, rid: str, *, rem_chunk_tokens: int | None = None) -> bool:
        state = self.prefetches.get(rid)
        if state is None:
            return True
        # FLAT_MEMORY: Outside this pass's window the state was not revalidated.
        if self._admission_rids is not None and rid not in self._admission_rids:
            return False
        if (
            state.streaming
            and rem_chunk_tokens is not None
            and self._owner_extend(state) > rem_chunk_tokens
        ):
            # A stream's owner prefills in one chunk, so a failed read is known
            # before its result; it waits for a pass with that much budget.
            return False
        return state.phase == "ready"

    def before_admission(self, req: Req, adder) -> None:
        state = self.prefetches.get(req.rid)
        if state is None:
            return
        if state.phase != "ready":
            raise RuntimeError("Flat attempted admission before storage readiness")
        if req.rid in self._leases:
            # FLAT_MEMORY: Ordinary admission now charges the owner's suffix/decode.
            self.carry_reservation(adder, exclude=req.rid)
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

    @_timed_control
    def finish_admission(self, adder, *, can_progress: bool) -> None:
        states = self._admission_states
        if states:
            admitted = [int(state.admitted) for state in states]
            minima, maxima = self._reduce_fused(
                [
                    (admitted, torch.distributed.ReduceOp.MIN),
                    (admitted, torch.distributed.ReduceOp.MAX),
                ]
            )
            if minima != maxima:
                raise RuntimeError("Flat request admission differs across TP ranks")
            for state, accepted in zip(states, minima):
                if accepted:
                    self._forget(state, keep_arrival=True)
        watched = self._active_leases()
        candidate = self.prefetches.get(self._admission_candidate)
        if candidate is not None and candidate.req.rid not in self._leases:
            watched.append(candidate)
        now = time.monotonic()
        # FLAT_MEMORY: A rank-local gate may skip admission without rejecting the KV.
        flags = self._reduce(
            [int(can_progress)]
            + [
                value
                for state in watched
                for value in (
                    int(state.admission_attempted),
                    int(
                        state.phase == "allocate"
                        and state.allocation_wait_eligible
                        and now < state.allocation_wait_deadline
                    ),
                )
            ],
            torch.distributed.ReduceOp.MIN,
        )
        can_progress = flags[0]
        for state, admission_attempted, can_wait in zip(
            watched, flags[1::2], flags[2::2]
        ):
            if (
                not state.bypass
                and not can_progress
                and (
                    (state.phase == "allocate" and not can_wait)
                    or (state.phase == "ready" and admission_attempted)
                )
            ):
                reason = (
                    "allocation_wait_timeout"
                    if state.phase == "allocate" and state.allocation_wait_eligible
                    else "no_admission_progress"
                )
                self._fallback(state, reason)
        self._admission_states = []
        self._admission_rids = None
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
        if state.restore_episode is not None:
            state.restore_episode.adopt(
                tokens=reused,
                source=state.restore_source if not dropped else RestoreSource(),
            )
        self.admission_dropped_tokens += dropped
        if dropped:
            for name in ("flat_dram", "flat_ssd", "flat_mixed"):
                req.flat_prefetch_stats[name] = 0 if reused == 0 else None
        now = time.monotonic()
        self._finalize_wait(state, "admitted", now=now)
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
        # FLAT_MEMORY: The first admission fixes the scope; a retracted request
        # re-admits with its own earlier tokens cached and must not look like turn 2.
        if req.flat_media_hint is None:
            req.flat_media_hint = self._classify_scope(
                cached_tokens=len(req.prefix_indices),
                input_tokens=len(req.origin_input_ids),
            )

    def on_admitted(self, requests: list) -> None:
        for req in requests:
            state = self.prefetches.get(req.rid)
            if state is not None:
                self._handoff(state)
                self._forget(state, keep_arrival=True)

    def _lease_flags(self, leases: list[_Prefetch], *, now: float) -> list[int]:
        # FLAT_MEMORY: A ready lease behind a running chunked prefill is waiting on
        # compute that is making progress, not on a stuck read.
        return [
            int(
                now - state.lease_started >= self.lease_timeout
                and not (state.phase == "ready" and self._prefill_busy)
            )
            for state in leases
        ]

    def _apply_lease_expiry(self, leases: list[_Prefetch], expired: list[int]) -> None:
        # A lease the query step already revoked keeps its first reason.
        for state, lapsed in zip(leases, expired):
            if lapsed and not state.revoke_reason:
                self._fallback(state, "lease_expired")

    def _fallback(self, state: _Prefetch, reason: str, *, count: bool = True) -> None:
        if state.restore_episode is not None:
            state.restore_episode.close(
                "cancelled"
                if state.cancelled
                else "failed" if "failed" in reason else "discarded"
            )
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
        self._finish_allocation_wait(state, reason)
        self._release_hold(state)
        self._drop_lease(state.req.rid)
        state.full_reservation = state.swa_reservation = 0
        state.phase, state.bypass = "ready", True
        self._set_wait(state, "ready_admission")
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
        if state.restore_episode is not None:
            state.restore_episode.close("cancelled")
        self._finalize_wait(state, "cancelled")
        self._finish_allocation_wait(state, "cancelled")
        if state.phase != "read":
            self._forget(state)

    def _release_hold(self, state: _Prefetch) -> None:
        if state.hold_params is not None:
            self.cache.dec_lock_ref(state.hold_node, state.hold_params)
            state.hold_node = state.hold_params = None

    def _forget(self, state: _Prefetch, *, keep_arrival: bool = False) -> None:
        if state.prepared is not None or state.phase == "read":
            raise RuntimeError("Flat cannot forget a restore before I/O drain")
        self._finalize_wait(state, "cancelled" if state.cancelled else "forgotten")
        self._release_hold(state)
        # A streaming read still owns its linker load until _finish_stream.
        if not state.streaming:
            self.cache_linker.forget_request(state.req.rid)
        self.tree_linker.hit_markers.pop(state.req.rid, None)
        self.prefetches.pop(state.req.rid, None)
        self._drop_lease(state.req.rid)
        if not keep_arrival:
            self.arrivals.pop(state.req.rid, None)

    def _classify_scope(self, *, cached_tokens: int, input_tokens: int) -> int:
        # FLAT_MEMORY: Table 2: nothing cached at arrival is turn 1 (SSD); a short
        # new suffix on a cached prefix is a short turn (DRAM) unless L_agent is
        # set and the context reaches it.
        if not self.scope_media_hints:
            return MEDIA_HINT_ANY
        if cached_tokens <= 0:
            return MEDIA_HINT_SSD
        if (
            self.scope_agent_context_tokens
            and input_tokens >= self.scope_agent_context_tokens
        ):
            return MEDIA_HINT_SSD
        if input_tokens - cached_tokens < self.scope_short_turn_tokens:
            return MEDIA_HINT_DRAM
        return MEDIA_HINT_SSD

    @contextmanager
    def write_scope(self, req: Req):
        # FLAT_MEMORY: Inserts run their BackupKV actions synchronously, so the
        # inserting request's scope is known for exactly the nodes it creates.
        self._write_scope_hint = (
            MEDIA_HINT_ANY if req.flat_media_hint is None else req.flat_media_hint
        )
        try:
            yield
        finally:
            self._write_scope_hint = MEDIA_HINT_ANY

    def _node_media_hint(self, node) -> int:
        if not self.scope_media_hints:
            return self._write_scope_hint
        # FLAT_MEMORY: A node already reached by another insert is a shared prefix.
        if node.hit_count > 1:
            return MEDIA_HINT_DRAM
        if (
            self.scope_shared_head_tokens
            and self._node_start_tokens(node) < self.scope_shared_head_tokens
        ):
            return MEDIA_HINT_DRAM
        return self._write_scope_hint

    @staticmethod
    def _node_start_tokens(node) -> int:
        start, parent = 0, node.parent
        while parent is not None and parent.key is not None:
            start += len(parent.key)
            parent = parent.parent
        return start

    def offload_nodes(self, node_ids) -> None:
        for node_id in node_ids:
            node = self.cache.resolve_node_handle(node_id)
            if node.external_cache_stored or node.write_through_pending_id is not None:
                continue
            hint = self._node_media_hint(node)
            self.scope_hint_tokens[
                {MEDIA_HINT_DRAM: "dram", MEDIA_HINT_SSD: "ssd"}.get(hint, "any")
            ] += len(node.key)
            namespace = request_storage_namespace(node.key)
            self.tree_linker._offload_node(
                node_id, key_namespace=namespace, media_hint=hint
            )
            for page_hash in node.hash_value or ():
                self._page_nodes[namespace + page_hash] = (node.id, page_hash)
            # FLAT_MEMORY: Pending deduplicates writes; it is not durable storage state.
            node.external_cache_stored = False

    def _sync_reclaimed(self) -> None:
        # FLAT_MEMORY: Each rank's store reclaims on its own, but external_cache_stored
        # gates write-through on every rank, so the cleared set must be the union.
        self._polls += 1
        if self._polls % _RECLAIM_SYNC_POLLS:
            return
        local = self._reclaimed_node_ids()
        if not self._reduce([len(local)], torch.distributed.ReduceOp.MAX)[0]:
            return
        for node_id in sorted({i for ids in self.gather(sorted(local)) for i in ids}):
            node = self.cache.tree_core.try_node_by_id(node_id)
            # A pending write is re-decided by its own completion.
            if (
                node is not None
                and node.external_cache_stored
                and node.write_through_pending_id is None
            ):
                node.external_cache_stored = False
                self.reclaim_invalidated_nodes += 1

    def _reclaimed_node_ids(self) -> set[int]:
        found = set()
        for page in self.cache_linker.take_reclaimed_pages():
            entry = self._page_nodes.pop(page, None)
            if entry is None:
                continue
            node_id, page_hash = entry
            node = self.cache.tree_core.try_node_by_id(node_id)
            # A split moves leading pages into a new parent, so walk up.
            while node is not None and page_hash not in (node.hash_value or ()):
                node = node.parent
            if node is not None and node.id is not None:
                found.add(node.id)
        if len(self._page_nodes) > _MAX_PAGE_NODES:
            self._page_nodes = {
                page: entry
                for page, entry in self._page_nodes.items()
                if self.cache.tree_core.try_node_by_id(entry[0]) is not None
            }
        return found

    def replace_pending_offload_node(self, ack_id, old_node_id, new_node_ids) -> None:
        self.tree_linker.replace_pending_offload_node(ack_id, old_node_id, new_node_ids)

    def _poll_offloads(self) -> None:
        self._apply_offloads(*self._fused_offload_flags())

    def _fused_offload_flags(self) -> tuple[list[int], list[int]]:
        counts, flags = self._offload_flags()
        return self._reduce_fused(
            [
                (counts, torch.distributed.ReduceOp.MIN),
                (flags, torch.distributed.ReduceOp.MAX),
            ]
        )

    def _offload_flags(self) -> tuple[list[int], list[int]]:
        counts = [
            self.tree_linker.num_source_safe_offloads(),
            self.tree_linker.num_completed_offloads(),
        ]
        # FLAT_MEMORY: Fixed-width slots let the outcome flags share the count
        # collective; every rank holds at least the agreed MIN of completions.
        results = self.cache_linker.peek_completed_offload_results(
            _OFFLOAD_FLAG_SLOTS
        )
        flags = [0] * (4 * _OFFLOAD_FLAG_SLOTS)
        for slot, result in enumerate(results):
            flags[4 * slot : 4 * slot + 4] = (
                int(not result.success),
                int(result.capacity_rejected),
                int(not result.success and not result.capacity_rejected),
                int(result.unsafe),
            )
        return counts, flags

    def _apply_offloads(self, counts: list[int], slot_flags: list[int]) -> None:
        source_count, durable_count = counts
        # Completions past the slot window retire on a later poll.
        durable_count = min(durable_count, _OFFLOAD_FLAG_SLOTS)
        successes = []
        if durable_count:
            for _ in range(durable_count):
                self.cache_linker.pop_completed_offload_result()
            # FLAT_MEMORY: Pressure on one rank must not mask another rank's I/O error.
            flags = slot_flags[: 4 * durable_count]
            if any(flags[3::4]):
                raise FlatUnsafeIOError(
                    "Flat offload did not quiesce on every TP rank; "
                    "outstanding source or staging ownership is retained"
                )
            successes = [not failed for failed in flags[0::4]]
            self.capacity_pressure_offloads += sum(flags[1::4])
            self.io_errors += sum(flags[2::4])
        # FLAT_MEMORY: Check common unsafe outcomes before either ownership transition.
        self.tree_linker.release_source_safe_offloads(source_count)
        if durable_count:
            collector = self.cache.storage_metrics_collector
            if collector is not None:
                for success, pending in zip(successes, self.pending_offloads):
                    if success:
                        collector.log_backuped_tokens(pending.token_count)
                        collector.log_storage_write_tokens(pending.token_count)
            self.tree_linker.commit_completed_offloads(successes)

    def _log_metrics(self) -> None:
        collector = self.cache.storage_metrics_collector
        now = time.monotonic()
        if collector is not None and now - self._last_metrics_log >= 1.0:
            collector.log_storage_metrics(self.cache_linker.get_stats())
            self._last_metrics_log = now

    def is_idle(self) -> bool:
        return not self.prefetches and not self.pending_offloads and not self._streaming

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
        self._page_nodes.clear()
        self._admission_states = []
        self._admission_rids = None
        self._admission_candidate = None
        self._leases = []
        self._prefill_busy = False

    def close(self) -> None:
        self.reset()
        self.tree_linker.close()
        if self._control_reducer is not None:
            self._control_reducer.close()
            self._control_reducer = None

    def start_layer_wise_loading(self) -> int:
        # Flat loads complete before admission, so no forward-stream layer wait is needed.
        return -1

    def snapshot(self) -> dict:
        result = self.cache_linker.get_flat_memory_stats()
        result.update(
            pending_backups=len(self.pending_offloads),
            backup_lifecycle=self.tree_linker.get_offload_stats(),
            pending_prefetches=len(self.prefetches),
            flat_io_errors=self.io_errors,
            flat_capacity_pressure_offloads=self.capacity_pressure_offloads,
            flat_restore_lease_active=len(self._leases),
            flat_restore_ssd_leases_active=sum(
                bool(state.ssd_pages) for state in self._active_leases()
            ),
            flat_restore_lease_fallbacks=self.lease_fallbacks,
            flat_restore_deferrals=self.restore_deferrals,
            flat_restore_ready_dropped_tokens=self.ready_dropped_tokens,
            flat_restore_admission_dropped_tokens=self.admission_dropped_tokens,
            flat_restore_revalidated_requests=self._revalidated_requests,
            flat_reclaim_invalidated_nodes=self.reclaim_invalidated_nodes,
            flat_restore_layer_wise_commits=self.layer_wise_commits,
            flat_restore_streaming=len(self._streaming),
            **{
                f"flat_scope_hint_{medium}_tokens": tokens
                for medium, tokens in self.scope_hint_tokens.items()
            },
            application_host_staging_bytes=0,
            host_pool_bytes=0,
            gds_mode="compat",
        )
        result.update(
            (f"flat_restore_control_{name}_seconds", seconds)
            for name, seconds in self._control_seconds.items()
        )
        result.update(
            (f"flat_restore_control_{name}_calls", calls)
            for name, calls in self._control_calls.items()
        )
        return result

    @_timed_control
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
