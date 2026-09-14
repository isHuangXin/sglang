"""FLAT_MEMORY: Completion-owned Mooncake GPU restores alongside Unified HiCache."""

from __future__ import annotations

import copy
import logging
import threading
import time
from concurrent.futures import Future
from queue import Queue
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.mooncake_store.gds_payload import GPUReadResult
from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components import LinkerTransferPhase
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    PreparedLinkerLoad,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.utils import get_hash_str

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


class _TieredWorker:
    def __init__(self, name):
        self.queue = Queue()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def submit(self, function, *args):
        future = Future()
        self.queue.put((future, function, args))
        return future

    def _run(self):
        while (job := self.queue.get()) is not None:
            future, function, args = job
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function(*args))
            except BaseException as error:
                future.set_exception(error)
            finally:
                # A worker must not retain the previous operation's staging while idle.
                del job, future, function, args

    def close(self):
        self.queue.put(None)
        self.thread.join(timeout=120.0)
        if self.thread.is_alive():
            raise RuntimeError("Mooncake worker did not stop; resources retained")


class _LoadRequest(msgspec.Struct, kw_only=True):
    rid: str
    prefix_indices: torch.Tensor
    last_node: Any
    kv: Any
    priority: int


class _TieredPrefetch(msgspec.Struct, kw_only=True):
    req: Any
    key: RadixKey
    anchor: Any
    prefix_len: int
    device_lock: Any
    host_lock: Any
    hashes: list[str]
    lookup_transfers: list
    started: float
    query: Future | None = None
    io: Future | None = None
    prepared: PreparedLinkerLoad | None = None
    result: GPUReadResult | None = None
    cancelled: bool = False
    full_tokens: int = 0
    swa_tokens: int = 0


class TieredMooncakeRuntime:
    def __init__(self, *, cache: UnifiedRadixCache):
        # This leaf needs live tree/allocator state; it never becomes cache.linker.
        self.cache = cache
        self.controller = cache.cache_controller
        self.storage = self.controller.storage_backend
        self.payload = self.controller.tiered_payload
        self.loads = UnifiedCacheLinkerWrapper(cache, None, configure_tree=False)
        self.pending: dict[str, _TieredPrefetch] = {}
        self.sources: dict[str, list[tuple[int, int, int]]] = {}
        self.loaded: dict[str, int] = {}
        self.latencies: dict[str, tuple[float, int]] = {}
        self.query_worker = _TieredWorker("mooncake-query")
        self.io_worker = _TieredWorker("mooncake-gds")
        self.closed = False

    def gather(self, value):
        if self.cache.tp_world_size == 1:
            return [value]
        values = [None] * self.cache.tp_world_size
        torch.distributed.all_gather_object(values, value, group=self.cache.tp_group)
        return values

    @property
    def full_reserved_tokens(self):
        return sum(op.full_tokens for op in self.pending.values())

    @property
    def swa_reserved_tokens(self):
        return sum(op.swa_tokens for op in self.pending.values())

    def is_idle(self):
        return not self.pending

    @staticmethod
    def _prefix_length(node):
        length = 0
        while node.parent is not None:
            length += len(node.key)
            node = node.parent
        return length

    def prefetch(self, req: Req) -> None:
        if (
            self.closed
            or req.rid in self.pending
            or req.session is not None
            or req.input_embeds is not None
            or req.positional_embed_overrides is not None
        ):
            return
        page = self.cache.page_size
        limit = req._compute_max_prefix_len(len(req.full_untruncated_fill_ids))
        key = RadixKey(
            req.full_untruncated_fill_ids[:limit],
            extra_key=req.extra_key,
            cache_salt=req.cache_salt,
        ).page_aligned(page)
        anchor = req.best_match_node
        prefix_len = self._prefix_length(self.cache.resolve_node_handle(anchor))
        if prefix_len != len(req.prefix_indices) + req.host_hit_length:
            return
        if len(key) - prefix_len < self.cache.prefetch_threshold:
            return
        hashes = get_hash_str(key, page_size=page)[prefix_len // page :]
        transfers = [
            component.build_external_linker_transfer(
                LinkerTransferPhase.LOOKUP, None, hashes
            )
            for component in self.cache._components_tuple
        ]
        if any(transfer is None for transfer in transfers):
            return
        op = _TieredPrefetch(
            req=req,
            key=key,
            anchor=anchor,
            prefix_len=prefix_len,
            device_lock=self.cache.inc_lock_ref(anchor).to_dec_params(),
            host_lock=self.cache.inc_host_lock_ref(anchor).to_dec_params(),
            hashes=hashes,
            lookup_transfers=transfers,
            started=time.monotonic(),
        )
        self.pending[req.rid] = op
        op.query = self.query_worker.submit(self._query, op)

    def query_storage_hit_length(self, anchor, tokens, last_hash):
        extra_key, salt = self.cache.tree_core.prefetch_anchor_info(anchor)
        key = RadixKey(tokens, extra_key=extra_key, cache_salt=salt).page_aligned(
            self.cache.page_size
        )
        if len(key) < self.cache.prefetch_threshold:
            return 0
        hashes = get_hash_str(key, last_hash, page_size=self.cache.page_size)
        transfers = [
            component.build_external_linker_transfer(
                LinkerTransferPhase.LOOKUP, None, hashes
            )
            for component in self.cache._components_tuple
        ]
        boundaries, error = [], None
        try:
            if not any(transfer is None for transfer in transfers):
                boundaries = self.payload.lookup(transfers)
        except Exception as exc:
            error = str(exc)
        ranks = self.gather((error, boundaries))
        if any(rank[0] for rank in ranks):
            logger.warning(
                "Mooncake storage query failed: %s", [rank[0] for rank in ranks]
            )
            return 0
        common = set(boundaries)
        for _, other in ranks:
            common.intersection_update(other)
        return max(common, default=0) * self.cache.page_size

    def _query(self, op):
        # No CUDA allocation or collective is allowed on this worker.
        if op.cancelled:
            return []
        self.controller.backup_idle_event.wait(
            timeout=self.controller._backup_wait_timeout
        )
        return [] if op.cancelled else self.payload.lookup(op.lookup_transfers)

    def _report(self, rid, op):
        query_ready, boundaries, error = op.query.done(), [], None
        if query_ready:
            try:
                boundaries = op.query.result()
            except Exception as exc:
                error = str(exc)
        io_done = op.io is not None and op.io.done()
        safe, success, media = False, False, {}
        if io_done:
            try:
                op.io.result()
            except Exception as exc:
                error = str(exc)
            safe = op.result.safe_to_release
            success = op.result.success
            media = {name: list(masks) for name, masks in op.result.pool_media.items()}
        return (
            rid,
            query_ready,
            boundaries,
            op.cancelled,
            error,
            io_done,
            safe,
            success,
            media,
        )

    def poll(self) -> None:
        reports = self.gather(
            [self._report(rid, op) for rid, op in self.pending.items()]
        )
        identities = [[row[0] for row in rank] for rank in reports]
        if any(ids != identities[0] for ids in identities):
            raise RuntimeError("Mooncake tiered TP prefetch ordering diverged")
        for index, (rid, op) in enumerate(list(self.pending.items())):
            ranks = [rank[index] for rank in reports]
            op.cancelled |= any(rank[3] or rank[4] for rank in ranks)
            if op.io is not None:
                if not all(rank[5] for rank in ranks):
                    continue
                if not all(rank[6] for rank in ranks):
                    raise RuntimeError(
                        "Mooncake GPU I/O completion is unknown; allocations retained"
                    )
                if op.cancelled or not all(rank[7] for rank in ranks):
                    self._discard(rid, op)
                else:
                    count = len(op.prepared.hit.tail_hashes)
                    media = {}
                    pools = set(ranks[0][8])
                    if any(set(rank[8]) != pools for rank in ranks):
                        raise RuntimeError(
                            "Mooncake TP completed physical pools diverged"
                        )
                    for pool in pools:
                        media[pool] = [0] * count
                        for rank in ranks:
                            masks = rank[8][pool]
                            if len(masks) != count:
                                raise RuntimeError(
                                    "Mooncake TP completed media lengths diverged"
                                )
                            for page, mask in enumerate(masks):
                                media[pool][page] |= mask
                    self._commit(rid, op, media)
            elif all(rank[1] for rank in ranks):
                if op.cancelled:
                    self._discard(rid, op)
                    continue
                # One private load/staging buffer at a time bounds GPU staging across requests.
                if any(other.io is not None for other in self.pending.values()):
                    continue
                common = set(ranks[0][2])
                for rank in ranks[1:]:
                    common.intersection_update(rank[2])
                self._start_read(rid, op, common)

    def _load_host_anchor(self, op, request):
        match = self.cache.match_prefix(MatchPrefixParams(key=op.key[: op.prefix_len]))
        needed = match.host_hit_length > 0 or match.swa_host_hit_length > 0
        needs = self.gather(needed)
        if any(needs) and not all(needs):
            raise RuntimeError("Mooncake TP Host anchor residency diverged")
        if needed:
            success = self.cache.load_back(op.anchor, req=request)
            producer = self.controller.start_loading()
            if producer >= 0:
                self.controller.layer_done_counter.events[
                    producer
                ].finish_event.synchronize()
            if not all(self.gather(success)):
                raise RuntimeError(
                    "Mooncake TP Host anchor could not be restored consistently"
                )
            start = min(
                len(match.device_indices), op.prefix_len - match.swa_host_hit_length
            )
            self.sources.setdefault(op.req.rid, []).append(
                (max(0, start), op.prefix_len, 4)
            )
            op.req.kv.swa_evicted_seqlen = max(
                op.req.kv.swa_evicted_seqlen, request.kv.swa_evicted_seqlen
            )
        new_lock = self.cache.inc_lock_ref(op.anchor).to_dec_params()
        self.cache.dec_lock_ref(op.anchor, op.device_lock)
        op.device_lock = new_lock
        root = self.cache.root_node_handle(extra_key=op.key.extra_key)
        prefix = self.cache.tree_core.collect_full_device_indices(op.anchor, root)
        if len(prefix) != op.prefix_len:
            raise RuntimeError(
                "Mooncake Host/GPU prefix no longer matches its pinned anchor"
            )
        return prefix

    def _start_read(self, rid, op, boundaries):
        page = self.cache.page_size
        host_needed = max(0, op.prefix_len - len(op.req.prefix_indices))
        budget = min(
            self.controller.prefetch_capacity_limit,
            self.cache._component_available_size(ComponentType.FULL)
            + self.cache.full_evictable_size()
            - host_needed
            - 2 * page,
        )
        budget = min(self.gather(max(0, budget)))
        candidates = [
            n for n in boundaries if self.cache.prefetch_threshold <= n * page <= budget
        ]
        if not candidates:
            self._discard(rid, op)
            return
        pages = max(candidates)
        if self.controller.load_fence_stream is not None:
            # Private FULL/SWA mappings can reuse slots still referenced by an overlapping forward.
            torch.cuda.current_stream().wait_stream(self.controller.load_fence_stream)
        request = _LoadRequest(
            rid=rid,
            prefix_indices=op.req.prefix_indices,
            last_node=op.anchor,
            kv=copy.copy(op.req.kv),
            priority=op.req.priority or 0,
        )
        prefix = self._load_host_anchor(op, request)
        hit = ExternalCacheHitMarker(
            prefix_key=op.key[: op.prefix_len + pages * page],
            tail_hashes=op.hashes[:pages],
            device_hit_len=op.prefix_len,
        )
        plan, error = None, None
        try:
            op.prepared = self.loads.prepare_load(
                request, hit=hit, prefix_indices=prefix, anchor=op.anchor
            )
            if op.prepared is not None:
                transfers = [
                    transfer for _, transfer in op.prepared.component_transfers
                ]
                plan = self.payload.prepare_read(transfers)
        except Exception as exc:
            error = str(exc)
        states = self.gather((plan is not None, error))
        if not all(state[0] for state in states):
            if any(state[1] for state in states):
                logger.warning("Mooncake GPU prepare failed: %s", states)
            self._discard(rid, op)
            return
        op.full_tokens = pages * page
        op.swa_tokens = sum(
            len(transfer.device_indices)
            for _, transfer in op.prepared.component_transfers
            if transfer.name == PoolName.SWA
        )
        self.controller.prefetch_tokens_occupied += op.full_tokens
        start = torch.cuda.Event()
        if self.controller.load_fence_stream is not None:
            torch.cuda.current_stream().wait_stream(self.controller.load_fence_stream)
        start.record()
        op.result = GPUReadResult()
        op.io = self.io_worker.submit(self._read, plan, op.result, start)

    def _read(self, plan, result, start):
        torch.cuda.set_device(self.payload.device)
        stream = torch.cuda.Stream(device=self.payload.device)
        try:
            with torch.cuda.stream(stream):
                stream.wait_event(start)
                self.payload.read(plan, result)
        except Exception as exc:
            result.error = str(exc)
            result.success = False
            logger.exception("Mooncake GPU restore failed")
        finally:
            stream.synchronize()
            result.safe_to_release = True

    def _commit(self, rid, op, media):
        # A newly restored side component must not replace an existing D2H ACK for the same node.
        self.controller.l2_transfer_engine.device_to_host_stream.synchronize()
        self.cache.writing_check(finish_count=len(self.controller.ack_write_queue))
        _, node_id = self.loads.commit_prepared_load(op.prepared, queue_io=False)
        # Only the existing Host backup owner may publish and persist L2 copies.
        node = self.cache.resolve_node_handle(node_id)
        path = []
        while node.parent is not None:
            path.append(node.id)
            node = node.parent
        self.cache._execute_and_commit_kv_backup(BackupKV(list(reversed(path))))
        page = self.cache.page_size
        adopted_media = {}
        for component, pool in (
            (ComponentType.FULL, PoolName.KV),
            (ComponentType.SWA, PoolName.SWA),
        ):
            for start, end in op.prepared.adopted_ranges.get(component, []):
                for pos in range(max(start, op.prefix_len), end, page):
                    mask = media[pool.value][(pos - op.prefix_len) // page]
                    if mask not in (1, 2, 3):
                        mask = 255
                    adopted_media[pos] = adopted_media.get(pos, 0) | mask
        loaded = len(adopted_media) * page
        self.sources.setdefault(rid, []).extend(
            (pos, pos + page, mask) for pos, mask in sorted(adopted_media.items())
        )
        op.req.kv.swa_evicted_seqlen = max(
            op.req.kv.swa_evicted_seqlen, op.prepared.req.kv.swa_evicted_seqlen
        )
        self.loaded[rid] = self.loaded.get(rid, 0) + loaded
        latency = (time.monotonic() - op.started) * 1000
        self.latencies[rid] = (latency, loaded)
        if self.cache.enable_storage_metrics:
            self.cache.storage_metrics_collector.log_prefetched_tokens(loaded)
            self.cache.storage_metrics_collector.log_prefetch_latency_ms(latency)
        logger.info(
            "MOONCAKE_GDS_PREFETCH req=%s completed_tokens=%d inserted_tokens=%d "
            "ssd_tokens=%d dram_tokens=%d latency_ms=%.3f",
            rid,
            op.full_tokens,
            loaded,
            sum(mask in (2, 3) for mask in adopted_media.values()) * page,
            sum(mask == 1 for mask in adopted_media.values()) * page,
            latency,
        )
        self._release(rid, op)

    def _discard(self, rid, op):
        if op.prepared is not None:
            if op.io is not None and (
                not op.io.done() or not op.result.safe_to_release
            ):
                raise RuntimeError(
                    "Cannot release private pages while Mooncake GPU I/O is outstanding"
                )
            self.loads.abort_prepared_load(op.prepared)
        if not op.cancelled:
            self.latencies[rid] = ((time.monotonic() - op.started) * 1000, 0)
        self._release(rid, op)

    def _release(self, rid, op):
        self.controller.prefetch_tokens_occupied -= op.full_tokens
        self.cache.dec_lock_ref(op.anchor, op.device_lock)
        self.cache.dec_host_lock_ref(op.anchor, op.host_lock)
        del self.pending[rid]

    def cancel(self, rid):
        op = self.pending.get(rid)
        if op is not None:
            op.cancelled = True
        self.sources.pop(rid, None)
        self.loaded.pop(rid, None)
        self.latencies.pop(rid, None)

    def drain(self, *, local=False):
        for op in self.pending.values():
            op.cancelled = True
        deadline = time.monotonic() + 120.0
        error = None
        try:
            for op in self.pending.values():
                for future in (op.query, op.io):
                    if future is not None:
                        try:
                            future.result(timeout=max(0.0, deadline - time.monotonic()))
                        except Exception:
                            if not future.done():
                                raise
                if op.io is not None and not op.result.safe_to_release:
                    raise RuntimeError("GPU I/O completion is unknown")
        except Exception as exc:
            error = str(exc)
        errors = [error] if local else self.gather(error)
        if any(errors):
            raise RuntimeError(
                f"Mooncake GDS drain failed; resources retained: {errors}"
            )
        for rid, op in list(self.pending.items()):
            self._discard(rid, op)
        self.sources.clear()
        self.loaded.clear()
        self.latencies.clear()

    def close(self):
        if self.closed:
            return
        self.drain(local=True)
        self.query_worker.close()
        self.io_worker.close()
        self.closed = True
