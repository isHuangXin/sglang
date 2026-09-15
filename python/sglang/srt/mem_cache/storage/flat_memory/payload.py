"""Versioned, byte-preserving Flat GPU objects and bounded aligned staging."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
from typing import Any

import msgspec
import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolGroup
from sglang.srt.mem_cache.storage.flat_memory.io_result import (
    FlatCapacityError,
    FlatUnsafeIOError,
)

ALIGNMENT = 4096
FORMAT_VERSION = 1
PageRecord = tuple[PoolName, str]


class LookupProbeResult(msgspec.Struct, frozen=True):
    boundaries: tuple[int, ...]
    missing_records: frozenset[PageRecord] = frozenset()


def align_up(size: int) -> int:
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def restorable_boundaries(
    page_exists: list[bool], *, policy: PoolHitPolicy, trailing_pages: int
) -> set[int]:
    if policy == PoolHitPolicy.ALL_PAGES:
        first_missing = next(
            (index for index, exists in enumerate(page_exists) if not exists),
            len(page_exists),
        )
        return set(range(1, first_missing + 1))
    if policy != PoolHitPolicy.TRAILING_PAGES:
        raise ValueError(f"Unsupported Flat pool hit policy: {policy}")
    if trailing_pages < 1:
        raise ValueError("A trailing pool must request at least one page")
    return {
        stop
        for stop in range(1, len(page_exists) + 1)
        if all(page_exists[max(0, stop - trailing_pages) : stop])
    }


class FragmentSpec(msgspec.Struct, frozen=True):
    component: int
    buffer: int
    offset: int
    length: int
    padded_length: int
    part: int


class PayloadFragment(msgspec.Struct, frozen=True):
    key: str
    source: torch.Tensor
    padded_length: int


class PayloadLayout:
    def __init__(
        self,
        *,
        pool_group: DevicePoolGroup,
        model_namespace: str,
        tp_rank: int,
        tp_size: int,
        max_io_bytes: int,
    ):
        if not model_namespace:
            raise ValueError("Flat GPU storage requires a model/config namespace")
        if tp_size not in (1, 4, 8) or not 0 <= tp_rank < tp_size:
            raise ValueError("Flat GPU storage supports TP1, TP4 and TP8")
        if max_io_bytes < ALIGNMENT or max_io_bytes % ALIGNMENT:
            raise ValueError("gds_max_io_bytes must be a positive multiple of 4096")
        self.pool_group = pool_group
        self.max_io_bytes = max_io_bytes
        self.specs: dict[PoolName, tuple[FragmentSpec, ...]] = {}
        identity = []
        for entry in pool_group.entries:
            specs = []
            buffers = []
            # FLAT_MEMORY: Pool rows may be padded, but their physical bytes must be stable.
            for component, group in enumerate(entry.components):
                for buffer_index, buffer in enumerate(group):
                    rows = entry._row_span
                    if buffer.ndim < 2 or not buffer[:rows].is_contiguous():
                        raise ValueError(
                            f"Flat GPU pool {entry.name} has a noncontiguous page layout"
                        )
                    size = buffer[0].numel() * buffer.element_size() * rows
                    if not size:
                        raise ValueError(
                            f"Flat GPU pool {entry.name} has an empty page"
                        )
                    for part, offset in enumerate(range(0, size, max_io_bytes)):
                        length = min(max_io_bytes, size - offset)
                        specs.append(
                            FragmentSpec(
                                component,
                                buffer_index,
                                offset,
                                length,
                                align_up(length),
                                part,
                            )
                        )
                    buffers.append(
                        (
                            component,
                            buffer_index,
                            str(buffer.dtype),
                            tuple(buffer.shape[1:]),
                            tuple(buffer.stride()),
                            rows,
                        )
                    )
            self.specs[entry.name] = tuple(specs)
            identity.append(
                (
                    str(entry.name),
                    str(entry.indices_from_pool),
                    entry.page_size,
                    sorted(entry.layer_mapping.items()),
                    buffers,
                )
            )
        descriptor = {
            "version": FORMAT_VERSION,
            "model": model_namespace,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "page_size": pool_group.page_size,
            "fragment_bytes": max_io_bytes,
            "pools": identity,
        }
        digest = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.namespace = f"flat-v{FORMAT_VERSION}:{digest}"

    def page_record(self, *, name: PoolName, page_key: str) -> PageRecord:
        return name, f"{self.namespace}:{page_key}"

    def coverage(self, transfers: list[PoolTransfer]) -> frozenset[PageRecord]:
        # FLAT_MEMORY: Callers provide resolved physical pools, not logical aliases.
        return frozenset(
            self.page_record(name=transfer.name, page_key=key)
            for transfer in transfers
            for key in transfer.keys or []
        )

    def keys_for_page(self, *, name: PoolName, page_key: str) -> list[str]:
        return [
            self._key(name=name, page_key=page_key, spec=spec)
            for spec in self.specs[name]
        ]

    def _key(self, *, name: PoolName, page_key: str, spec: FragmentSpec) -> str:
        return (
            f"{self.namespace}:{name}:c{spec.component}:b{spec.buffer}:"
            f"f{spec.part}:n{spec.length}:{page_key}"
        )

    def padded_bytes(self, transfers: list[PoolTransfer]) -> int:
        return sum(
            len(transfer.keys or [])
            * sum(spec.padded_length for spec in self.specs[transfer.name])
            for transfer in transfers
        )

    def fragments(self, transfers: list[PoolTransfer]) -> Iterator[PayloadFragment]:
        for transfer in transfers:
            entry = self.pool_group.entry_map[transfer.name]
            if transfer.host_indices is None:
                raise ValueError(
                    f"Flat GPU transfer {transfer.name} has no device slots"
                )
            locations = entry.prepare_locations(transfer.host_indices)
            keys = transfer.keys or []
            if len(keys) != len(locations):
                raise ValueError(
                    f"Flat GPU transfer {transfer.name} keys/slots disagree"
                )
            for page_key, row in zip(keys, locations):
                for spec in self.specs[transfer.name]:
                    buffer = entry.components[spec.component][spec.buffer]
                    page = buffer.narrow(0, row, entry._row_span)
                    if not page.is_contiguous():
                        raise ValueError(
                            f"Flat GPU pool {entry.name} moved or changed layout"
                        )
                    raw = page.view(torch.uint8).reshape(-1)
                    yield PayloadFragment(
                        self._key(name=transfer.name, page_key=page_key, spec=spec),
                        raw.narrow(0, spec.offset, spec.length),
                        spec.padded_length,
                    )


class StagingBudget:
    def __init__(self, total_bytes: int):
        if total_bytes < 2 * ALIGNMENT:
            raise ValueError("Flat GPU staging budget must be at least 8192 bytes")
        self.total_bytes = total_bytes
        self.current_bytes = 0
        self.peak_bytes = 0
        self.wait_seconds = 0.0
        self.wait_count = 0
        self._condition = threading.Condition()
        self._error: FlatUnsafeIOError | None = None

    def acquire(self, size: int) -> None:
        if size > self.total_bytes:
            raise ValueError("One Flat staging allocation exceeds the total budget")
        with self._condition:
            if self._error is None and self.current_bytes + size > self.total_bytes:
                started = time.perf_counter()
                self.wait_count += 1
                self._condition.wait_for(
                    lambda: self._error is not None
                    or self.current_bytes + size <= self.total_bytes
                )
                self.wait_seconds += time.perf_counter() - started
            if self._error is not None:
                raise self._error
            self.current_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.current_bytes)

    def try_acquire(self, size: int) -> bool:
        with self._condition:
            if self._error is not None:
                raise self._error
            if self.current_bytes + size > self.total_bytes:
                return False
            self.current_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.current_bytes)
            return True

    def poison(self, error: FlatUnsafeIOError) -> None:
        with self._condition:
            self._error = error
            self._condition.notify_all()

    def release(self, size: int) -> None:
        with self._condition:
            self.current_bytes -= size
            self._condition.notify_all()


class GPUTransfer:
    def __init__(
        self,
        *,
        layout: PayloadLayout,
        manager: Any,
        device: torch.device,
        staging_bytes: int,
        batch_pages: int = 128,
    ):
        self.layout = layout
        self.manager = manager
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if any(
            buffer.device != self.device
            for entry in layout.pool_group.entries
            for component in entry.components
            for buffer in component
        ):
            raise ValueError(
                "Flat payload buffers must all belong to the configured device"
            )
        self.budget = StagingBudget(staging_bytes)
        self.batch_pages = batch_pages
        if batch_pages < 1:
            raise ValueError("Flat GPU batch size must be positive")
        # FLAT_MEMORY: Include alignment slack in the actual allocation budget.
        self.batch_bytes = (staging_bytes - ALIGNMENT + 1) // ALIGNMENT * ALIGNMENT
        if layout.max_io_bytes > self.batch_bytes:
            raise ValueError(
                "Flat staging budget cannot fit gds_max_io_bytes plus alignment"
            )
        # FLAT_MEMORY: One write worker owns at most W, leaving actual read headroom.
        write_quota = staging_bytes // 2
        read_quota = staging_bytes - write_quota
        largest_fragment = max(
            (spec.padded_length for specs in layout.specs.values() for spec in specs),
            default=0,
        )
        self.snapshot_enabled = largest_fragment + ALIGNMENT - 1 <= min(
            write_quota, read_quota
        )
        self.write_quota_bytes = write_quota if self.snapshot_enabled else staging_bytes
        self.read_quota_bytes = read_quota if self.snapshot_enabled else staging_bytes
        self.write_batch_bytes = (
            (self.write_quota_bytes - ALIGNMENT + 1) // ALIGNMENT * ALIGNMENT
        )
        self.read_batch_bytes = (
            (self.read_quota_bytes - ALIGNMENT + 1) // ALIGNMENT * ALIGNMENT
        )
        self.snapshot_writes = 0
        self.streaming_writes = 0
        self.snapshot_budget_fallbacks = 0
        self.logical_read_bytes = 0
        self.logical_write_bytes = 0
        self.padded_read_bytes = 0
        self.padded_write_bytes = 0
        self._metrics_lock = threading.Lock()
        self._unsafe_error: FlatUnsafeIOError | None = None
        self._retained_staging: list[torch.Tensor] = []

    def _batches(self, fragments: Iterable[PayloadFragment], *, batch_bytes: int):
        batch, total = [], 0
        for fragment in fragments:
            if batch and (
                total + fragment.padded_length > batch_bytes
                or len(batch) >= self.batch_pages
            ):
                yield batch, total
                batch, total = [], 0
            batch.append(fragment)
            total += fragment.padded_length
        if batch:
            yield batch, total

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _context(self):
        return (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )

    def _stream_context(self):
        if self.device.type == "cuda":
            return torch.cuda.stream(torch.cuda.Stream(device=self.device))
        return nullcontext()

    def write(
        self,
        transfers: list[PoolTransfer],
        *,
        on_source_consumed: Callable[[], None] | None = None,
    ) -> None:
        self.raise_if_unsafe()
        with self._context(), self._stream_context():
            self._write_transfers(transfers, on_source_consumed=on_source_consumed)

    def read(self, transfers: list[PoolTransfer]) -> None:
        self.raise_if_unsafe()
        with self._context(), self._stream_context():
            for batch, total in self._batches(
                self.layout.fragments(transfers), batch_bytes=self.read_batch_bytes
            ):
                self._with_stage(
                    total,
                    lambda stage: self._transfer_batch(
                        batch=batch, stage=stage, write=False
                    ),
                )
                self._record_batch(batch, write=False)

    def raise_if_unsafe(self) -> None:
        if self._unsafe_error is not None:
            raise self._unsafe_error

    def _retain_stage(self, owner, error: FlatUnsafeIOError) -> None:
        # FLAT_MEMORY: Unproven completion must not return buffers to the allocator.
        with self._metrics_lock:
            self._unsafe_error = error
            if owner is not None:
                self._retained_staging.append(owner)
        self.budget.poison(error)

    @staticmethod
    def _write_results(result: dict, *, count: int, complete: bool) -> list[bool]:
        results = result["results"]
        if len(results) != count:
            raise RuntimeError("Flat GPU write returned the wrong number of results")
        status = result["status"]
        if status == "capacity":
            raise FlatCapacityError(result["error"])
        if status != "ok":
            raise RuntimeError(f"Flat GPU write {status}: {result['error']}")
        if complete and not all(results):
            raise RuntimeError("Flat GPU write did not complete every payload fragment")
        return results

    def _write_transfers(self, transfers, *, on_source_consumed) -> None:
        whole_bytes = self.layout.padded_bytes(transfers)
        remaining = whole_bytes
        batches = iter(
            self._batches(
                self.layout.fragments(transfers), batch_bytes=self.write_batch_bytes
            )
        )
        attempted_snapshot = False
        for batch, total in batches:
            remaining -= total
            if self._duplicate_batch(batch):
                self._record_batch(batch, write=True)
                continue
            if not attempted_snapshot:
                attempted_snapshot = True
                allocated = whole_bytes + ALIGNMENT - 1
                fits = self.snapshot_enabled and allocated <= self.write_quota_bytes
                if fits and self.budget.try_acquire(allocated):
                    with self._metrics_lock:
                        self.snapshot_writes += 1
                    self._with_stage(
                        whole_bytes,
                        lambda stage: self._write_snapshot(
                            first=(batch, total),
                            batches=batches,
                            stage=stage,
                            on_source_consumed=on_source_consumed,
                        ),
                        reserved=True,
                    )
                    return
                with self._metrics_lock:
                    self.streaming_writes += 1
                    self.snapshot_budget_fallbacks += int(fits)
            self._with_stage(
                total,
                lambda stage: self._transfer_batch(
                    batch=batch,
                    stage=stage,
                    write=True,
                    on_source_consumed=(on_source_consumed if remaining == 0 else None),
                ),
            )
            self._record_batch(batch, write=True)
            if remaining == 0:
                return
        # FLAT_MEMORY: Only duplicate batches (possibly a trailing suffix) remain.
        if on_source_consumed is not None:
            on_source_consumed()

    def _duplicate_batch(self, batch) -> bool:
        admission = self.manager.check_gpu_write(
            [fragment.key for fragment in batch],
            [fragment.padded_length for fragment in batch],
        )
        return all(self._write_results(admission, count=len(batch), complete=False))

    def _write_snapshot(self, *, first, batches, stage, on_source_consumed) -> None:
        owned_batches = [first, *batches]
        offset = 0
        for batch, total in owned_batches:
            self._pack_batch(batch, stage.narrow(0, offset, total))
            offset += total
        self._sync()
        if on_source_consumed is not None:
            on_source_consumed()
        offset = 0
        for index, (batch, total) in enumerate(owned_batches):
            # FLAT_MEMORY: Admission remains per native batch, never one storage extent.
            if index == 0 or not self._duplicate_batch(batch):
                self._write_staged_batch(batch, stage.narrow(0, offset, total))
            self._record_batch(batch, write=True)
            offset += total

    def _with_stage(self, total: int, transfer, *, reserved: bool = False) -> None:
        allocated = total + ALIGNMENT - 1
        if not reserved:
            self.budget.acquire(allocated)
        owner = stage = None
        unsafe = False
        try:
            owner = torch.empty(allocated, dtype=torch.uint8, device=self.device)
            offset = (-owner.data_ptr()) % ALIGNMENT
            stage = owner.narrow(0, offset, total)
            transfer(stage)
        except FlatUnsafeIOError as error:
            unsafe = True
            self._retain_stage(owner, error)
            raise
        finally:
            if not unsafe:
                try:
                    self._sync()
                except Exception as error:
                    failure = FlatUnsafeIOError(
                        f"Flat GPU staging did not quiesce: {error}"
                    )
                    self._retain_stage(owner, failure)
                    raise failure from error
                owner = stage = None
                self.budget.release(allocated)

    @staticmethod
    def _pack_batch(batch, stage) -> None:
        offset = 0
        for fragment in batch:
            view = stage.narrow(0, offset, fragment.padded_length)
            view.zero_()
            view[: fragment.source.numel()].copy_(fragment.source)
            offset += fragment.padded_length

    def _write_staged_batch(self, batch, stage) -> None:
        pointers, offset = [], 0
        for fragment in batch:
            pointer = stage.data_ptr() + offset
            if pointer % ALIGNMENT:
                raise RuntimeError("Flat GPU staging pointer is not 4096-byte aligned")
            pointers.append(pointer)
            offset += fragment.padded_length
        result = self.manager.put_gpu_file_detailed(
            [fragment.key for fragment in batch],
            pointers,
            [fragment.padded_length for fragment in batch],
        )
        self._write_results(result, count=len(batch), complete=True)

    def _transfer_batch(
        self, *, batch, stage: torch.Tensor, write: bool, on_source_consumed=None
    ) -> None:
        if write:
            self._pack_batch(batch, stage)
            self._sync()
            if on_source_consumed is not None:
                on_source_consumed()
            self._write_staged_batch(batch, stage)
            return
        views, pointers, sizes, keys = [], [], [], []
        offset = 0
        for fragment in batch:
            view = stage.narrow(0, offset, fragment.padded_length)
            views.append(view)
            pointers.append(view.data_ptr())
            sizes.append(fragment.padded_length)
            keys.append(fragment.key)
            if view.data_ptr() % ALIGNMENT:
                raise RuntimeError("Flat GPU staging pointer is not 4096-byte aligned")
            offset += fragment.padded_length
        # FLAT_MEMORY: Native streams must wait for this allocation's prior users.
        self._sync()
        addresses = self.manager.lookup_addresses(keys)
        if len(addresses) != len(keys) or not all(addresses):
            raise RuntimeError("Flat GPU restore lost a required payload fragment")
        if any(address % ALIGNMENT for address in addresses):
            raise RuntimeError("Flat GPU storage offset is not 4096-byte aligned")
        results = self.manager.read_gpu(addresses, pointers, sizes)
        if len(results) != len(batch) or not all(results):
            raise RuntimeError("Flat GPU read did not complete every payload fragment")
        for fragment, view in zip(batch, views):
            fragment.source.copy_(view[: fragment.source.numel()])
        self._sync()

    def _record_batch(self, batch: list[PayloadFragment], *, write: bool) -> None:
        logical = sum(fragment.source.numel() for fragment in batch)
        padded = sum(fragment.padded_length for fragment in batch)
        with self._metrics_lock:
            if write:
                self.logical_write_bytes += logical
                self.padded_write_bytes += padded
            else:
                self.logical_read_bytes += logical
                self.padded_read_bytes += padded

    def lookup(self, *, keys: list[str], transfers: list[PoolTransfer]) -> list[int]:
        return list(self.lookup_probe(keys=keys, transfers=transfers).boundaries)

    def lookup_probe(
        self,
        *,
        keys: list[str],
        transfers: list[PoolTransfer],
        pending_coverage: frozenset[PageRecord] = frozenset(),
    ) -> LookupProbeResult:
        candidates = set(range(1, len(keys) + 1))
        possible = set(candidates)
        pools = []
        for transfer in transfers:
            states = self.page_addresses(name=transfer.name, keys=keys)
            exists = [bool(addresses) and all(addresses) for addresses in states]
            records = [
                self.layout.page_record(name=transfer.name, page_key=key)
                for key in keys
            ]
            trailing_pages = max(1, len(transfer.keys or []))
            candidates &= restorable_boundaries(
                exists, policy=transfer.hit_policy, trailing_pages=trailing_pages
            )
            # FLAT_MEMORY: Pending writes only explain a retry; never publish them as hits.
            possible &= restorable_boundaries(
                [
                    found or record in pending_coverage
                    for found, record in zip(exists, records)
                ],
                policy=transfer.hit_policy,
                trailing_pages=trailing_pages,
            )
            pools.append((transfer.hit_policy, trailing_pages, exists, records))
            if not possible:
                break

        best = max(candidates, default=0)
        better = sorted(stop for stop in possible if stop > best)
        missing = set()
        if better:
            for policy, trailing_pages, exists, records in pools:
                if policy == PoolHitPolicy.ALL_PAGES:
                    required = range(better[-1])
                else:
                    required = {
                        index
                        for stop in better
                        for index in range(max(0, stop - trailing_pages), stop)
                    }
                missing.update(
                    records[index] for index in required if not exists[index]
                )
        return LookupProbeResult(tuple(sorted(candidates)), frozenset(missing))

    def page_addresses(self, *, name: PoolName, keys: list[str]) -> list[list[int]]:
        states = []
        width = len(self.layout.specs[name])
        for start in range(0, len(keys), self.batch_pages):
            query = [
                key
                for page in keys[start : start + self.batch_pages]
                for key in self.layout.keys_for_page(name=name, page_key=page)
            ]
            addresses = self.manager.lookup_addresses(query)
            if len(addresses) != len(query):
                raise RuntimeError("Flat lookup returned the wrong number of addresses")
            states.extend(
                addresses[i : i + width] for i in range(0, len(addresses), width)
            )
        return states

    def media(self, transfers: list[PoolTransfer]) -> str:
        for transfer in transfers:
            for addresses in self.page_addresses(
                name=transfer.name, keys=transfer.keys or []
            ):
                if any(address and address >> 60 == 1 for address in addresses):
                    return "ssd"
        return "dram"

    def get_stats(self) -> dict:
        with self._metrics_lock:
            return {
                "logical_read_bytes": self.logical_read_bytes,
                "logical_write_bytes": self.logical_write_bytes,
                "padded_read_bytes": self.padded_read_bytes,
                "padded_write_bytes": self.padded_write_bytes,
                "gpu_staging_bytes": self.budget.current_bytes,
                "gpu_staging_peak_bytes": self.budget.peak_bytes,
                "gpu_staging_budget_bytes": self.budget.total_bytes,
                "gpu_staging_wait_seconds": self.budget.wait_seconds,
                "gpu_staging_wait_count": self.budget.wait_count,
                "snapshot_enabled": self.snapshot_enabled,
                "write_owner_quota_bytes": self.write_quota_bytes,
                "read_owner_quota_bytes": self.read_quota_bytes,
                "snapshot_writes": self.snapshot_writes,
                "streaming_writes": self.streaming_writes,
                "snapshot_budget_fallbacks": self.snapshot_budget_fallbacks,
            }
