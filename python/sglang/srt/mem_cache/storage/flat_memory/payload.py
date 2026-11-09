"""Versioned, byte-preserving Flat GPU objects and bounded aligned staging."""

from __future__ import annotations

import hashlib
import json
import re
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
    # FLAT_MEMORY: Per probed page, whether any fragment's GVA is SSD (>= 2^60).
    ssd_pages: tuple[bool, ...] = ()


# FLAT_MEMORY: Must match MediaHint in flat_memory_manager.h.
MEDIA_HINT_ANY, MEDIA_HINT_DRAM, MEDIA_HINT_SSD = 0, 1, 2
# Must match page_states in flat_memory_pybind.cpp.
PAGE_FOUND, PAGE_ON_DRAM, PAGE_ON_SSD, PAGE_ON_OTHER = 1, 2, 4, 8


class PlanSegments(msgspec.Struct, frozen=True):
    """Layer-major fragment order as segments: one (layer, pool, spec) each,
    covering every page of its pool. Mirrors PayloadLayout._layer_major."""

    prefixes: list[str]
    bases: list[int]
    row_bytes: list[int]
    lengths: list[int]
    padded: list[int]
    pools: list[int]
    layers: list[int]
    page_keys: list[list[str]]
    rows: list[list[int]]

    def counts(self) -> list[int]:
        return [len(self.rows[pool]) for pool in self.pools]


class PlanBatch(msgspec.Struct, frozen=True):
    """Fragments [begin, end) of a native FragmentPlan."""

    plan: Any
    begin: int
    end: int
    padded_bytes: int
    logical_bytes: int
    # Layer of the first fragment after this batch, or None at the end.
    next_layer: int | None
    last_layer: int

    def __len__(self) -> int:
        return self.end - self.begin


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
    # The fragment's bytes as a uint8 tensor, or, when row >= 0, the pool
    # buffer holding them at `pointer`. The latter skips three torch view ops
    # per fragment (~9.5 us; 1.4 s of Python per 150K-fragment restore).
    source: torch.Tensor
    padded_length: int
    # FLAT_MEMORY: Model layer id the KV pools wait on (wait_until(layer_id)).
    layer: int = 0
    pointer: int = 0
    length: int = 0
    row: int = -1
    row_span: int = 0
    offset: int = 0

    def nbytes(self) -> int:
        return self.length if self.row >= 0 else self.source.numel()

    def address(self) -> int:
        return self.pointer if self.row >= 0 else self.source.data_ptr()

    def view(self) -> torch.Tensor:
        if self.row < 0:
            return self.source
        page = self.source.narrow(0, self.row, self.row_span)
        return page.view(torch.uint8).reshape(-1).narrow(0, self.offset, self.length)


class PayloadLayout:
    def __init__(
        self,
        *,
        pool_group: DevicePoolGroup,
        model_namespace: str,
        tp_rank: int,
        tp_size: int,
        max_io_bytes: int,
        rank_replicated: bool = False,
    ):
        if not model_namespace:
            raise ValueError("Flat GPU storage requires a model/config namespace")
        if tp_size not in (1, 4, 8):
            raise ValueError("Flat GPU storage supports TP1, TP4 and TP8")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"Flat GPU tp_rank {tp_rank} is outside TP{tp_size}")
        if max_io_bytes < ALIGNMENT or max_io_bytes % ALIGNMENT:
            raise ValueError("gds_max_io_bytes must be a positive multiple of 4096")
        self.pool_group = pool_group
        self.max_io_bytes = max_io_bytes
        self.specs: dict[PoolName, tuple[FragmentSpec, ...]] = {}
        # FLAT_MEMORY: FC-CIO layer-major order: model layer -> spec indices per pool.
        self._specs_by_layer: dict[PoolName, dict[int, tuple[int, ...]]] = {}
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
            # A buffer no layer maps to is grouped with layer 0, so it lands first.
            buffer_layer = {}
            for layer, buffer_index in entry.layer_mapping.items():
                buffer_layer[buffer_index] = min(
                    layer, buffer_layer.get(buffer_index, layer)
                )
            by_layer: dict[int, list[int]] = {}
            for index, spec in enumerate(specs):
                by_layer.setdefault(buffer_layer.get(spec.buffer, 0), []).append(index)
            self._specs_by_layer[entry.name] = {
                layer: tuple(indices) for layer, indices in sorted(by_layer.items())
            }
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
            "tp_size": tp_size,
            "page_size": pool_group.page_size,
            "fragment_bytes": max_io_bytes,
            "pools": identity,
        }
        # FLAT_MEMORY: rank_replicated pools hold byte-identical KV on every
        # rank, so the namespace drops tp_rank and all ranks address one copy.
        # Head-sharded pools must keep it: a shared namespace would make eight
        # different shards collide on the same key. tp_size stays either way so
        # a TP4 and a TP8 run never share keys.
        if not rank_replicated:
            descriptor["tp_rank"] = tp_rank
        digest = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.namespace = f"flat-v{FORMAT_VERSION}:{digest}"
        # FLAT_MEMORY: The inverse of _key(); must change with it.
        self._page_key_pattern = re.compile(
            re.escape(self.namespace) + r":[^:]+:c\d+:b\d+:f\d+:n\d+:(.+)\Z"
        )

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

    def page_key_of(self, storage_key: str) -> str | None:
        match = self._page_key_pattern.match(storage_key)
        return match.group(1) if match else None

    def _key(self, *, name: PoolName, page_key: str, spec: FragmentSpec) -> str:
        return self._key_prefix(name=name, spec=spec) + page_key

    def _key_prefix(self, *, name: PoolName, spec: FragmentSpec) -> str:
        return (
            f"{self.namespace}:{name}:c{spec.component}:b{spec.buffer}:"
            f"f{spec.part}:n{spec.length}:"
        )

    def padded_bytes(self, transfers: list[PoolTransfer]) -> int:
        return sum(
            len(transfer.keys or [])
            * sum(spec.padded_length for spec in self.specs[transfer.name])
            for transfer in transfers
        )

    def fragments(self, transfers: list[PoolTransfer]) -> Iterator[PayloadFragment]:
        # FLAT_MEMORY: FC-CIO layer-major order, ascending model layer across pools,
        # so a read can release layer l to the forward pass before l + 1 lands.
        yield from self._layer_major(self._prepare(transfers))

    def write_groups(
        self, transfers: list[PoolTransfer], *, max_bytes: int, max_fragments: int
    ) -> Iterator[list[PayloadFragment]]:
        # FLAT_MEMORY: Each write batch becomes one extent. A batch holds whole
        # pages across every pool, layer-major inside, so a failed later batch or
        # a reclaimed extent costs only its own pages: a restore needs each pool's
        # share of a page, and splitting pools across extents voids the page too.
        prepared = self._prepare(transfers)
        rows = [dict(pages) for _, _, pages in prepared]
        for keys in self._page_groups(
            prepared, rows, max_bytes=max_bytes, max_count=max_fragments, count_pages=False
        ):
            yield self._group_fragments(prepared, rows, keys)

    def _page_groups(
        self, prepared, rows, *, max_bytes: int, max_count: int, count_pages: bool
    ) -> Iterator[list[str]]:
        # The pool covering the most pages (the full KV prefix) fixes page order;
        # pools holding a subset, such as SWA's trailing window, align by key.
        reference = max(prepared, key=lambda item: len(item[2]), default=None)
        order: dict[str, None] = {}
        for _, _, pages in ([reference] if reference else []) + prepared:
            for key, _ in pages:
                order.setdefault(key)
        page_bytes = {
            name: sum(spec.padded_length for spec in self.specs[name])
            for name, _, _ in prepared
        }
        group, group_bytes, group_fragments = [], 0, 0
        for key in order:
            owners = [i for i, pool_rows in enumerate(rows) if key in pool_rows]
            size = sum(page_bytes[prepared[i][0]] for i in owners)
            fragments = (
                1
                if count_pages
                else sum(len(self.specs[prepared[i][0]]) for i in owners)
            )
            if group and (
                group_bytes + size > max_bytes
                or group_fragments + fragments > max_count
            ):
                yield group
                group, group_bytes, group_fragments = [], 0, 0
            group.append(key)
            group_bytes += size
            group_fragments += fragments
        if group:
            yield group

    def _group_fragments(self, prepared, rows, keys) -> list[PayloadFragment]:
        return list(self._layer_major(self._group_subset(prepared, rows, keys)))

    @staticmethod
    def _group_subset(prepared, rows, keys) -> list:
        subset = [
            (name, entry, [(key, pool_rows[key]) for key in keys if key in pool_rows])
            for (name, entry, _), pool_rows in zip(prepared, rows)
        ]
        return [item for item in subset if item[2]]

    def plan_segments(self, transfers: list[PoolTransfer]) -> PlanSegments:
        return self._plan_segments(self._prepare(transfers))

    def write_plan_groups(
        self, transfers: list[PoolTransfer], *, max_bytes: int, max_pages: int
    ) -> Iterator[PlanSegments]:
        # FLAT_MEMORY: write_groups with the cap counted in pages, not fragments:
        # a V4 page alone is ~190 fragments, so a fragment cap made every
        # extent one page and scattered each layer across all of them.
        prepared = self._prepare(transfers)
        rows = [dict(pages) for _, _, pages in prepared]
        for keys in self._page_groups(
            prepared, rows, max_bytes=max_bytes, max_count=max_pages, count_pages=True
        ):
            yield self._plan_segments(self._group_subset(prepared, rows, keys))

    def _plan_segments(self, prepared: list) -> PlanSegments:
        fields: dict[str, list] = {
            name: []
            for name in ("prefixes", "bases", "row_bytes", "lengths", "padded", "pools", "layers")
        }
        layers = sorted(
            {layer for name, _, _ in prepared for layer in self._specs_by_layer[name]}
        )
        for layer in layers:
            for pool, (name, entry, pages) in enumerate(prepared):
                if not pages:
                    continue
                specs = self.specs[name]
                for index in self._specs_by_layer[name].get(layer, ()):
                    spec = specs[index]
                    buffer = entry.components[spec.component][spec.buffer]
                    if not buffer[: entry._row_span].is_contiguous():
                        raise ValueError(
                            f"Flat GPU pool {entry.name} moved or changed layout"
                        )
                    fields["prefixes"].append(self._key_prefix(name=name, spec=spec))
                    fields["bases"].append(buffer.data_ptr() + spec.offset)
                    fields["row_bytes"].append(buffer.stride(0) * buffer.element_size())
                    fields["lengths"].append(spec.length)
                    fields["padded"].append(spec.padded_length)
                    fields["pools"].append(pool)
                    fields["layers"].append(layer)
        return PlanSegments(
            **fields,
            page_keys=[[key for key, _ in pages] for _, _, pages in prepared],
            rows=[[int(row) for _, row in pages] for _, _, pages in prepared],
        )

    def _prepare(self, transfers: list[PoolTransfer]) -> list:
        prepared = []
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
            prepared.append((transfer.name, entry, list(zip(keys, locations))))
        return prepared

    def _layer_major(self, prepared: list) -> Iterator[PayloadFragment]:
        layers = sorted(
            {layer for name, _, _ in prepared for layer in self._specs_by_layer[name]}
        )
        for layer in layers:
            for name, entry, pages in prepared:
                specs = self.specs[name]
                span = entry._row_span
                for index in self._specs_by_layer[name].get(layer, ()):
                    spec = specs[index]
                    buffer = entry.components[spec.component][spec.buffer]
                    # Equal strides on every page, so one check covers each
                    # page the per-fragment narrow used to check.
                    if not buffer[:span].is_contiguous():
                        raise ValueError(
                            f"Flat GPU pool {entry.name} moved or changed layout"
                        )
                    base = buffer.data_ptr() + spec.offset
                    row_bytes = buffer.stride(0) * buffer.element_size()
                    prefix = self._key_prefix(name=name, spec=spec)
                    for page_key, row in pages:
                        yield PayloadFragment(
                            prefix + page_key,
                            buffer,
                            spec.padded_length,
                            layer,
                            pointer=base + row * row_bytes,
                            length=spec.length,
                            row=row,
                            row_span=span,
                            offset=spec.offset,
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
        native_page_copies: bool = False,
    ):
        self.layout = layout
        self.manager = manager
        self.device = torch.device(device)
        # FLAT_MEMORY: FC-CIO. Fragments are enumerated natively from a
        # FragmentPlan; reads land straight in the KV page slices and writes
        # pack with one gather launch, instead of a Python object and torch
        # copy_ per ~10 KiB fragment (~150K per V4 restore). Needs the native
        # manager's *_plan methods; fakes and CPU devices keep the old path.
        self.native_page_copies = native_page_copies
        self._plan_mode = native_page_copies and self.device.type == "cuda"
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
        # FLAT_MEMORY: FC-MAS. DRAM restores get a separate one-batch budget,
        # so they never queue behind SSD read or write staging.
        self.dram_read_budget = StagingBudget(self.read_quota_bytes)
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
        media_hint: int = MEDIA_HINT_ANY,
    ) -> None:
        self.raise_if_unsafe()
        with self._context(), self._stream_context():
            self._write_transfers(
                transfers, on_source_consumed=on_source_consumed, media_hint=media_hint
            )

    def read(
        self,
        transfers: list[PoolTransfer],
        *,
        medium: str = "ssd",
        on_progress: Callable[[int, int, int], None] | None = None,
    ) -> None:
        # FLAT_MEMORY: on_progress(done_bytes, total_bytes, layers_done_through)
        # runs after each batch, once its bytes are in place in the device pages.
        self.raise_if_unsafe()
        if self._plan_mode:
            self._read_plan(transfers, on_progress=on_progress)
            return
        budget = self.dram_read_budget if medium == "dram" else self.budget
        fragments = list(self.layout.fragments(transfers))
        total_bytes = sum(fragment.padded_length for fragment in fragments)
        done_bytes = consumed = 0
        with self._context(), self._stream_context():
            for batch, total in self._batches(
                fragments, batch_bytes=self.read_batch_bytes
            ):
                self._with_stage(
                    total,
                    lambda stage: self._transfer_batch(
                        batch=batch, stage=stage, write=False
                    ),
                    budget=budget,
                )
                self._record_batch(batch, write=False)
                done_bytes += total
                consumed += len(batch)
                if on_progress is not None:
                    # Fragments ascend by layer, so everything below the next
                    # pending fragment's layer is complete.
                    through = (
                        fragments[consumed].layer - 1
                        if consumed < len(fragments)
                        else fragments[-1].layer
                    )
                    on_progress(done_bytes, total_bytes, through)

    def _read_plan(self, transfers, *, on_progress) -> None:
        segments = self.layout.plan_segments(transfers)
        plan = self._make_plan(segments)
        total_bytes = sum(n * p for n, p in zip(segments.counts(), segments.padded))
        done_bytes = 0
        with self._context(), self._stream_context():
            for batch in self._plan_batches(
                plan, segments, batch_bytes=self.read_batch_bytes
            ):
                # Native streams must wait for earlier users of the destination
                # pages; the pages are private until commit, so a failed read
                # leaves nothing visible.
                self._sync()
                self.manager.read_plan(plan, batch.begin, batch.end)
                self._record_batch(batch, write=False)
                done_bytes += batch.padded_bytes
                if on_progress is not None:
                    through = (
                        batch.next_layer - 1
                        if batch.next_layer is not None
                        else batch.last_layer
                    )
                    on_progress(done_bytes, total_bytes, through)

    def _make_plan(self, segments: PlanSegments):
        return self.manager.make_plan(
            prefixes=segments.prefixes,
            bases=segments.bases,
            row_bytes=segments.row_bytes,
            lengths=segments.lengths,
            padded=segments.padded,
            pools=segments.pools,
            page_keys=segments.page_keys,
            rows=segments.rows,
        )

    @staticmethod
    def _plan_batches(plan, segments: PlanSegments, *, batch_bytes: int):
        # Splits by padded bytes alone; a fragment larger than the budget still
        # gets a batch of its own.
        begin = at = padded = logical = 0
        last_layer = 0
        for count, size, length, layer in zip(
            segments.counts(), segments.padded, segments.lengths, segments.layers
        ):
            left = count
            while left:
                take = min(left, max(0, batch_bytes - padded) // size)
                if not take and at == begin:
                    take = 1
                if not take:
                    yield PlanBatch(plan, begin, at, padded, logical, layer, last_layer)
                    begin, padded, logical = at, 0, 0
                    continue
                left -= take
                at += take
                padded += take * size
                logical += take * length
                last_layer = layer
        if at > begin:
            yield PlanBatch(plan, begin, at, padded, logical, None, last_layer)

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
        self.dram_read_budget.poison(error)

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

    def _write_transfers(self, transfers, *, on_source_consumed, media_hint) -> None:
        whole_bytes = self.layout.padded_bytes(transfers)
        remaining = whole_bytes
        # A group that fits is exactly one batch; only an oversized page splits.
        if self._plan_mode:
            batches = iter(
                [
                    (batch, batch.padded_bytes)
                    for group in self.layout.write_plan_groups(
                        transfers,
                        max_bytes=self.write_batch_bytes,
                        max_pages=self.batch_pages,
                    )
                    for batch in self._plan_batches(
                        self._make_plan(group), group, batch_bytes=self.write_batch_bytes
                    )
                ]
            )
        else:
            batches = iter(
                [
                    batch
                    for group in self.layout.write_groups(
                        transfers,
                        max_bytes=self.write_batch_bytes,
                        max_fragments=self.batch_pages,
                    )
                    for batch in self._batches(group, batch_bytes=self.write_batch_bytes)
                ]
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
                            media_hint=media_hint,
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
                    media_hint=media_hint,
                ),
            )
            self._record_batch(batch, write=True)
            if remaining == 0:
                return
        # FLAT_MEMORY: Only duplicate batches (possibly a trailing suffix) remain.
        if on_source_consumed is not None:
            on_source_consumed()

    def _duplicate_batch(self, batch) -> bool:
        if isinstance(batch, PlanBatch):
            admission = self.manager.check_write_plan(batch.plan, batch.begin, batch.end)
            return all(self._write_results(admission, count=len(batch), complete=False))
        admission = self.manager.check_gpu_write(
            [fragment.key for fragment in batch],
            [fragment.padded_length for fragment in batch],
        )
        return all(self._write_results(admission, count=len(batch), complete=False))

    def _write_snapshot(
        self, *, first, batches, stage, on_source_consumed, media_hint
    ) -> None:
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
                self._write_staged_batch(
                    batch, stage.narrow(0, offset, total), media_hint=media_hint
                )
            self._record_batch(batch, write=True)
            offset += total

    def _with_stage(
        self,
        total: int,
        transfer,
        *,
        reserved: bool = False,
        budget: StagingBudget | None = None,
    ) -> None:
        budget = self.budget if budget is None else budget
        allocated = total + ALIGNMENT - 1
        if not reserved:
            budget.acquire(allocated)
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
                budget.release(allocated)

    def _pack_batch(self, batch, stage) -> None:
        if isinstance(batch, PlanBatch):
            stage.zero_()
            stream = torch.cuda.current_stream(self.device).cuda_stream
            self.manager.pack_plan(
                batch.plan, batch.begin, batch.end, stage.data_ptr(), stream
            )
            return
        offset = 0
        for fragment in batch:
            view = stage.narrow(0, offset, fragment.padded_length)
            view.zero_()
            source = fragment.view()
            view[: source.numel()].copy_(source)
            offset += fragment.padded_length

    def _write_staged_batch(self, batch, stage, *, media_hint) -> None:
        if isinstance(batch, PlanBatch):
            result = self.manager.put_plan(
                batch.plan, batch.begin, batch.end, stage.data_ptr(), media_hint
            )
            self._write_results(result, count=len(batch), complete=True)
            return
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
            **({"media_hint": media_hint} if media_hint != MEDIA_HINT_ANY else {}),
        )
        self._write_results(result, count=len(batch), complete=True)

    def _transfer_batch(
        self,
        *,
        batch,
        stage: torch.Tensor,
        write: bool,
        on_source_consumed=None,
        media_hint: int = MEDIA_HINT_ANY,
    ) -> None:
        if write:
            self._pack_batch(batch, stage)
            self._sync()
            if on_source_consumed is not None:
                on_source_consumed()
            self._write_staged_batch(batch, stage, media_hint=media_hint)
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
            source = fragment.view()
            source.copy_(view[: source.numel()])
        self._sync()

    def _record_batch(self, batch, *, write: bool) -> None:
        if isinstance(batch, PlanBatch):
            logical, padded = batch.logical_bytes, batch.padded_bytes
        else:
            logical = sum(fragment.nbytes() for fragment in batch)
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
        ssd_pages = [False] * len(keys)
        for transfer in transfers:
            states = self.page_states(name=transfer.name, keys=keys)
            exists = [bool(state & PAGE_FOUND) for state in states]
            for index, state in enumerate(states):
                ssd_pages[index] |= bool(state & PAGE_ON_SSD)
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
        return LookupProbeResult(
            tuple(sorted(candidates)), frozenset(missing), tuple(ssd_pages)
        )

    def page_states(self, *, name: PoolName, keys: list[str]) -> list[int]:
        # FLAT_MEMORY: per page, PAGE_FOUND when every fragment exists plus one
        # PAGE_ON_* bit per medium holding a fragment.
        if self._plan_mode:
            prefixes = [
                self.layout._key_prefix(name=name, spec=spec)
                for spec in self.layout.specs[name]
            ]
            return self.manager.page_states(prefixes, keys)
        states = []
        for addresses in self.page_addresses(name=name, keys=keys):
            state = PAGE_FOUND if addresses and all(addresses) else 0
            for address in addresses:
                if address:
                    media = address >> 60
                    state |= (
                        PAGE_ON_DRAM if media == 0 else PAGE_ON_SSD if media == 1 else PAGE_ON_OTHER
                    )
            states.append(state)
        return states

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
            for state in self.page_states(name=transfer.name, keys=transfer.keys or []):
                if state & PAGE_ON_SSD:
                    return "ssd"
        return "dram"

    def get_stats(self) -> dict:
        with self._metrics_lock:
            return {
                "logical_read_bytes": self.logical_read_bytes,
                "logical_write_bytes": self.logical_write_bytes,
                "padded_read_bytes": self.padded_read_bytes,
                "padded_write_bytes": self.padded_write_bytes,
                "gpu_staging_bytes": self.budget.current_bytes
                + self.dram_read_budget.current_bytes,
                "gpu_dram_read_staging_bytes": self.dram_read_budget.current_bytes,
                "gpu_dram_read_staging_peak_bytes": self.dram_read_budget.peak_bytes,
                "gpu_dram_read_staging_wait_count": self.dram_read_budget.wait_count,
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
