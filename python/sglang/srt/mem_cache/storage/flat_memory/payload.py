"""Versioned, byte-preserving Flat GPU objects and bounded aligned staging."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from typing import Any

import msgspec
import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolGroup

ALIGNMENT = 4096
FORMAT_VERSION = 1


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
        self._condition = threading.Condition()

    def acquire(self, size: int) -> None:
        if size > self.total_bytes:
            raise ValueError("One Flat staging allocation exceeds the total budget")
        with self._condition:
            self._condition.wait_for(
                lambda: self.current_bytes + size <= self.total_bytes
            )
            self.current_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.current_bytes)

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
        self.logical_read_bytes = 0
        self.logical_write_bytes = 0
        self.padded_read_bytes = 0
        self.padded_write_bytes = 0
        self._metrics_lock = threading.Lock()

    def _batches(self, fragments: Iterable[PayloadFragment]):
        batch, total = [], 0
        for fragment in fragments:
            if batch and (
                total + fragment.padded_length > self.batch_bytes
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

    def write(self, transfers: list[PoolTransfer]) -> None:
        self._transfer(transfers=transfers, write=True)

    def read(self, transfers: list[PoolTransfer]) -> None:
        self._transfer(transfers=transfers, write=False)

    def _transfer(self, *, transfers: list[PoolTransfer], write: bool) -> None:
        with self._context(), self._stream_context():
            for batch, total in self._batches(self.layout.fragments(transfers)):
                allocated = total + ALIGNMENT - 1
                self.budget.acquire(allocated)
                owner = stage = None
                failure = None
                try:
                    owner = torch.empty(
                        allocated, dtype=torch.uint8, device=self.device
                    )
                    offset = (-owner.data_ptr()) % ALIGNMENT
                    stage = owner.narrow(0, offset, total)
                    self._transfer_batch(batch=batch, stage=stage, write=write)
                except Exception as error:
                    failure = f"{type(error).__name__}: {error}"
                finally:
                    # Copies on an exceptional path still own their source allocation.
                    try:
                        self._sync()
                    finally:
                        owner = stage = None
                        self.budget.release(allocated)
                if failure is not None:
                    raise RuntimeError(failure)

    def _transfer_batch(self, *, batch, stage: torch.Tensor, write: bool) -> None:
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
            if write:
                view.zero_()
                view[: fragment.source.numel()].copy_(fragment.source)
            offset += fragment.padded_length
        if write:
            self._sync()
            results = self.manager.put_gpu_file(keys, pointers, sizes)
        else:
            addresses = self.manager.lookup_addresses(keys)
            if len(addresses) != len(keys) or not all(addresses):
                raise RuntimeError("Flat GPU restore lost a required payload fragment")
            if any(address % ALIGNMENT for address in addresses):
                raise RuntimeError("Flat GPU storage offset is not 4096-byte aligned")
            results = self.manager.read_gpu(addresses, pointers, sizes)
        if len(results) != len(batch) or not all(results):
            operation = "write" if write else "read"
            raise RuntimeError(
                f"Flat GPU {operation} did not complete every payload fragment"
            )
        if not write:
            for fragment, view in zip(batch, views):
                fragment.source.copy_(view[: fragment.source.numel()])
            self._sync()
        logical = sum(fragment.source.numel() for fragment in batch)
        padded = sum(sizes)
        with self._metrics_lock:
            if write:
                self.logical_write_bytes += logical
                self.padded_write_bytes += padded
            else:
                self.logical_read_bytes += logical
                self.padded_read_bytes += padded

    def lookup(self, *, keys: list[str], transfers: list[PoolTransfer]) -> list[int]:
        candidates = set(range(1, len(keys) + 1))
        for transfer in transfers:
            states = self.page_addresses(name=transfer.name, keys=keys)
            exists = [bool(addresses) and all(addresses) for addresses in states]
            candidates &= restorable_boundaries(
                exists,
                policy=transfer.hit_policy,
                trailing_pages=max(1, len(transfer.keys or [])),
            )
            if not candidates:
                break
        return sorted(candidates)

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
            }
