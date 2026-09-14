"""FLAT_MEMORY: Native Mooncake Host-object layout mapped to private GPU pages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolGroup

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host import HostPoolGroup
    from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore


class GPUObjectSlice(msgspec.Struct, frozen=True):
    buffer: torch.Tensor
    offset: int
    size: int
    row_span: int


class GPUPageLayout(msgspec.Struct, frozen=True):
    suffixes: tuple[str, ...]
    sizes: tuple[int, ...]
    slices: tuple[tuple[GPUObjectSlice, ...], ...]


class GPUObjectRead(msgspec.Struct, frozen=True):
    key: str
    size: int
    row: int
    page: int
    source_pool: PoolName
    slices: tuple[GPUObjectSlice, ...]


class GPUReadPlan(msgspec.Struct, frozen=True):
    objects: tuple[GPUObjectRead, ...]
    num_pages: int


class GPUReadResult(msgspec.Struct, kw_only=True):
    success: bool = False
    media: list[int] = []
    pool_media: dict[str, list[int]] = {}
    staging: torch.Tensor | None = None
    safe_to_release: bool = False
    error: str | None = None


class MooncakeGDSPayload:
    def __init__(
        self,
        *,
        storage: MooncakeStore,
        device_pools: DevicePoolGroup,
        host_pools: HostPoolGroup,
        stage_bytes: int,
    ):
        self.storage = storage
        self.device_pools = device_pools
        self.host_pools = host_pools
        self.page_size = device_pools.page_size
        self.stage_bytes = stage_bytes
        buffers = [
            buffer for entry in device_pools.entries for buffer in entry.kv_buffer
        ]
        if not buffers or any(buffer.device != buffers[0].device for buffer in buffers):
            raise ValueError("Mooncake GDS physical buffers must share one GPU")
        self.device = buffers[0].device
        if self.device.type != "cuda":
            raise ValueError("Mooncake GDS requires CUDA device buffers")
        if not any(
            entry.indices_from_pool == PoolName.KV for entry in device_pools.entries
        ):
            raise ValueError("Mooncake GDS requires a physical full-prefix payload")
        self.layouts = {
            entry.name: self._build_layout(entry) for entry in device_pools.entries
        }
        self.logical_page_bytes = sum(
            sum(layout.sizes) for layout in self.layouts.values()
        )
        self.physical_page_bytes = {
            name: sum(part.size for parts in layout.slices for part in parts)
            for name, layout in self.layouts.items()
        }
        if any(
            size > stage_bytes
            for layout in self.layouts.values()
            for size in layout.sizes
        ):
            raise ValueError(
                "A native Mooncake object exceeds the bounded GPU staging budget"
            )
        self.device_capacity_bytes = sum(
            buffer.nbytes
            for buffer in {buffer.data_ptr(): buffer for buffer in buffers}.values()
        )
        host_buffers = {
            buffer.data_ptr(): buffer
            for entry in device_pools.entries
            for buffer in storage._iter_host_pool_buffers(
                host_pools.entry_map[entry.name].host_pool
            )
            if buffer is not None
        }
        self.host_capacity_bytes = sum(
            buffer.nbytes for buffer in host_buffers.values()
        )

    def _build_layout(self, entry) -> GPUPageLayout:
        host_entry = self.host_pools.entry_map[entry.name]
        host = host_entry.host_pool
        if host.layout != "page_first_direct" or host.page_size != self.page_size:
            raise ValueError(
                "Tiered Mooncake GDS requires page_first_direct Host pages"
            )
        indices = torch.arange(self.page_size, dtype=torch.int64, device="cpu")
        if entry.name == PoolName.KV:
            names, _, sizes = self.storage._batch_preprocess(["layout"], indices)
        else:
            names, _ = self.storage._get_hybrid_page_component_keys(
                ["layout"], PoolTransfer(name=entry.name)
            )
            pointers, sizes = host.get_page_buffer_meta(indices)
            _, sizes = self.storage._pack_multi_buffer_meta(names, pointers, sizes)
        object_sizes = tuple(
            sum(size) if isinstance(size, (list, tuple)) else size for size in sizes
        )
        expected_objects = 1 if entry.packed else len(entry.components)
        if len(object_sizes) != expected_objects or (
            entry.packed and len(entry.components) != 1
        ):
            raise ValueError(
                f"Unrepresentable native Mooncake object layout for {entry.name}"
            )
        host_layers = {}
        for layer, physical in entry.layer_mapping.items():
            host_layer = host_entry.layer_mapper(layer)
            if host_layer is None or not 0 <= host_layer < host.layer_num:
                raise ValueError(
                    f"GPU layer {layer} has no Host counterpart for {entry.name}"
                )
            previous = host_layers.setdefault(physical, host_layer)
            if previous != host_layer:
                raise ValueError(
                    "Aliased physical layers disagree on Host serialization order"
                )
        slices = [[] for _ in object_sizes]
        for component_index, component in enumerate(entry.components):
            object_index = 0 if entry.packed else component_index
            object_size = object_sizes[object_index]
            if object_size <= 0 or object_size % host.layer_num:
                raise ValueError(f"Invalid Mooncake layer stride for {entry.name}")
            layer_bytes = object_size // host.layer_num
            for physical, buffer in enumerate(component):
                if physical not in host_layers or not buffer.is_contiguous():
                    raise ValueError(
                        f"Unmapped or non-contiguous GPU buffer for {entry.name}"
                    )
                row_span = entry._row_span
                size = buffer.nbytes // buffer.shape[0] * row_span
                if size != layer_bytes:
                    raise ValueError(
                        f"Host/GPU logical byte size differs for {entry.name}: {layer_bytes} != {size}"
                    )
                byte_view = buffer.view(torch.uint8).reshape(buffer.shape[0], -1)
                slices[object_index].append(
                    GPUObjectSlice(
                        byte_view,
                        host_layers[physical] * layer_bytes,
                        size,
                        row_span,
                    )
                )
        # Holes (GLM shared-topk layers) remain in the native object, but have no GPU destination.
        return GPUPageLayout(
            tuple(name.removeprefix("layout") for name in names),
            object_sizes,
            tuple(tuple(parts) for parts in slices),
        )

    def _object_names(self, pool_name, keys):
        layout = self.layouts[pool_name]
        return self.storage._tag_keys(
            [key + suffix for key in keys for suffix in layout.suffixes]
        )

    def lookup(self, transfers: list[PoolTransfer]) -> list[int]:
        by_name = {transfer.name: transfer for transfer in transfers}
        full = by_name.get(PoolName.KV)
        if full is None or not full.keys:
            return []
        physical = self.device_pools.resolve_transfers(transfers)
        if len(physical) != len(self.layouts):
            return []
        count = len(full.keys)
        restorable = set(range(1, count + 1))
        for transfer in physical:
            names = self._object_names(transfer.name, full.keys)
            exists = []
            for start in range(0, len(names), 4096):
                batch = names[start : start + 4096]
                result = self.storage._batch_exist(batch)
                if len(result) != len(batch):
                    raise RuntimeError(
                        "Mooncake existence result length differs from query"
                    )
                exists.extend(result)
            components = len(self.layouts[transfer.name].sizes)
            pages = [
                all(
                    value == 1
                    for value in exists[index * components : (index + 1) * components]
                )
                for index in range(count)
            ]
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                first_miss = next(
                    (i for i, present in enumerate(pages) if not present), count
                )
                restorable.intersection_update(range(1, first_miss + 1))
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                window = len(transfer.keys)
                restorable = {
                    boundary
                    for boundary in restorable
                    if all(pages[max(0, boundary - window) : boundary])
                }
            else:
                raise ValueError(
                    f"Unsupported Mooncake GDS hit policy: {transfer.hit_policy}"
                )
            if not restorable:
                break
        return sorted(restorable)

    def prepare_read(self, transfers: list[PoolTransfer]) -> GPUReadPlan:
        full = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        keys = list(full.keys)
        key_pages = {key: index for index, key in enumerate(keys)}
        if len(key_pages) != len(keys):
            raise ValueError("Mooncake prefix contains duplicate page hashes")
        objects = []
        physical = self.device_pools.resolve_transfers(transfers)
        if len(physical) != len(self.layouts):
            raise ValueError("Mooncake GDS read is missing a required physical pool")
        for transfer in physical:
            entry = self.device_pools.entry_map[transfer.name]
            locations = entry.prepare_locations(transfer.host_indices)
            if len(locations) != len(transfer.keys):
                raise ValueError("Mooncake GDS transfer key/page count differs")
            layout = self.layouts[transfer.name]
            names = self._object_names(transfer.name, transfer.keys)
            for index, (key, row) in enumerate(zip(transfer.keys, locations)):
                for component, size in enumerate(layout.sizes):
                    objects.append(
                        GPUObjectRead(
                            names[index * len(layout.sizes) + component],
                            size,
                            row,
                            key_pages[key],
                            entry.indices_from_pool,
                            layout.slices[component],
                        )
                    )
        if len({item.key for item in objects}) != len(objects):
            raise ValueError("Mooncake GDS read contains duplicate object keys")
        return GPUReadPlan(tuple(objects), len(keys))

    def read(self, plan: GPUReadPlan, result: GPUReadResult) -> None:
        result.media = [0] * plan.num_pages
        result.staging = torch.empty(
            min(self.stage_bytes, sum(item.size for item in plan.objects)),
            dtype=torch.uint8,
            device=self.device,
        )
        stream = torch.cuda.current_stream(self.device)
        offset = 0
        success = True
        while offset < len(plan.objects):
            batch, used = [], 0
            while offset < len(plan.objects) and len(batch) < 4096:
                item = plan.objects[offset]
                if used + item.size > result.staging.numel():
                    break
                batch.append((item, used))
                used += item.size
                offset += 1
            if not batch:
                raise RuntimeError("Mooncake GDS object does not fit GPU staging")
            stream.synchronize()
            status = self.storage.store.batch_get_into_gpu(
                [item.key for item, _ in batch],
                [result.staging.data_ptr() + start for _, start in batch],
                [item.size for item, _ in batch],
            )
            if len(status) != len(batch):
                raise RuntimeError("Mooncake GDS returned an incomplete result vector")
            for (item, start), medium in zip(batch, status):
                if medium not in (1, 2):
                    success = False
                    continue
                for part in item.slices:
                    data = result.staging[
                        start + part.offset : start + part.offset + part.size
                    ]
                    part.buffer[item.row : item.row + part.row_span].copy_(
                        data.reshape(part.row_span, -1)
                    )
                result.media[item.page] |= medium
                pool_media = result.pool_media.setdefault(
                    item.source_pool.value, [0] * plan.num_pages
                )
                pool_media[item.page] |= medium
            # Reuse staging only after every scatter from the previous batch has completed.
            stream.synchronize()
            if not success:
                break
        result.success = success and all(mask in (1, 2, 3) for mask in result.media)
