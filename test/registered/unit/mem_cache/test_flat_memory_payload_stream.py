"""A Flat read must not overwrite a staging allocation's pending prior user."""

import ctypes
import ctypes.util
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.flat_memory.flat_memory_direct_linker import (
    FlatMemoryDirectLinker,
)
from sglang.srt.mem_cache.storage.flat_memory.payload import (
    ALIGNMENT,
    GPUTransfer,
    PayloadFragment,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _NativeReader:
    def __init__(self):
        self.cuda = ctypes.CDLL(ctypes.util.find_library("cudart"))
        self.cuda.cudaStreamCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        self.cuda.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.cuda.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.cuda.cudaMemsetAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self.stream = ctypes.c_void_p()
        self._check(self.cuda.cudaStreamCreateWithFlags(ctypes.byref(self.stream), 1))
        self.pointer = None

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"CUDA runtime status {status}")

    def lookup_addresses(self, keys):
        return [ALIGNMENT] * len(keys)

    def read_gpu(self, addresses, pointers, sizes):
        self.pointer = pointers[0]
        for pointer, size in zip(pointers, sizes):
            self._check(
                self.cuda.cudaMemsetAsync(
                    ctypes.c_void_p(pointer), 119, size, self.stream
                )
            )
        self._check(self.cuda.cudaStreamSynchronize(self.stream))
        return [True] * len(pointers)

    def close(self):
        self._check(self.cuda.cudaStreamDestroy(self.stream))


class _NativeSnapshotWriter(_NativeReader):
    def __init__(self):
        super().__init__()
        self.cuda.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.objects = {}
        self.pointers = []

    @staticmethod
    def _result(results):
        return {"status": "ok", "results": results, "error": ""}

    def check_gpu_write(self, keys, sizes):
        return self._result([key in self.objects for key in keys])

    def put_gpu_file_detailed(self, keys, pointers, sizes):
        self.entered.set()
        if not self.resume.wait(timeout=10):
            raise RuntimeError("Native write was not resumed")
        for key, pointer, size in zip(keys, pointers, sizes):
            destination = ctypes.create_string_buffer(size)
            self._check(
                self.cuda.cudaMemcpyAsync(
                    ctypes.cast(destination, ctypes.c_void_p),
                    ctypes.c_void_p(pointer),
                    size,
                    2,
                    self.stream,
                )
            )
            self._check(self.cuda.cudaStreamSynchronize(self.stream))
            self.objects[key] = destination.raw
            self.pointers.append(pointer)
        return self._result([True] * len(keys))


class TestFlatMemoryPayloadStream(CustomTestCase):
    def test_snapshot_survives_source_overwrite_before_native_gpu_copy(self):
        """After the pack fence, native batches read the snapshot, never reused source pages."""
        device = torch.device("cuda", 0)
        with torch.cuda.device(device):
            source = torch.arange(3 * 17, dtype=torch.uint8, device=device).reshape(
                3, 17
            )
            expected = source.cpu()
            group = DevicePoolGroup(
                [
                    DevicePoolEntry(
                        name=PoolName.KV,
                        indices_from_pool=PoolName.KV,
                        device_pool=None,
                        components=[[source]],
                        layer_mapping={0: 0},
                        page_size=1,
                        rows_are_pages=True,
                    )
                ],
                num_layers=1,
                page_size=1,
            )
            manager = _NativeSnapshotWriter()
            linker = FlatMemoryDirectLinker(
                pool_group=group,
                config={
                    "gds_max_io_bytes": ALIGNMENT,
                    "gpu_stage_bytes": 8 * ALIGNMENT,
                    "gpu_batch_pages": 1,
                },
                model_namespace="snapshot-stream-reuse",
                tp_rank=0,
                tp_size=1,
                device=device,
                manager=manager,
            )
            try:
                linker.offload(
                    [
                        PoolTransfer(
                            name=PoolName.KV,
                            keys=["a", "b", "c"],
                            device_indices=torch.tensor([0, 1, 2], device=device),
                        )
                    ]
                )
                self.assertTrue(manager.entered.wait(timeout=5))
                self.assertEqual(linker.num_source_safe_offloads(), 1)
                self.assertEqual(linker.num_completed_offloads(), 0)
                source.fill_(231)
                torch.cuda.synchronize(device)
                manager.resume.set()
                self.assertTrue(linker.drain(timeout=5))
                self.assertTrue(linker.pop_completed_offload())
                self.assertEqual(linker.transfer.snapshot_writes, 1)
                self.assertEqual(linker.transfer.budget.peak_bytes, 4 * ALIGNMENT - 1)
                self.assertEqual(linker.transfer.budget.current_bytes, 0)
                self.assertEqual(
                    manager.pointers,
                    [manager.pointers[0] + i * ALIGNMENT for i in range(3)],
                )
                for row, stored in zip(expected, manager.objects.values()):
                    self.assertEqual(stored[: row.numel()], bytes(row.tolist()))
                    self.assertEqual(
                        stored[row.numel() :], bytes(ALIGNMENT - row.numel())
                    )
            finally:
                manager.resume.set()
                linker.close()

    def test_read_orders_recycled_staging_after_its_previous_user(self):
        """Native reads must preserve earlier same-stream users of recycled storage."""
        device = torch.device("cuda", 0)
        with torch.cuda.device(device):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            stream = torch.cuda.Stream()
            allocated = 2 * ALIGNMENT - 1
            observed = torch.empty(allocated, dtype=torch.uint8, device=device)
            destination = torch.empty(ALIGNMENT, dtype=torch.uint8, device=device)
            torch.cuda.synchronize()
            layout = SimpleNamespace(
                pool_group=SimpleNamespace(
                    entries=[SimpleNamespace(components=[[destination]])]
                ),
                max_io_bytes=ALIGNMENT,
                specs={"test": [SimpleNamespace(padded_length=ALIGNMENT)]},
                fragments=lambda transfers: iter(
                    [PayloadFragment("page", destination, ALIGNMENT)]
                ),
            )
            manager = _NativeReader()
            self.addCleanup(manager.close)
            transfer = GPUTransfer(
                layout=layout,
                manager=manager,
                device=device,
                staging_bytes=2 * ALIGNMENT,
                batch_pages=1,
            )
            previous_done = torch.cuda.Event()
            with torch.cuda.stream(stream):
                old = torch.full((allocated,), 17, dtype=torch.uint8, device=device)
                old_pointer = old.data_ptr()
                stream.synchronize()
                torch.cuda._sleep(500_000_000)
                observed.copy_(old)
                previous_done.record()
                del old
            self.assertFalse(previous_done.query())
            with patch.object(
                transfer,
                "_stream_context",
                side_effect=lambda: torch.cuda.stream(stream),
            ):
                transfer.read([])
            torch.cuda.synchronize()
            aligned_old = (old_pointer + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
            self.assertEqual(
                manager.pointer, aligned_old, "Must exercise allocator reuse"
            )
            self.assertEqual(int((observed != 17).sum().item()), 0)
            self.assertTrue(bool((destination == 119).all().item()))
            self.assertEqual(transfer.budget.current_bytes, 0)


if __name__ == "__main__":
    unittest.main()
