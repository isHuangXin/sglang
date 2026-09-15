"""A Flat read must not overwrite a staging allocation's pending prior user."""

import ctypes
import ctypes.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

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


class TestFlatMemoryPayloadStream(CustomTestCase):
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
                transfer, "_stream_context", side_effect=lambda: torch.cuda.stream(stream)
            ):
                transfer.read([])
            torch.cuda.synchronize()
            aligned_old = (old_pointer + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
            self.assertEqual(manager.pointer, aligned_old, "Must exercise allocator reuse")
            self.assertEqual(int((observed != 17).sum().item()), 0)
            self.assertTrue(bool((destination == 119).all().item()))
            self.assertEqual(transfer.budget.current_bytes, 0)


if __name__ == "__main__":
    unittest.main()
