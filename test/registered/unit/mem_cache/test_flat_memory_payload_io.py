"""CPU coverage for Flat payload admission, completion, and staging ownership."""

import ctypes
import sys
import threading
import unittest
import weakref
from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.flat_memory import flat_memory_direct_linker
from sglang.srt.mem_cache.storage.flat_memory.flat_memory_direct_linker import (
    FlatMemoryDirectLinker,
)
from sglang.srt.mem_cache.storage.flat_memory.io_result import (
    FlatCapacityError,
    FlatUnsafeIOError,
)
from sglang.srt.mem_cache.storage.flat_memory.payload import (
    ALIGNMENT,
    GPUTransfer,
    PayloadLayout,
    StagingBudget,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _result(status, results, *, error="", requested_bytes=0):
    return {
        "status": status,
        "results": results,
        "error": error,
        "requested_bytes": requested_bytes,
    }


class _NativeIOError(RuntimeError):
    pass


class _CPUManager:
    def __init__(self, capacity):
        self.capacity = capacity
        self.objects = {}
        self.addresses = {}
        self.check_gpu_write = Mock(side_effect=self._check)
        self.put_gpu_file_detailed = Mock(side_effect=self._put)
        self.lookup_addresses = Mock(
            side_effect=lambda keys: [self.addresses.get(key, 0) for key in keys]
        )
        self.read_gpu = Mock(side_effect=self._read)

    def _check(self, keys, sizes):
        existing = [key in self.objects for key in keys]
        requested = sum(size for size, found in zip(sizes, existing) if not found)
        available = self.capacity - sum(map(len, self.objects.values()))
        return _result(
            "ok" if requested <= available else "capacity",
            existing,
            error="" if requested <= available else "configured store has no room",
            requested_bytes=requested,
        )

    def _put(self, keys, pointers, sizes):
        admission = self._check(keys, sizes)
        if admission["status"] != "ok":
            return admission
        for key, pointer, size in zip(keys, pointers, sizes):
            if key not in self.objects:
                self.addresses[key] = (len(self.objects) + 1) * ALIGNMENT
                self.objects[key] = ctypes.string_at(pointer, size)
        return _result(
            "ok", [True] * len(keys), requested_bytes=admission["requested_bytes"]
        )

    def _read(self, addresses, pointers, sizes):
        keys = {address: key for key, address in self.addresses.items()}
        for address, pointer, size in zip(addresses, pointers, sizes):
            ctypes.memmove(pointer, self.objects[keys[address]], size)
        return [True] * len(addresses)


def _request(rows):
    return [
        PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.tensor(rows, dtype=torch.int64, device="cpu"),
            keys=[f"page-{row}" for row in rows],
        )
    ]


def _make_transfer(*, pages=1, capacity=8 * ALIGNMENT, batch_pages=128):
    buffer = torch.arange(pages * 17, dtype=torch.uint8, device="cpu").reshape(
        pages, 17
    )
    entry = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        components=[[buffer]],
        layer_mapping={0: 0},
        page_size=1,
        _row_span=1,
        prepare_locations=lambda indices: indices.tolist(),
    )
    layout = PayloadLayout(
        pool_group=SimpleNamespace(
            entries=[entry], entry_map={PoolName.KV: entry}, page_size=1
        ),
        model_namespace="cpu-payload-io",
        tp_rank=0,
        tp_size=1,
        max_io_bytes=ALIGNMENT,
    )
    manager = _CPUManager(capacity)
    transfer = GPUTransfer(
        layout=layout,
        manager=manager,
        device=torch.device("cpu"),
        staging_bytes=3 * ALIGNMENT,
        batch_pages=batch_pages,
    )
    return transfer, manager, buffer, _request(list(range(pages)))


@contextmanager
def _observe_cleanup(transfer):
    events = []
    owners = []
    allocate = torch.empty
    release = transfer.budget.release

    def track_owner(*args, **kwargs):
        owner = allocate(*args, **kwargs)
        owners.append(weakref.ref(owner))
        return owner

    def sync():
        if not owners or owners[-1]() is None:
            raise AssertionError("Staging owner was released before synchronization")
        events.append(("sync", transfer.budget.current_bytes))

    def release_after_sync(size):
        events.append(("release", transfer.budget.current_bytes))
        release(size)

    with (
        patch.object(torch, "empty", side_effect=track_owner),
        patch.object(transfer, "_sync", side_effect=sync),
        patch.object(transfer.budget, "release", side_effect=release_after_sync),
    ):
        yield events


def _make_linker(transfer):
    linker = FlatMemoryDirectLinker.__new__(FlatMemoryDirectLinker)
    linker.transfer = transfer
    linker.layout = transfer.layout
    linker.device = torch.device("cpu")
    linker.tp_rank = 0
    linker._lock = threading.RLock()
    linker._backup_samples = []
    linker._last_capacity_warning = float("-inf")
    linker._unsafe_error = None
    linker._offloads = deque()
    return linker


class TestFlatPayloadIO(CustomTestCase):
    def test_capacity_preflight_rejects_before_staging_or_packing(self):
        transfer, manager, _, request = _make_transfer(capacity=0)
        with (
            patch.object(torch, "empty") as allocate,
            patch.object(transfer, "_transfer_batch") as pack,
            patch.object(transfer.budget, "acquire") as acquire,
            self.assertRaisesRegex(FlatCapacityError, "configured store has no room"),
        ):
            transfer.write(request)
        allocate.assert_not_called()
        pack.assert_not_called()
        acquire.assert_not_called()
        manager.put_gpu_file_detailed.assert_not_called()
        self.assertEqual(manager.check_gpu_write.call_args.args[1], [ALIGNMENT])
        self.assertEqual(transfer.budget.current_bytes, 0)
        self.assertEqual(transfer.budget.peak_bytes, 0)
        self.assertEqual(transfer.padded_write_bytes, 0)

    def test_duplicate_only_write_skips_staging_when_store_is_full(self):
        transfer, manager, buffer, request = _make_transfer(capacity=ALIGNMENT)
        transfer.write(request)
        stored = dict(manager.objects)
        buffer.fill_(255)
        with (
            patch.object(torch, "empty") as allocate,
            patch.object(transfer, "_transfer_batch") as pack,
            patch.object(transfer.budget, "acquire") as acquire,
        ):
            transfer.write(request)
        allocate.assert_not_called()
        pack.assert_not_called()
        acquire.assert_not_called()
        self.assertEqual(manager.objects, stored)
        self.assertEqual(manager.check_gpu_write.call_count, 2)
        self.assertEqual(manager.put_gpu_file_detailed.call_count, 1)
        self.assertEqual(transfer.budget.current_bytes, 0)

    def test_oversized_rejection_does_not_latch_out_a_smaller_write(self):
        transfer, manager, buffer, request = _make_transfer(pages=2, capacity=ALIGNMENT)
        expected = buffer[0].clone()
        with self.assertRaises(FlatCapacityError):
            transfer.write(request)
        self.assertEqual(manager.objects, {})
        self.assertEqual(transfer.budget.peak_bytes, 0)
        small = _request([0])
        transfer.write(small)
        buffer[0].zero_()
        transfer.read(small)
        torch.testing.assert_close(buffer[0], expected)
        self.assertEqual(len(manager.objects), 1)
        self.assertEqual(transfer.logical_write_bytes, expected.numel())
        self.assertEqual(transfer.budget.current_bytes, 0)

    def test_mixed_duplicates_and_new_keys_still_stage_fresh_payloads(self):
        transfer, manager, buffer, request = _make_transfer(pages=2)
        expected = buffer.clone()
        transfer.write(_request([0]))
        buffer[0].fill_(255)
        transfer.write(request)
        self.assertEqual(manager.put_gpu_file_detailed.call_count, 2)
        self.assertEqual(len(manager.objects), 2)
        buffer.zero_()
        transfer.read(request)
        torch.testing.assert_close(buffer, expected)

    def test_round_trip_uses_aligned_zero_padded_payloads_and_syncs_before_io(self):
        transfer, manager, buffer, request = _make_transfer(pages=2)
        expected = buffer.clone()
        with _observe_cleanup(transfer) as events:
            put = manager.put_gpu_file_detailed.side_effect
            read = manager.read_gpu.side_effect

            def observed_put(keys, pointers, sizes):
                events.append(("write", transfer.budget.current_bytes))
                self.assertTrue(all(pointer % ALIGNMENT == 0 for pointer in pointers))
                self.assertEqual(sizes, [ALIGNMENT, ALIGNMENT])
                return put(keys, pointers, sizes)

            def observed_read(addresses, pointers, sizes):
                events.append(("read", transfer.budget.current_bytes))
                return read(addresses, pointers, sizes)

            manager.put_gpu_file_detailed.side_effect = observed_put
            manager.read_gpu.side_effect = observed_read
            transfer.write(request)
            buffer.zero_()
            transfer.read(request)
        torch.testing.assert_close(buffer, expected)
        for row, stored in zip(expected, manager.objects.values()):
            self.assertEqual(stored[: row.numel()], bytes(row.tolist()))
            self.assertEqual(stored[row.numel() :], bytes(ALIGNMENT - row.numel()))
        self.assertEqual(
            [event for event, _ in events],
            [
                "sync",
                "write",
                "sync",
                "release",
                "sync",
                "read",
                "sync",
                "sync",
                "release",
            ],
        )
        self.assertTrue(all(size == 3 * ALIGNMENT - 1 for _, size in events))
        self.assertEqual(transfer.logical_read_bytes, expected.numel())
        self.assertEqual(transfer.padded_read_bytes, 2 * ALIGNMENT)
        self.assertEqual(transfer.logical_write_bytes, expected.numel())
        self.assertEqual(transfer.padded_write_bytes, 2 * ALIGNMENT)
        self.assertEqual(transfer.budget.current_bytes, 0)

    def test_capacity_race_after_preflight_releases_staging_after_sync(self):
        transfer, manager, _, request = _make_transfer()
        with _observe_cleanup(transfer) as events:

            def reject_write(keys, pointers, sizes):
                events.append(("write", transfer.budget.current_bytes))
                return _result(
                    "capacity", [False], error="another writer won admission"
                )

            manager.put_gpu_file_detailed.side_effect = reject_write
            with self.assertRaisesRegex(
                FlatCapacityError, "another writer won admission"
            ):
                transfer.write(request)
        manager.check_gpu_write.assert_called_once()
        manager.put_gpu_file_detailed.assert_called_once()
        self.assertEqual(
            [event for event, _ in events], ["sync", "write", "sync", "release"]
        )
        self.assertTrue(all(size == 2 * ALIGNMENT - 1 for _, size in events))
        self.assertEqual(transfer.budget.current_bytes, 0)
        self.assertEqual(transfer.padded_write_bytes, 0)
        transfer.raise_if_unsafe()

    def test_io_failures_preserve_diagnostics_without_copying_failed_reads(self):
        cases = (
            ("write_status", RuntimeError, "native pwrite returned EIO"),
            ("write_exception", _NativeIOError, "write completion failed: errno 5"),
            ("read_exception", _NativeIOError, "read completion failed: errno 5"),
            ("read_false", RuntimeError, "did not complete every payload fragment"),
            ("read_short", RuntimeError, "did not complete every payload fragment"),
        )
        for mode, error_type, diagnostic in cases:
            with self.subTest(mode=mode):
                transfer, manager, buffer, request = _make_transfer(pages=2)
                transfer.write(request)
                buffer.fill_(123)
                before = buffer.clone()
                native_error = _NativeIOError(diagnostic)
                reading = mode.startswith("read")
                with _observe_cleanup(transfer) as events:

                    def fail_io(keys_or_addresses, pointers, sizes):
                        events.append(("native", transfer.budget.current_bytes))
                        if reading:
                            for pointer, size in zip(pointers, sizes):
                                ctypes.memset(pointer, 7, size)
                        if mode.endswith("exception"):
                            raise native_error
                        if mode == "write_status":
                            return _result("io_error", [False, False], error=diagnostic)
                        return [True, False] if mode == "read_false" else [True]

                    if reading:
                        manager.read_gpu.side_effect = fail_io
                    else:
                        manager.check_gpu_write.side_effect = (
                            lambda keys, sizes: _result("ok", [False, False])
                        )
                        manager.put_gpu_file_detailed.side_effect = fail_io
                    with self.assertRaisesRegex(error_type, diagnostic) as raised:
                        (transfer.read if reading else transfer.write)(request)
                self.assertNotIsInstance(raised.exception, FlatCapacityError)
                self.assertNotIsInstance(raised.exception, FlatUnsafeIOError)
                if mode.endswith("exception"):
                    self.assertIs(raised.exception, native_error)
                torch.testing.assert_close(buffer, before)
                self.assertEqual(
                    [event for event, _ in events],
                    ["sync", "native", "sync", "release"],
                )
                self.assertTrue(all(size == 3 * ALIGNMENT - 1 for _, size in events))
                self.assertEqual(transfer.budget.current_bytes, 0)
                self.assertEqual(transfer._retained_staging, [])
                self.assertEqual(transfer.logical_read_bytes, 0)
                transfer.raise_if_unsafe()

    def test_successful_earlier_batch_survives_later_capacity_rejection(self):
        transfer, manager, buffer, request = _make_transfer(
            pages=3, capacity=ALIGNMENT, batch_pages=1
        )
        expected = buffer[0].clone()
        with self.assertRaises(FlatCapacityError):
            transfer.write(request)
        self.assertEqual(manager.check_gpu_write.call_count, 2)
        self.assertEqual(manager.put_gpu_file_detailed.call_count, 1)
        self.assertEqual(len(manager.objects), 1)
        self.assertEqual(transfer.logical_write_bytes, expected.numel())
        self.assertEqual(transfer.padded_write_bytes, ALIGNMENT)
        self.assertEqual(transfer.budget.current_bytes, 0)
        buffer[0].zero_()
        transfer.read(_request([0]))
        torch.testing.assert_close(buffer[0], expected)

    def test_malformed_write_results_are_not_capacity_pressure(self):
        cases = (
            (_result("capacity", [], error="bad count"), "wrong number of results"),
            (_result("ok", []), "wrong number of results"),
            (
                _result("unknown_status", [False], error="bad protocol"),
                "unknown_status",
            ),
        )
        for preflight in (True, False):
            for result, diagnostic in cases:
                with self.subTest(preflight=preflight, result=result):
                    transfer, manager, _, request = _make_transfer()
                    method = (
                        manager.check_gpu_write
                        if preflight
                        else manager.put_gpu_file_detailed
                    )
                    method.side_effect = None
                    method.return_value = result
                    with self.assertRaisesRegex(RuntimeError, diagnostic) as raised:
                        transfer.write(request)
                    self.assertNotIsInstance(raised.exception, FlatCapacityError)
                    self.assertEqual(transfer.budget.current_bytes, 0)
                    self.assertEqual(transfer.padded_write_bytes, 0)
                    if preflight:
                        self.assertEqual(transfer.budget.peak_bytes, 0)
                        manager.put_gpu_file_detailed.assert_not_called()

    def test_ok_status_with_incomplete_write_is_an_io_error(self):
        transfer, manager, _, request = _make_transfer()
        manager.put_gpu_file_detailed.side_effect = None
        manager.put_gpu_file_detailed.return_value = _result("ok", [False])
        with self.assertRaisesRegex(RuntimeError, "did not complete every") as raised:
            transfer.write(request)
        self.assertNotIsInstance(raised.exception, FlatCapacityError)
        self.assertEqual(transfer.budget.current_bytes, 0)
        self.assertEqual(transfer.padded_write_bytes, 0)

    def test_cleanup_sync_failure_quarantines_owner_and_blocks_future_io(self):
        transfer, manager, _, request = _make_transfer()
        sync_error = _NativeIOError("stream completion cannot be established")
        owners = []
        allocate = torch.empty

        def track_owner(*args, **kwargs):
            owner = allocate(*args, **kwargs)
            owners.append(weakref.ref(owner))
            return owner

        with (
            patch.object(torch, "empty", side_effect=track_owner),
            patch.object(transfer, "_sync", side_effect=[None, sync_error]),
            patch.object(transfer.budget, "release") as release,
            self.assertRaisesRegex(FlatUnsafeIOError, str(sync_error)) as raised,
        ):
            transfer.write(request)
        release.assert_not_called()
        self.assertIs(raised.exception.__cause__, sync_error)
        self.assertEqual(len(transfer._retained_staging), 1)
        self.assertIs(transfer._retained_staging[0], owners[0]())
        self.assertEqual(transfer._retained_staging[0].device.type, "cpu")
        self.assertEqual(transfer.budget.current_bytes, 2 * ALIGNMENT - 1)
        self.assertEqual(transfer.padded_write_bytes, 0)
        manager.check_gpu_write.reset_mock()
        manager.put_gpu_file_detailed.reset_mock()
        manager.lookup_addresses.reset_mock()
        with patch.object(torch, "empty") as allocate:
            for operation in (transfer.write, transfer.read):
                with self.assertRaises(FlatUnsafeIOError) as next_error:
                    operation(request)
                self.assertIs(next_error.exception, raised.exception)
        allocate.assert_not_called()
        manager.check_gpu_write.assert_not_called()
        manager.put_gpu_file_detailed.assert_not_called()
        manager.lookup_addresses.assert_not_called()
        with self.assertRaises(FlatUnsafeIOError) as budget_error:
            transfer.budget.acquire(1)
        self.assertIs(budget_error.exception, raised.exception)

    def test_native_unsafe_error_keeps_staging_without_retrying_sync(self):
        transfer, manager, _, request = _make_transfer()
        failure = FlatUnsafeIOError("native DMA has not quiesced")
        manager.put_gpu_file_detailed.side_effect = failure
        with (
            patch.object(transfer, "_sync") as sync,
            patch.object(transfer.budget, "release") as release,
            self.assertRaises(FlatUnsafeIOError) as raised,
        ):
            transfer.write(request)
        self.assertIs(raised.exception, failure)
        sync.assert_called_once()
        release.assert_not_called()
        self.assertEqual(len(transfer._retained_staging), 1)
        self.assertEqual(transfer.budget.current_bytes, 2 * ALIGNMENT - 1)
        with self.assertRaises(FlatUnsafeIOError) as poisoned:
            transfer.budget.acquire(1)
        self.assertIs(poisoned.exception, failure)

    def test_poison_wakes_a_blocked_staging_budget_waiter(self):
        budget = StagingBudget(2 * ALIGNMENT)
        budget.acquire(budget.total_bytes)
        waiting = threading.Event()
        completed = Future()
        error = FlatUnsafeIOError("staging completion failed")
        condition_wait = budget._condition.wait

        def observed_wait(timeout=None):
            waiting.set()
            return condition_wait(timeout)

        def acquire():
            try:
                budget.acquire(1)
            except BaseException as failure:
                completed.set_exception(failure)
            else:
                completed.set_result(None)

        thread = threading.Thread(target=acquire, daemon=True)
        with patch.object(budget._condition, "wait", side_effect=observed_wait):
            thread.start()
            try:
                self.assertTrue(waiting.wait(timeout=2), "waiter did not block")
                budget.poison(error)
                with self.assertRaises(FlatUnsafeIOError) as raised:
                    completed.result(timeout=2)
                self.assertIs(raised.exception, error)
                self.assertEqual(budget.current_bytes, budget.total_bytes)
            finally:
                budget.release(budget.total_bytes)
                thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        with self.assertRaises(FlatUnsafeIOError) as raised:
            budget.acquire(1)
        self.assertIs(raised.exception, error)


class TestFlatLinkerWriteIO(CustomTestCase):
    def test_capacity_warning_is_rate_limited_without_error_or_success_samples(self):
        transfer, _, _, request = _make_transfer(capacity=0)
        linker = _make_linker(transfer)
        with (
            patch.object(flat_memory_direct_linker, "logger") as logger,
            patch.object(
                flat_memory_direct_linker.time,
                "monotonic",
                side_effect=[100.0, 104.999, 105.0],
            ),
        ):
            results = [linker._write(request, None) for _ in range(3)]
        self.assertEqual(logger.warning.call_count, 2)
        logger.exception.assert_not_called()
        self.assertEqual(linker._backup_samples, [])
        for result in results:
            self.assertFalse(result.success)
            self.assertTrue(result.capacity_rejected)
            self.assertFalse(result.unsafe)
            self.assertIn("configured store has no room", result.error)
        for result in results[:2]:
            future = Future()
            future.set_result(result)
            linker._offloads.append(future)
        self.assertIs(linker.pop_completed_offload_result(), results[0])
        self.assertIs(linker.pop_completed_offload(), False)

    def test_io_and_unsafe_failures_log_real_exceptions_and_keep_typed_results(self):
        for failure in (
            _NativeIOError("pwrite failed with errno 5"),
            FlatUnsafeIOError("DMA completion is unknown"),
        ):
            with self.subTest(failure=type(failure).__name__):
                transfer, manager, _, request = _make_transfer()
                linker = _make_linker(transfer)
                manager.put_gpu_file_detailed.side_effect = failure
                logged = []
                with patch.object(flat_memory_direct_linker, "logger") as logger:
                    logger.exception.side_effect = lambda *args: logged.append(
                        sys.exc_info()[1]
                    )
                    result = linker._write(request, None)
                self.assertFalse(result.success)
                self.assertFalse(result.capacity_rejected)
                self.assertEqual(result.unsafe, isinstance(failure, FlatUnsafeIOError))
                self.assertEqual(result.error, f"{type(failure).__name__}: {failure}")
                self.assertEqual(linker._backup_samples, [])
                logger.warning.assert_not_called()
                logger.exception.assert_called_once()
                self.assertEqual(logged, [failure])
                future = Future()
                future.set_result(result)
                linker._offloads.append(future)
                if result.unsafe:
                    self.assertIs(linker._unsafe_error, failure)
                    with self.assertRaisesRegex(FlatUnsafeIOError, str(failure)):
                        linker.pop_completed_offload()
                else:
                    self.assertIsNone(linker._unsafe_error)
                    self.assertIs(linker.pop_completed_offload(), False)

    def test_successful_completion_records_one_sample_and_retires_in_order(self):
        transfer, _, buffer, request = _make_transfer(pages=2)
        linker = _make_linker(transfer)
        with patch.object(
            flat_memory_direct_linker.time, "perf_counter", side_effect=[10.0, 12.0]
        ):
            result = linker._write(request, None)
        self.assertTrue(result.success)
        self.assertEqual(linker._backup_samples, [(2, buffer.numel(), 2.0)])
        first, second = Future(), Future()
        second.set_result(result)
        linker._offloads.extend((first, second))
        self.assertEqual(linker.num_completed_offloads(), 0)
        with self.assertRaisesRegex(RuntimeError, "No Flat offload has completed"):
            linker.pop_completed_offload_result()
        first.set_result(result)
        self.assertEqual(linker.num_completed_offloads(), 2)
        self.assertIs(linker.pop_completed_offload_result(), result)
        self.assertIs(linker.pop_completed_offload(), True)
        self.assertEqual(linker.num_completed_offloads(), 0)


if __name__ == "__main__":
    unittest.main()
