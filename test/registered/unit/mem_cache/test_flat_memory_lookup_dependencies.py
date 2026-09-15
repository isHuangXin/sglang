"""Flat queries wait only for writes that can improve a restorable boundary."""

import threading
import time
import unittest
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.flat_memory import flat_memory_direct_linker
from sglang.srt.mem_cache.storage.flat_memory.flat_memory_direct_linker import (
    FlatMemoryDirectLinker,
)
from sglang.srt.mem_cache.storage.flat_memory.io_result import FlatCapacityError
from sglang.srt.mem_cache.storage.flat_memory.payload import ALIGNMENT
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SteppedExecutor:
    def __init__(self):
        self.jobs = deque()
        self.submissions = 0

    def submit(self, function, *args, **kwargs):
        future = Future()
        self.jobs.append((future, function, args, kwargs))
        self.submissions += 1
        return future

    def run_next(self):
        future, function, args, kwargs = self.jobs.popleft()
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(function(*args, **kwargs))
            except BaseException as error:
                future.set_exception(error)
        return future

    def run_all(self):
        while self.jobs:
            self.run_next()

    def shutdown(self, wait=True):
        self.run_all()


class _MetadataManager:
    def __init__(self):
        self.addresses = {}
        self.close = Mock()

    def lookup_addresses(self, keys):
        return [self.addresses.get(key, 0) for key in keys]

    def publish(self, layout, name, keys):
        for page_key in keys:
            for key in layout.keys_for_page(name=name, page_key=page_key):
                self.addresses[key] = ALIGNMENT


def _request(keys, *, trailing_pages=None):
    transfers = [PoolTransfer(name=PoolName.KV, keys=list(keys))]
    if trailing_pages is not None:
        transfers.append(
            PoolTransfer(
                name=PoolName.SWA,
                keys=list(keys[-trailing_pages:]),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        )
    return transfers


class TestFlatLookupDependencies(CustomTestCase):
    def make_linker(self, *, sources=None, real_queries=False, backup_wait=10.0):
        sources = sources or {PoolName.KV: PoolName.KV}
        group = DevicePoolGroup(
            [
                DevicePoolEntry(
                    name=name,
                    indices_from_pool=source,
                    device_pool=None,
                    components=[[torch.zeros((16, 8), dtype=torch.uint8)]],
                    layer_mapping={0: 0},
                    page_size=1,
                    rows_are_pages=True,
                )
                for name, source in sources.items()
            ],
            num_layers=1,
            page_size=1,
        )
        manager = _MetadataManager()
        linker = FlatMemoryDirectLinker(
            pool_group=group,
            config={
                "backup_wait_seconds": backup_wait,
                "drain_timeout": 0.2,
                "gds_max_io_bytes": ALIGNMENT,
                "gpu_stage_bytes": 3 * ALIGNMENT,
            },
            model_namespace="query-dependencies",
            tp_rank=0,
            tp_size=1,
            device="cpu",
            manager=manager,
        )
        linker._write_executor.shutdown(wait=True)
        linker._write_executor = _SteppedExecutor()
        if not real_queries:
            linker._lookup_executor.shutdown(wait=True)
            linker._lookup_executor = _SteppedExecutor()

        def write(transfers, *, on_source_consumed=None):
            if on_source_consumed is not None:
                on_source_consumed()
            for transfer in transfers:
                manager.publish(linker.layout, transfer.name, transfer.keys)

        linker.transfer.write = Mock(side_effect=write)
        linker.transfer.lookup_probe = Mock(wraps=linker.transfer.lookup_probe)

        def cleanup():
            for rid in list(linker._lookups):
                linker.release_lookup(rid)
            linker._write_executor.run_all()
            if not real_queries:
                linker._lookup_executor.run_all()
            linker.close()

        self.addCleanup(cleanup)
        return linker

    def publish(self, linker, keys, *, name=PoolName.KV):
        linker.manager.publish(linker.layout, name, keys)

    def run_probe(self, linker):
        linker._lookup_executor.run_next()

    def test_source_receipt_does_not_retire_durable_lookup_dependency(self):
        linker = self.make_linker()
        linker.offload(_request(["a"]))
        record = linker._offloads[0]
        query = linker.submit_lookup("waiting", _request(["a"]))
        self.run_probe(linker)
        record.source_consumed.set_result(None)
        self.assertEqual(linker.num_source_safe_offloads(), 1)
        self.assertEqual(linker.num_completed_offloads(), 0)
        self.assertIn(record.future, linker._offload_coverage)
        self.assertEqual(linker.storage_generation, 0)
        linker.poll_queries()
        self.assertFalse(query.done())
        self.assertFalse(linker._lookup_executor.jobs)
        self.assertFalse(linker.drain(timeout=0))
        linker.transfer.write.side_effect = lambda transfers, **kwargs: self.publish(
            linker, ["a"]
        )
        linker._write_executor.run_next()
        linker.poll_queries()
        self.run_probe(linker)
        self.assertEqual(query.result(), [1])
        self.assertTrue(linker.pop_completed_offload())
        self.assertEqual(linker.num_source_safe_offloads(), 0)

    def test_failed_no_receipt_prefix_only_unblocks_after_durable_retirement(self):
        linker = self.make_linker()
        linker.offload(_request(["failed"]))
        linker.offload(_request(["safe"]))
        linker.offload(_request(["queued"]))
        write = linker.transfer.write.side_effect
        linker.transfer.write.side_effect = FlatCapacityError("full")
        linker._write_executor.run_next()
        linker.transfer.write.side_effect = write
        linker._write_executor.run_next()
        self.assertEqual(linker.num_source_safe_offloads(), 0)
        self.assertEqual(linker.num_completed_offloads(), 2)
        self.assertFalse(linker.pop_completed_offload())
        self.assertEqual(linker.num_source_safe_offloads(), 1)
        self.assertTrue(linker.pop_completed_offload())
        self.assertEqual(linker.num_source_safe_offloads(), 0)
        self.assertEqual(linker.num_completed_offloads(), 0)

    def test_unrelated_backup_does_not_block_partial_or_next_request(self):
        """An absent new tail must not occupy the single query worker for ten seconds."""
        linker = self.make_linker(real_queries=True)
        self.publish(linker, ["cached"])
        linker.offload(_request(["unrelated"]))
        partial = linker.submit_lookup("partial", _request(["cached", "new"]))
        self.assertEqual(partial.result(timeout=1), [1])
        next_query = linker.submit_lookup("next", _request(["cached"]))
        self.assertEqual(next_query.result(timeout=1), [1])
        self.assertFalse(linker._offloads[0].future.done())
        self.assertEqual(linker.transfer.lookup_probe.call_count, 2)

    def test_relevant_backup_defers_logically_and_improves_boundary(self):
        linker = self.make_linker()
        self.publish(linker, ["a"])
        request = _request(["a", "b", "never-written"])
        linker.offload(_request(["b"]))
        result = linker.submit_lookup("waiting", request)
        self.run_probe(linker)
        self.assertFalse(result.done())
        self.assertFalse(linker._physical_query_jobs)
        self.assertEqual(linker.get_lookup_generation("waiting"), 0)

        independent = linker.submit_lookup("next", _request(["a"]))
        self.run_probe(linker)
        self.assertEqual(independent.result(), [1])
        linker._write_executor.run_next()
        self.assertEqual(linker.storage_generation, 1)
        self.assertFalse(linker._physical_query_jobs)
        linker.poll_queries()
        linker.poll_queries()
        self.assertEqual(len(linker._lookup_executor.jobs), 1)
        self.run_probe(linker)
        self.assertEqual(result.result(), [1, 2])
        self.assertIs(result, linker.submit_lookup("waiting", request))
        self.assertEqual(linker.get_lookup_generation("waiting"), 1)
        self.assertEqual(linker.num_completed_offloads(), 1)
        self.assertTrue(linker.pop_completed_offload())

    def test_resolved_offload_coverage_is_immutable_and_namespaced(self):
        linker = self.make_linker(
            sources={PoolName.KV: PoolName.KV, PoolName.INDEXER: PoolName.KV}
        )
        transfers = _request(["a"])
        linker.offload(transfers)
        job = linker._offloads[0].future
        coverage = linker._offload_coverage[job]
        self.assertIsInstance(coverage, frozenset)
        self.assertEqual(
            coverage,
            frozenset(
                (name, f"{linker.layout.namespace}:a")
                for name in (PoolName.KV, PoolName.INDEXER)
            ),
        )
        transfers[0].keys[0] = "mutated"
        linker._write_executor.run_next()
        self.assertEqual(linker.lookup("check", _request(["a"])), [1])
        self.assertEqual(linker.lookup("check", _request(["mutated"])), [])
        self.assertNotIn(job, linker._offload_coverage)

    def test_missing_sidecar_or_gap_is_not_a_satisfiable_dependency(self):
        linker = self.make_linker(
            sources={PoolName.KV: PoolName.KV, PoolName.INDEXER: PoolName.KV}
        )
        self.publish(linker, ["a", "b"])
        self.publish(linker, ["a"], name=PoolName.INDEXER)
        expanded = linker.pool_group.resolve_transfers(_request(["a", "b", "new"]))
        for records in (
            frozenset({linker.layout.page_record(name=PoolName.KV, page_key="b")}),
            frozenset({(PoolName.INDEXER, "wrong-namespace:b")}),
            frozenset(
                {linker.layout.page_record(name=PoolName.INDEXER, page_key="new")}
            ),
        ):
            with self.subTest(records=records):
                probe = linker.transfer.lookup_probe(
                    keys=["a", "b", "new"], transfers=expanded, pending_coverage=records
                )
                self.assertEqual(probe.boundaries, (1,))
                self.assertFalse(probe.missing_records)

    def test_sparse_swa_dependencies_only_enable_better_valid_boundaries(self):
        linker = self.make_linker(
            sources={PoolName.KV: PoolName.KV, PoolName.SWA: PoolName.SWA}
        )
        keys = [f"p{i}" for i in range(6)] + ["new"]
        self.publish(linker, keys[:-1])
        self.publish(linker, ["p0", "p3"], name=PoolName.SWA)
        expanded = linker.pool_group.resolve_transfers(_request(keys, trailing_pages=2))
        records = frozenset(
            linker.layout.page_record(name=PoolName.SWA, page_key=key)
            for key in ("p1", "p4", "new")
        )
        probe = linker.transfer.lookup_probe(
            keys=keys, transfers=expanded, pending_coverage=records
        )
        self.assertEqual(probe.boundaries, (1,))
        self.assertEqual(
            probe.missing_records,
            records - {linker.layout.page_record(name=PoolName.SWA, page_key="new")},
        )
        self.publish(linker, ["p4"], name=PoolName.SWA)
        probe = linker.transfer.lookup_probe(
            keys=keys, transfers=expanded, pending_coverage=records
        )
        self.assertEqual(probe.boundaries, (1, 5))
        self.assertFalse(probe.missing_records)

    def test_failed_backup_is_not_reported_present_and_advances_generation(self):
        for error in (FlatCapacityError("full"), OSError("write failed")):
            with self.subTest(error=type(error).__name__):
                linker = self.make_linker()
                linker.transfer.write.side_effect = error
                linker.offload(_request(["a"]))
                result = linker.submit_lookup("request", _request(["a"]))
                self.run_probe(linker)
                self.assertFalse(result.done())
                linker._write_executor.run_next()
                linker.poll_queries()
                self.run_probe(linker)
                self.assertEqual(result.result(), [])
                self.assertEqual(linker.storage_generation, 1)
                offload = linker.pop_completed_offload_result()
                self.assertFalse(offload.success)
                self.assertEqual(
                    offload.capacity_rejected, isinstance(error, FlatCapacityError)
                )
                self.assertFalse(linker.manager.addresses)

    def test_partial_failure_refreshes_a_completed_miss(self):
        linker = self.make_linker()
        request = _request(["a", "b"])
        old = linker.submit_lookup("request", request)
        self.run_probe(linker)
        self.assertEqual(old.result(), [])

        def publish_then_fail(transfers, *, on_source_consumed=None):
            self.publish(linker, ["a"])
            raise OSError("later batch failed")

        linker.transfer.write.side_effect = publish_then_fail
        linker.offload(request)
        linker._write_executor.run_next()
        self.assertLess(
            linker.get_lookup_generation("request"), linker.storage_generation
        )
        refreshed = linker.refresh_lookup("request", request)
        self.assertIsNot(refreshed, old)
        self.assertIs(linker.refresh_lookup("request", request), refreshed)
        self.run_probe(linker)
        self.assertEqual(refreshed.result(), [1])
        self.assertEqual(old.result(), [])
        self.assertEqual(linker.get_lookup_generation("request"), 1)
        linker.release_lookup("request")
        self.assertEqual(linker.get_lookup_generation("request"), -1)

    def test_completion_during_probe_keeps_a_conservative_generation(self):
        linker = self.make_linker()
        linker.offload(_request(["a"]))
        result = linker.submit_lookup("request", _request(["a"]))
        linker._write_executor.run_next()
        self.run_probe(linker)
        self.assertEqual(result.result(), [1])
        self.assertEqual(linker.get_lookup_generation("request"), 0)
        self.assertEqual(linker.storage_generation, 1)

    def test_cancelled_identity_cannot_requeue_a_reused_request_id(self):
        linker = self.make_linker()
        linker.offload(_request(["a"]))
        old = linker.submit_lookup("same", _request(["a"]))
        linker.release_lookup("same")
        self.assertTrue(old.cancelled())
        replacement = linker.submit_lookup("same", _request(["new"]))
        linker.poll_queries()
        self.assertEqual(len(linker._lookup_executor.jobs), 1)
        self.run_probe(linker)
        self.assertFalse(replacement.done())
        self.assertEqual(linker.get_lookup_generation("same"), -1)
        linker.poll_queries()
        self.assertEqual(len(linker._lookup_executor.jobs), 1)
        self.run_probe(linker)
        self.assertEqual(replacement.result(), [])
        linker._write_executor.run_next()
        linker.poll_queries()
        self.assertEqual(linker._lookup_executor.submissions, 2)
        self.assertFalse(linker.has_unfinished_io())

    def test_external_cancel_and_forget_retire_deferred_queries(self):
        linker = self.make_linker()
        linker.offload(_request(["a"]))
        result = linker.submit_lookup("request", _request(["a"]))
        self.run_probe(linker)
        result.cancel()
        linker.poll_queries()
        linker.forget_request("request")
        linker._write_executor.run_next()
        linker.poll_queries()
        self.assertFalse(linker._queries)
        self.assertFalse(linker._lookups)
        self.assertEqual(linker._lookup_executor.submissions, 1)

    def test_cancel_racing_with_final_completion_does_not_set_a_result(self):
        for failure in (None, OSError("metadata failed")):
            with self.subTest(failure=failure):
                linker = self.make_linker()
                linker.transfer.lookup_probe.side_effect = failure
                result = linker.submit_lookup("request", _request(["new"]))
                done = result.done

                def cancel_after_done_check():
                    observed = done()
                    result.cancel()
                    return observed

                with (
                    patch.object(result, "done", side_effect=cancel_after_done_check),
                    patch.object(
                        result, "set_result", wraps=result.set_result
                    ) as finish,
                    patch.object(
                        result, "set_exception", wraps=result.set_exception
                    ) as fail,
                ):
                    self.run_probe(linker)
                    finish.assert_not_called()
                    fail.assert_not_called()
                self.assertTrue(result.cancelled())
                self.assertFalse(linker._queries)
                self.assertFalse(linker._physical_query_jobs)

    def test_probe_error_retires_physical_work_and_preserves_the_error(self):
        linker = self.make_linker()
        error = OSError("metadata failed")
        linker.transfer.lookup_probe.side_effect = error
        result = linker.submit_lookup("request", _request(["new"]))
        self.run_probe(linker)
        self.assertIs(result.exception(), error)
        self.assertFalse(linker.has_unfinished_io())
        linker.reset()
        self.assertFalse(linker._lookups)

    def test_deadline_final_probe_does_not_wait_for_unfinished_backup(self):
        linker = self.make_linker()
        clock = SimpleNamespace(
            monotonic=lambda: 100.0, perf_counter=time.perf_counter, sleep=time.sleep
        )
        with patch.object(flat_memory_direct_linker, "time", clock):
            linker.offload(_request(["a", "b"]))
            result = linker.submit_lookup("request", _request(["a", "b"]))
            self.run_probe(linker)
            self.assertFalse(result.done())
            self.publish(linker, ["a"])
            clock.monotonic = lambda: 110.0
            linker.poll_queries()
            linker.poll_queries()
            self.assertEqual(len(linker._lookup_executor.jobs), 1)
            self.run_probe(linker)
            self.assertEqual(result.result(), [1])
            self.assertFalse(linker._offloads[0].future.done())
            linker._write_executor.run_next()
            linker.poll_queries()
            self.assertEqual(result.result(), [1])
            self.assertEqual(linker._lookup_executor.submissions, 2)

    def test_overlapping_completions_coalesce_without_duplicate_final_result(self):
        linker = self.make_linker()
        linker.offload(_request(["a"]))
        linker.offload(_request(["b"]))
        result = linker.submit_lookup("request", _request(["a", "b", "new"]))
        completed = Mock()
        result.add_done_callback(completed)
        self.run_probe(linker)
        linker._write_executor.run_next()
        linker.poll_queries()
        linker._write_executor.run_next()
        for _ in range(3):
            linker.poll_queries()
        self.assertEqual(len(linker._lookup_executor.jobs), 1)
        self.run_probe(linker)
        self.assertEqual(result.result(), [1, 2])
        linker.poll_queries()
        completed.assert_called_once_with(result)
        self.assertEqual(linker._lookup_executor.submissions, 2)

    def test_fifo_offload_retirement_and_legacy_bool_results(self):
        linker = self.make_linker()
        linker._write = Mock(side_effect=[True, False])
        linker.offload(_request(["first"]))
        linker.offload(_request(["second"]))
        linker._write_executor.jobs.reverse()
        linker._write_executor.run_next()
        self.assertEqual(linker.num_completed_offloads(), 0)
        self.assertEqual(linker.storage_generation, 1)
        linker._write_executor.run_next()
        self.assertEqual(linker.num_completed_offloads(), 2)
        self.assertFalse(linker.pop_completed_offload())
        self.assertTrue(linker.pop_completed_offload())
        self.assertFalse(linker._offload_coverage)

    def test_reset_timeout_retains_physical_cancelled_probe(self):
        linker = self.make_linker()
        result = linker.submit_lookup("request", _request(["new"]))
        linker.release_lookup("request")
        self.assertTrue(result.cancelled())
        self.assertFalse(linker.drain(timeout=0))
        self.assertTrue(linker._physical_query_jobs)
        linker.drain_timeout = 0.01
        with self.assertRaisesRegex(TimeoutError, "resources remain owned"):
            linker.reset()
        self.assertTrue(linker._physical_query_jobs)
        self.run_probe(linker)
        linker.reset()
        self.assertFalse(linker._physical_query_jobs)
        self.assertFalse(linker._queries)
        self.assertFalse(linker._lookups)

    def test_drain_services_deferred_retry_and_reset_clears_receipts(self):
        linker = self.make_linker(real_queries=True)
        linker.offload(_request(["a"]))
        probe_complete = threading.Event()
        probe = linker.transfer.lookup_probe

        def observed_probe(**kwargs):
            result = probe(**kwargs)
            probe_complete.set()
            return result

        linker.transfer.lookup_probe = observed_probe
        result = linker.submit_lookup("request", _request(["a"]))
        self.assertTrue(probe_complete.wait(timeout=1))
        linker._write_executor.run_next()
        self.assertTrue(linker.drain(timeout=1))
        self.assertEqual(result.result(), [1])
        linker.reset()
        self.assertEqual(linker.get_lookup_generation("request"), -1)
        self.assertFalse(linker._queries)
        self.assertFalse(linker._physical_query_jobs)
        self.assertFalse(linker.has_unfinished_io())


if __name__ == "__main__":
    unittest.main()
