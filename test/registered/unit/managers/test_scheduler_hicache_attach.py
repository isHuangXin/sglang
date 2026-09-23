"""Runtime HiCache attach/detach lands on the config bags.

The attach RPC used to mutate the scheduler's ServerArgs so the readback would
show the change; the namespace readers never saw it. Both now go through
get_context().override, so get_memory() and the resolved-config readback agree
and the published instance stays as the launcher left it.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.managers.io_struct import (
    AttachHiCacheStorageReqInput,
    DetachHiCacheStorageReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_memory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerHiCacheAttach(CustomTestCase):
    def _scheduler(self, **fields):
        override = get_context().override_server_args(
            enable_hierarchical_cache=True, **fields
        )
        self.server_args = override.install()
        self.addCleanup(override.restore)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = self.server_args
        scheduler.enable_hierarchical_cache = True
        scheduler.enable_hicache_storage = False
        scheduler.is_fully_idle = lambda: True
        scheduler.tree_cache = SimpleNamespace(
            attach_storage_backend=lambda **kwargs: (True, "attached"),
            detach_storage_backend=lambda: (True, "detached"),
        )
        return scheduler

    def test_pressure_retry_works_without_miss_retry_polling(self):
        scheduler = self._scheduler(hicache_storage_prefetch_retry_poll_interval=0)
        scheduler.enable_hicache_storage = True
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache._storage_prefetch_deferred_rids = {"r"}
        scheduler.tree_cache = cache
        scheduler._prefetch_kvcache = MagicMock()
        req = SimpleNamespace(rid="r", storage_prefetch_retry_attempts=0)

        scheduler._retry_deferred_storage_prefetch(req)
        scheduler._retry_deferred_storage_prefetch(req)

        scheduler._prefetch_kvcache.assert_called_once_with(req)
        self.assertEqual(req.storage_prefetch_retry_attempts, 0)
        self.assertEqual(get_memory().hicache_storage_prefetch_retry_poll_interval, 0)

    def test_rejected_pressure_retry_does_not_spend_query_attempts(self):
        scheduler = self._scheduler(hicache_storage_prefetch_retry_poll_interval=0)
        scheduler.enable_hicache_storage = True
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache._storage_prefetch_deferred_rids = {"r"}
        scheduler.tree_cache = cache
        scheduler._prefetch_kvcache = MagicMock(
            side_effect=lambda req: cache._storage_prefetch_deferred_rids.add(req.rid)
        )
        req = SimpleNamespace(rid="r", storage_prefetch_retry_attempts=2)

        scheduler._retry_deferred_storage_prefetch(req)

        self.assertTrue(cache.pop_storage_prefetch_deferred("r"))
        self.assertEqual(req.storage_prefetch_retry_attempts, 2)

    def test_no_pressure_retry_when_storage_is_disabled(self):
        scheduler = self._scheduler()
        scheduler._prefetch_kvcache = MagicMock()
        scheduler._retry_deferred_storage_prefetch(SimpleNamespace(rid="r"))
        scheduler._prefetch_kvcache.assert_not_called()

    def test_attach_reaches_the_namespace_readers(self):
        scheduler = self._scheduler(hicache_storage_backend=None)
        out = scheduler.attach_hicache_storage_wrapped(
            AttachHiCacheStorageReqInput(
                hicache_storage_backend="file",
                hicache_write_policy="write_through",
            )
        )

        self.assertTrue(out.success)
        self.assertEqual(get_memory().hicache_storage_backend, "file")
        self.assertEqual(get_memory().hicache_write_policy, "write_through")
        self.assertEqual(
            get_context().resolved_server_args_dict()["hicache_storage_backend"],
            "file",
        )
        self.assertIsNone(self.server_args.hicache_storage_backend)

    def test_detach_clears_the_backend_for_the_same_readers(self):
        scheduler = self._scheduler(hicache_storage_backend="file")
        scheduler.enable_hicache_storage = True

        out = scheduler.detach_hicache_storage_wrapped(DetachHiCacheStorageReqInput())

        self.assertTrue(out.success)
        self.assertIsNone(get_memory().hicache_storage_backend)
        self.assertIsNone(
            get_context().resolved_server_args_dict()["hicache_storage_backend"]
        )
        # The record is not written any more: the attach is a declaration on
        # it and the detach is a bag override (asserted above), so the two are
        # meant to differ here.
        self.assertEqual(
            resolution_result(self.server_args, "hicache_storage_backend"), "file"
        )


if __name__ == "__main__":
    unittest.main()
