"""CPU regressions for PR #37424 against the v0.5.19 prefetch contract."""

import importlib
import os
import sys
import unittest
from array import array
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch


def _module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def _cpu_modules():
    """Import whole production files without server, allocator or GPU startup."""
    python_root = Path(__file__).resolve().parents[4] / "python"
    with patch.dict(sys.modules), patch.dict(
        os.environ, {"CUDA_VISIBLE_DEVICES": "0", "SGLANG_IS_IN_CI": "0"}
    ), patch.object(
        torch.cuda,
        "_lazy_init",
        side_effect=AssertionError("CPU test initialized CUDA"),
    ):
        for name in list(sys.modules):
            if name == "sglang" or name.startswith("sglang."):
                del sys.modules[name]
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.managers",
            "sglang.srt.mem_cache",
            "sglang.srt.mem_cache.unified_cache",
            "sglang.srt.mem_cache.unified_cache.components",
            "sglang.srt.observability",
            "sglang.lang",
            "sglang.benchmark",
            "sglang.test",
            "sglang.test.ci",
        ):
            _module(name).__path__ = [str(python_root / name.replace(".", "/"))]
        _module(
            "sglang.srt.managers.cache_controller",
            HICACHE_WRITE_STAGING_POOL_FRACTION=0.2,
        )
        _module("sglang.srt.mem_cache.allocator", BaseTokenToKVPoolAllocator=object)
        _module("sglang.srt.mem_cache.events", KVCacheEventRecorder=object)
        _module("sglang.srt.mem_cache.memory_pool", ReqToTokenPool=object)
        _module(
            "sglang.srt.mem_cache.cache_init_params", CacheInitParams=SimpleNamespace
        )
        _module(
            "sglang.srt.observability.metrics_collector",
            STAT_LOGGER_ROLE_RADIX_CACHE="radix",
            RadixCacheMetricsCollector=object,
            resolve_collector_class=MagicMock(),
        )
        _module("sglang.srt.runtime_context", get_observability=MagicMock())
        _module(
            "sglang.srt.mem_cache.utils",
            get_eviction_strategy=MagicMock(),
            get_hash_str=MagicMock(),
            split_node_hash_value=MagicMock(),
        )
        component = importlib.import_module(
            "sglang.srt.mem_cache.unified_cache.components.tree_component"
        )
        package = sys.modules["sglang.srt.mem_cache.unified_cache.components"]
        package.CacheTransferPhase = component.CacheTransferPhase
        package.ComponentType = component.ComponentType
        pipeline = importlib.import_module("sglang.srt.mem_cache.buffer_mode.pipeline")
        storage = importlib.import_module("sglang.srt.mem_cache.hicache_storage")
        radix = importlib.import_module("sglang.srt.mem_cache.radix_cache")
        tree = importlib.import_module(
            "sglang.srt.mem_cache.unified_cache.unified_tree_core_interface"
        )
        register = importlib.import_module("sglang.test.ci.ci_register")
        # test_utils also imports server/eval utilities and probes the device.
        _module("sglang.benchmark.serving", run_benchmark=MagicMock())
        _module("sglang.lang.global_config", global_config=SimpleNamespace())
        _module("sglang.test.run_eval", run_eval=MagicMock())
        _module("sglang.utils", normalize_base_url=lambda url: url)
        _module(
            "sglang.srt.utils",
            get_bool_env_var=lambda *args: False,
            get_device=lambda: "cpu",
            is_blackwell=lambda: False,
            is_cuda=lambda: False,
            is_xpu=lambda: False,
            kill_process_tree=MagicMock(),
            retry=lambda fn, **kwargs: fn(),
        ).__path__ = []
        _module("sglang.srt.utils.network", is_port_available=MagicMock())
        utilities = importlib.import_module("sglang.test.test_utils")
        return SimpleNamespace(
            pipeline=pipeline,
            storage=storage,
            radix=radix,
            tree=tree,
            component=component,
            custom_test_case=utilities.CustomTestCase,
            register_cpu_ci=register.register_cpu_ci,
        )


_modules = _cpu_modules()
BufferModePipeline = _modules.pipeline.BufferModePipeline
_UnifiedBackupIntent = _modules.pipeline._UnifiedBackupIntent
validate_buffer_only_stack = _modules.pipeline.validate_buffer_only_stack
PoolHitPolicy = _modules.storage.PoolHitPolicy
PoolName = _modules.storage.PoolName
PoolTransfer = _modules.storage.PoolTransfer
SidecarPoolSpec = _modules.storage.SidecarPoolSpec
RadixKey = _modules.radix.RadixKey
ComponentType = _modules.component.ComponentType
BufferBackupSnapshot = _modules.tree.BufferBackupSnapshot
CustomTestCase = _modules.custom_test_case
register_cpu_ci = _modules.register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestBufferModeSidecar(CustomTestCase):
    @staticmethod
    def _swa_component():
        return SimpleNamespace(
            full_window_pages=2,
            _swa_kv_pool_host=SimpleNamespace(page_size=2, size=8),
        )

    @staticmethod
    def _dsv4_specs():
        return [
            SidecarPoolSpec(pool_name=name, indices_from_pool=PoolName.KV)
            for name in (
                PoolName.DEEPSEEK_V4_C4,
                PoolName.DEEPSEEK_V4_C4_INDEXER,
                PoolName.DEEPSEEK_V4_C128,
            )
        ] + [
            SidecarPoolSpec(
                pool_name=name,
                indices_from_pool=PoolName.SWA,
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
            for name in (
                PoolName.DEEPSEEK_V4_C4_STATE,
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
                PoolName.DEEPSEEK_V4_C128_STATE,
            )
        ]

    @classmethod
    def _pool_group(cls, kv_size, swa_size, *, override_size=None):
        sizes = {PoolName.KV: kv_size, PoolName.SWA: swa_size}
        for spec in cls._dsv4_specs():
            sizes[spec.pool_name] = sizes[spec.indices_from_pool]
        sizes.update(override_size or {})
        return SimpleNamespace(
            entry_map={
                name: SimpleNamespace(host_pool=SimpleNamespace(logical_size=size))
                for name, size in sizes.items()
            }
        )

    def test_stack_accepts_dsv4_full_and_swa_sidecars(self):
        """V4 sidecars can borrow valid FULL and SWA staging namespaces."""
        validate_buffer_only_stack(
            sidecar_pool_specs=self._dsv4_specs(),
            host_pool_group=self._pool_group(kv_size=16, swa_size=8),
            swa_component=self._swa_component(),
        )

    def test_stack_rejects_sidecar_smaller_than_source(self):
        """Borrowed indices must not address beyond the sidecar's logical pool."""
        for spec in self._dsv4_specs():
            with self.subTest(pool=spec.pool_name):
                with self.assertRaisesRegex(
                    ValueError, "smaller than its index source"
                ):
                    validate_buffer_only_stack(
                        sidecar_pool_specs=[spec],
                        host_pool_group=self._pool_group(
                            kv_size=16,
                            swa_size=8,
                            override_size={spec.pool_name: 4},
                        ),
                        swa_component=self._swa_component(),
                    )

    def test_stack_preserves_mapping_and_swa_capacity_guards(self):
        """Enabling sidecars must not accept missing owners or unusable SWA staging."""
        for spec in self._dsv4_specs():
            for missing in (spec.pool_name, spec.indices_from_pool):
                with self.subTest(pool=spec.pool_name, missing=missing):
                    group = self._pool_group(kv_size=16, swa_size=8)
                    del group.entry_map[missing]
                    with self.assertRaisesRegex(ValueError, "mapping is incomplete"):
                        validate_buffer_only_stack(
                            sidecar_pool_specs=[spec],
                            host_pool_group=group,
                            swa_component=self._swa_component(),
                        )
        for size in (None, 7):
            with self.subTest(swa_size=size):
                swa = self._swa_component()
                if size is None:
                    swa._swa_kv_pool_host = None
                else:
                    swa._swa_kv_pool_host.size = size
                with self.assertRaisesRegex(ValueError, "requires an SWA"):
                    validate_buffer_only_stack(
                        sidecar_pool_specs=[],
                        host_pool_group=self._pool_group(kv_size=16, swa_size=8),
                        swa_component=swa,
                    )

    def test_write_stages_and_persists_dsv4_full_and_swa_sidecars(self):
        """Backup must persist every sidecar under its source keys and free slots once."""
        page_size = 2
        device_indices = torch.arange(4, dtype=torch.int64, device="cpu")
        host_indices = torch.arange(10, 14, dtype=torch.int64, device="cpu")
        swa_device_indices = torch.arange(30, 32, dtype=torch.int64, device="cpu")
        swa_host_indices = torch.arange(20, 22, dtype=torch.int64, device="cpu")
        swa = PoolTransfer(
            name=PoolName.SWA,
            device_indices=swa_device_indices,
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        sidecars = [
            PoolTransfer(
                name=spec.pool_name,
                hit_policy=spec.hit_policy,
                indices_from_pool=spec.indices_from_pool,
            )
            for spec in self._dsv4_specs()
        ]
        controller = MagicMock()
        controller.storage_backend_type = "mooncake"
        controller.mem_pool_host.entry_map = {
            name: SimpleNamespace(
                host_pool=SimpleNamespace(page_size=page_size, free=MagicMock())
            )
            for name in (PoolName.SWA, *(spec.pool_name for spec in self._dsv4_specs()))
        }

        def write(device_value, *, node_id, extra_pools):
            self.assertEqual(node_id, 7)
            self.assertEqual(
                [transfer.name for transfer in extra_pools],
                [PoolName.SWA, *[transfer.name for transfer in sidecars]],
            )
            swa.host_indices = swa_host_indices
            for sidecar in sidecars:
                if sidecar.indices_from_pool == PoolName.KV:
                    sidecar.host_indices = host_indices
                    sidecar.device_indices = device_value
                else:
                    sidecar.host_indices = swa_host_indices
                    sidecar.device_indices = swa_device_indices
            return host_indices

        controller.write.side_effect = write
        controller.write_storage.return_value = 99
        cache = MagicMock()
        cache.cache_controller = controller
        cache.page_size = page_size
        cache._build_backup_sidecar.return_value = sidecars
        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = cache
        pipeline.ongoing_write_through = {}
        pipeline.ongoing_backup = {}
        pipeline.inflight_backup_hashes = {}
        pipeline.inflight_backup_node_ids = {7}
        pipeline.write_staged_tokens_ = 0
        pipeline.write_backlog_tokens_ = len(device_indices)
        hashes = ["page-0", "page-1"]
        snapshot = BufferBackupSnapshot(
            node_id=7,
            parent_node_id=0,
            parent_is_root=True,
            parent_last_hash=None,
            hash_values=hashes,
            key=RadixKey(array("q", [1, 2, 3, 4])),
            prefix_keys=None,
        )
        intent = _UnifiedBackupIntent(snapshot=snapshot)
        self.assertTrue(
            pipeline._launch_backup_intent(
                intent,
                device_indices,
                comp_xfers={ComponentType.SWA: [swa]},
            )
        )
        self.assertEqual(pipeline.ongoing_write_through[7].aux_xfers, [swa, *sidecars])
        pipeline.finish_backup_ack(7)
        storage_transfers = {
            transfer.name: transfer
            for transfer in controller.write_storage.call_args.kwargs["extra_pools"]
        }
        self.assertEqual(storage_transfers[PoolName.SWA].keys, [hashes[-1]])
        self.assertTrue(
            torch.equal(storage_transfers[PoolName.SWA].host_indices, swa_host_indices)
        )
        for spec in self._dsv4_specs():
            sidecar = storage_transfers[spec.pool_name]
            expected_keys = (
                hashes if spec.indices_from_pool == PoolName.KV else [hashes[-1]]
            )
            self.assertEqual(sidecar.keys, expected_keys)
            self.assertEqual(sidecar.hit_policy, spec.hit_policy)
            self.assertEqual(sidecar.indices_from_pool, spec.indices_from_pool)
            self.assertIsNone(sidecar.host_indices)
        self.assertIn(99, pipeline.ongoing_backup)
        controller.mem_pool_host.free.assert_not_called()
        pipeline.finish_storage_write_ack(99)
        controller.mem_pool_host.free.assert_called_once()
        self.assertTrue(
            torch.equal(controller.mem_pool_host.free.call_args.args[0], host_indices)
        )
        swa_free = controller.mem_pool_host.entry_map[PoolName.SWA].host_pool.free
        swa_free.assert_called_once()
        self.assertTrue(torch.equal(swa_free.call_args.args[0], swa_host_indices))
        for spec in self._dsv4_specs():
            controller.mem_pool_host.entry_map[
                spec.pool_name
            ].host_pool.free.assert_not_called()
        self.assertEqual(pipeline.write_staged_tokens_, 0)
        self.assertFalse(pipeline.inflight_backup_hashes)
        self.assertFalse(pipeline.inflight_backup_node_ids)
        self.assertFalse(pipeline.ongoing_backup)

    def test_completed_prefetch_keeps_dsv4_full_and_swa_sidecars_for_h2d(self):
        """Storage-fetched sidecars must survive staging until H2D source resolution."""
        swa = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.arange(20, 22, dtype=torch.int64, device="cpu"),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        sidecars = [
            PoolTransfer(
                name=spec.pool_name,
                hit_policy=spec.hit_policy,
                indices_from_pool=spec.indices_from_pool,
            )
            for spec in self._dsv4_specs()
        ]
        operation = SimpleNamespace(id=23, pool_transfers=sidecars)
        host_indices = torch.arange(4, dtype=torch.int64, device="cpu")
        req_id = "sidecar-prefetch"
        cache = MagicMock()
        cache.page_size = 2
        cache.cache_controller.prefetch_tokens_occupied = len(host_indices)
        cache.ongoing_prefetch = {
            req_id: (
                0,
                RadixKey(array("q", [1, 2, 3, 4])),
                host_indices,
                operation,
                None,
                {ComponentType.SWA: [swa]},
            )
        }
        cache.prefetch_loaded_tokens_by_reqid = {}
        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = cache
        pipeline._prefetch_prefix_ctx = {req_id: ([], None, None)}
        pipeline.staged_prefetches = {}
        self.assertTrue(
            pipeline.stage_completed_prefetch(
                req_id=req_id,
                num_tokens=len(host_indices),
                hash_value=["page-0", "page-1"],
            )
        )
        staged = pipeline.staged_prefetches[req_id]
        self.assertEqual(staged.aux_xfers, [swa, *sidecars])
        self.assertEqual(staged.num_tokens, len(host_indices))
        self.assertEqual(
            cache.prefetch_loaded_tokens_by_reqid[req_id], len(host_indices)
        )


if __name__ == "__main__":
    unittest.main()
