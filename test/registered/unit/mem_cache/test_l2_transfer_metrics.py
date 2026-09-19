"""Physical L2 metering follows resolved descriptors and the actual layer schedule."""

import ast
import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, NamedTuple, Optional
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_SRT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def _definitions(relative, names, scope, *, owner=None):
    tree = ast.parse((_SRT / relative).read_text())
    if owner is not None:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner
        )
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(_SRT / relative), "exec"), scope
    )


class Indices:
    is_cuda = False

    def __init__(self, size):
        self.size = size

    def numel(self):
        return self.size

    def __len__(self):
        return self.size


class TestL2TransferMetrics(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.device_module = SimpleNamespace(
            Stream=Mock(),
            Event=Mock(side_effect=lambda **_: Mock()),
            stream=lambda _: contextlib.nullcontext(),
        )
        self.scope = dict(
            Any=Any,
            Callable=Callable,
            NamedTuple=NamedTuple,
            Optional=Optional,
            device_module=self.device_module,
        )
        _definitions(
            "mem_cache/l2_transfer.py",
            (
                "L2Transfer",
                "TransferCompletion",
                "L2TransferEngine",
                "make_timing_event_pair",
            ),
            self.scope,
        )
        self.scope["_timing_events_supported"] = lambda: True
        self.transfer_type = self.scope["L2Transfer"]
        self.pools = {}
        for name in (
            "LogicalHostPool",
            "DeepSeekV4PagedHostPool",
            "DeepSeekV4StateHostPool",
            "MHATokenToKVPoolHost",
            "MLATokenToKVPoolHost",
            "DSAIndexerPoolHost",
        ):
            self.pools[name] = type(name, (), {})
        self.scope.update(self.pools)
        ownership_methods = (
            "_is_device_layer_owned",
            "_device_owned_layer_range",
            "_is_device_layer_sharded",
        )
        _definitions(
            "mem_cache/pool_host/base.py",
            ownership_methods,
            self.scope,
            owner="HostKVCache",
        )
        for pool in self.pools.values():
            for name in ownership_methods:
                setattr(pool, name, self.scope[name])
        _definitions(
            "mem_cache/l2_transfer_metrics.py",
            ("_bytes_per_layer", "l2_transfer_num_bytes"),
            self.scope,
        )
        self.meter = self.scope["l2_transfer_num_bytes"]

    def pool(
        self,
        kind="DeepSeekV4PagedHostPool",
        *,
        layers=5,
        page_size=256,
        item_bytes=37440,
        layout="layer_first",
    ):
        pool = self.pools[kind]()
        pool.layer_num = layers
        pool.slot_page_size = pool.swa_page_size = pool.page_size = page_size
        pool.item_bytes = pool.state_page_bytes = item_bytes
        pool.dtype = SimpleNamespace(itemsize=1)
        pool.size_per_token = layers * item_bytes
        pool.layout = layout
        pool.dcp_size = 1
        pool.device_pool = SimpleNamespace(layer_num=layers, layer_shard_enabled=False)
        pool.backup_from_device_all_layer = Mock()
        pool.load_to_device_per_layer = Mock()
        return pool

    def transfer(self, pool, slots, *, mapper=None):
        indices = Indices(slots)
        return self.transfer_type(pool, pool.device_pool, indices, indices, mapper)

    def test_v4_page_rows_and_partial_c4_payload_not_token_estimates(self):
        for layout in ("layer_first", "page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                pool = self.pool(layout=layout)
                self.assertEqual(
                    self.meter([self.transfer(pool, 512)], io_backend="kernel"),
                    2 * 5 * 37440,
                )
        for slots in (1, 63, 257):
            with self.subTest(slots=slots):
                xfer = self.transfer(self.pool(), slots, mapper={0: 1, 2: 4}.get)
                self.assertEqual(
                    self.meter([xfer], io_backend="kernel"), slots * 584 * 5
                )
                self.assertEqual(
                    self.meter([xfer], io_backend="kernel", layer_num=4),
                    slots * 584 * 2,
                )

    def test_v4_state_includes_dtype_width_and_requires_whole_rows(self):
        pool = self.pool(
            "DeepSeekV4StateHostPool", layers=3, page_size=128, item_bytes=147456
        )
        pool.dtype.itemsize = 2
        self.assertEqual(
            self.meter([self.transfer(pool, 256)], io_backend="kernel"),
            2 * 3 * 147456 * 2,
        )
        self.assertIsNone(self.meter([self.transfer(pool, 255)], io_backend="kernel"))

    def test_mha_mla_and_indexer_load_only_owned_layers_with_packed_draft(self):
        for kind, bytes_per_token in (
            ("MHATokenToKVPoolHost", 1024),
            ("MLATokenToKVPoolHost", 656),
            ("DSAIndexerPoolHost", 132),
        ):
            with self.subTest(kind=kind):
                pool = self.pool(
                    kind, layers=4, page_size=64, item_bytes=bytes_per_token
                )
                pool.indexer_page_stride_size = 64 * bytes_per_token
                pool.device_pool.layer_num = 3
                xfer = self.transfer(pool, 128)
                self.assertEqual(
                    self.meter([xfer], io_backend="direct"), 128 * 4 * bytes_per_token
                )
                self.assertEqual(
                    self.meter([xfer], io_backend="direct", layer_num=4),
                    128 * 3 * bytes_per_token,
                )
                draft = xfer._replace(layer_mapper={0: 3}.get, is_draft=True)
                self.assertEqual(
                    self.meter([xfer, draft], io_backend="direct", layer_num=4),
                    128 * 4 * bytes_per_token,
                )

    def test_meter_and_engine_share_sidecar_layer_selection(self):
        anchor = self.transfer(self.pool("LogicalHostPool"), 512)
        swa = self.transfer(
            self.pool(page_size=128, item_bytes=75264),
            256,
            mapper={i: i for i in range(5)}.get,
        )
        c4 = self.transfer(self.pool(layers=2), 512, mapper={1: 0, 3: 1}.get)
        state = self.transfer(
            self.pool(
                "DeepSeekV4StateHostPool", layers=2, page_size=128, item_bytes=147456
            ),
            256,
            mapper={1: 0, 3: 1}.get,
        )
        draft = swa._replace(layer_mapper={0: 4}.get, is_draft=True)
        transfers = [anchor, swa, c4, state, draft]
        expected = 2 * 5 * 75264 + 2 * 2 * 37440 + 2 * 2 * 147456
        self.assertEqual(
            self.meter(transfers, io_backend="kernel", layer_num=4), expected
        )
        engine = self.scope["L2TransferEngine"]("kernel")
        completion = engine.submit_host_to_device(
            transfers, layer_num=4, on_layer_done=Mock()
        )
        self.assertEqual(swa.host_pool.load_to_device_per_layer.call_count, 5)
        self.assertEqual(c4.host_pool.load_to_device_per_layer.call_count, 2)
        self.assertEqual(state.host_pool.load_to_device_per_layer.call_count, 2)
        completion.start_event.record.assert_called_once()
        completion.finish_event.record.assert_called_once()
        second = engine.submit_host_to_device(
            transfers, layer_num=4, on_layer_done=Mock()
        )
        self.assertIsNot(completion.finish_event, second.finish_event)
        secondary = self.transfer(self.pool(layers=2), 256)
        self.assertEqual(
            self.meter([swa, secondary], io_backend="kernel", layer_num=4),
            2 * 4 * 75264 + 2 * 37440,
        )

    def test_hybrid_resolves_sidecars_and_draft_before_payload_metering(self):
        relative = "mem_cache/hybrid_cache/hybrid_cache_controller.py"
        names = ("_l2_transfers", "_l2_load_transfers")
        _definitions(relative, names, self.scope, owner="HybridCacheController")
        controller_type = type(
            "HybridController", (), {name: self.scope[name] for name in names}
        )
        controller = controller_type()
        anchor = self.pool("LogicalHostPool")
        swa = self.pool(page_size=128, item_bytes=75264)
        anchor_entry = SimpleNamespace(
            host_pool=anchor,
            device_pool=anchor.device_pool,
            layer_mapper=None,
            packed_draft_device_pools=(),
        )
        swa_entry = SimpleNamespace(
            host_pool=swa,
            device_pool=swa.device_pool,
            layer_mapper={i: i for i in range(5)}.get,
            packed_draft_device_pools=(object(),),
        )
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=anchor_entry,
            entry_map={"kv": anchor_entry, "swa": swa_entry},
        )
        controller.layer_num = 4
        indices = Indices(256)
        sidecars = [
            SimpleNamespace(name="swa", host_indices=indices, device_indices=indices)
        ]
        writes = controller._l2_transfers(Indices(512), Indices(512), sidecars)
        loads = controller._l2_load_transfers(Indices(512), Indices(512), sidecars)
        self.assertEqual(len(loads), len(writes) + 1)
        self.assertTrue(loads[-1].is_draft)
        self.assertEqual(self.meter(writes, io_backend="kernel"), 2 * 5 * 75264)
        self.assertEqual(
            self.meter(loads, io_backend="kernel", layer_num=4), 2 * 5 * 75264
        )

    def test_unknown_mismatched_and_unprovable_descriptors_are_not_zero(self):
        pool = self.pool()
        xfer = self.transfer(pool, 256)
        unknown = xfer._replace(host_pool=object())
        mismatched = xfer._replace(device_indices=Indices(255))
        invalid_layer = xfer._replace(layer_mapper=lambda _: 5)
        for value in (unknown, mismatched, invalid_layer):
            self.assertIsNone(self.meter([value], io_backend="kernel", layer_num=1))
        self.assertIsNone(self.meter([xfer], io_backend="kernel_ascend"))
        mla = self.pool("MLATokenToKVPoolHost", page_size=64)
        self.assertIsNone(self.meter([self.transfer(mla, 63)], io_backend="direct"))
        mla.dcp_size = 2
        self.assertIsNone(self.meter([self.transfer(mla, 64)], io_backend="kernel"))
        mla.dcp_size = 1
        mla.device_pool.layer_shard_enabled = True
        self.assertIsNone(self.meter([self.transfer(mla, 64)], io_backend="kernel"))

    def test_controller_records_the_same_resolved_payload_in_native_and_ack_paths(self):
        self.scope.update(List=List)
        _definitions("managers/cache_controller.py", ("HiCacheAck",), self.scope)
        names = ("start_writing", "start_loading", "_transfer_num_bytes")
        _definitions(
            "managers/cache_controller.py", names, self.scope, owner="HiCacheController"
        )
        controller_type = type(
            "Controller", (), {name: self.scope[name] for name in names}
        )
        for unknown in (False, True):
            for direction in ("read", "write"):
                with self.subTest(unknown=unknown, direction=direction):
                    controller = controller_type()
                    controller.io_backend = "kernel"
                    controller.layer_num = 4
                    xfer = self.transfer(self.pool(), 512, mapper={0: 1, 2: 4}.get)
                    if unknown:
                        xfer = xfer._replace(host_pool=object())
                    op = SimpleNamespace(node_ids=[1], device_indices=Indices(512))
                    self.scope["CacheOperation"] = SimpleNamespace(
                        merge_ops=lambda _: op
                    )
                    controller.enable_storage = False
                    controller.write_queue = [op]
                    controller.load_queue = [op]
                    controller.ack_write_queue = []
                    controller.ack_load_queue = []
                    controller._move_write_operation = lambda _: (None, None, None)
                    controller._move_op_indices = controller._move_write_operation
                    controller._l2_transfers = lambda *_: [xfer]
                    controller._l2_load_transfers = controller._l2_transfers
                    controller._num_tokens_by_pool = lambda _: {}
                    controller.l2_transfer_engine = Mock()
                    completion = SimpleNamespace(
                        start_event=Mock(), finish_event=Mock(), timing_enabled=True
                    )
                    engine = controller.l2_transfer_engine
                    engine.submit_device_to_host.return_value = completion
                    engine.submit_host_to_device.return_value = completion
                    controller._host_io_metrics = Mock()
                    controller.load_fence_stream = None
                    controller.layer_done_counter = SimpleNamespace(
                        update_producer=lambda: 0, events=[Mock()]
                    )
                    if direction == "read":
                        controller.start_loading()
                        item = controller.ack_load_queue[0]
                    else:
                        controller.start_writing()
                        item = controller.ack_write_queue[0]
                    expected = (
                        None
                        if unknown
                        else 2 * 37440 * (2 if direction == "read" else 5)
                    )
                    self.assertEqual(item.num_bytes, expected)
                    controller._host_io_metrics.record.assert_called_once_with(
                        direction=direction, completion=completion, num_bytes=expected
                    )
                    self.assertIs(item.start_event, completion.start_event)
                    self.assertIs(item.finish_event, completion.finish_event)


if __name__ == "__main__":
    unittest.main()
