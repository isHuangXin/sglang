"""The readonly selector must survive HTTP and IPC without invoking native progress."""

import ast
import asyncio
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, get_type_hints
from unittest.mock import AsyncMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SRT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def _node(relative, name, *, owner=None):
    tree = ast.parse((_SRT / relative).read_text())
    if owner is not None:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner
        )
    return next(
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and n.name == name
    )


def _compile(node, scope):
    module = ast.Module(body=[node], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), "<readonly-control-path>", "exec"),
        scope,
    )
    return scope[node.name]


class TestHiCacheReadonlyRouting(CustomTestCase):
    def test_http_selector_is_validated_and_forwarded_with_default_unchanged(self):
        manager = SimpleNamespace(
            get_internal_state=AsyncMock(return_value=[{"hicache_io": {"ranks": []}}]),
            server_args=SimpleNamespace(resolved_dict=lambda: {}),
            startup_time=0,
        )
        routes = {}

        def get(path):
            def decorate(handler):
                routes[path] = handler
                return handler

            return decorate

        scope = dict(
            app=SimpleNamespace(get=get),
            Optional=Optional,
            Literal=Literal,
            _global_state=SimpleNamespace(tokenizer_manager=manager, scheduler_info={}),
            msgspec_to_builtins=lambda value: value,
            describe_kv_events_publisher=lambda _: None,
            __version__="test",
        )
        handler = _compile(_node("entrypoints/http_server.py", "server_info"), scope)
        self.assertIs(routes["/server_info"], handler)
        self.assertEqual(
            get_type_hints(handler)["hicache_io_mode"], Optional[Literal["readonly"]]
        )
        self.assertIsNone(
            inspect.signature(handler).parameters["hicache_io_mode"].default
        )
        for mode in (None, "readonly"):
            result = asyncio.run(
                handler() if mode is None else handler(hicache_io_mode=mode)
            )
            manager.get_internal_state.assert_awaited_with(hicache_io_mode=mode)
            self.assertEqual(result["internal_states"], [{"hicache_io": {"ranks": []}}])

    def test_tokenizer_forwards_selector_and_rejects_unknown_before_ipc(self):
        class BaseReq:
            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__()

            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        scope = dict(BaseReq=BaseReq, Optional=Optional, Literal=Literal)
        request_type = _compile(
            _node("managers/io_struct.py", "GetInternalStateReq"), scope
        )
        self.assertEqual(
            get_type_hints(request_type)["hicache_io_mode"],
            Optional[Literal["readonly"]],
        )
        self.assertIsNone(request_type().hicache_io_mode)
        scope.update(TokenizerManager=object, List=List, Dict=Dict, Any=Any)
        handler = _compile(
            _node(
                "managers/tokenizer_control_mixin.py",
                "get_internal_state",
                owner="TokenizerControlMixin",
            ),
            scope,
        )
        manager = SimpleNamespace(
            auto_create_handle_loop=Mock(),
            get_internal_state_communicator=AsyncMock(
                return_value=[SimpleNamespace(internal_state={"rank": 0})]
            ),
        )
        for mode in (None, "readonly"):
            result = asyncio.run(handler(manager, hicache_io_mode=mode))
            request = manager.get_internal_state_communicator.call_args.args[0]
            self.assertIsInstance(request, request_type)
            self.assertEqual(request.hicache_io_mode, mode)
            self.assertEqual(result, [{"rank": 0}])
        manager.auto_create_handle_loop.reset_mock()
        manager.get_internal_state_communicator.reset_mock()
        for invalid in ("", "native", "drain", "READONLY", True):
            with self.subTest(mode=invalid), self.assertRaises(ValueError):
                asyncio.run(handler(manager, hicache_io_mode=invalid))
        manager.auto_create_handle_loop.assert_not_called()
        manager.get_internal_state_communicator.assert_not_called()

    def test_scheduler_readonly_branch_excludes_native_exporter_and_idle_poll(self):
        # Execute the production selection block without importing scheduler's GPU stack.
        method = _node("managers/scheduler.py", "get_internal_state", owner="Scheduler")
        selection = next(
            node
            for node in method.body
            if isinstance(node, ast.If)
            and any(
                isinstance(value, ast.Attribute) and value.attr == "hicache_io_mode"
                for value in ast.walk(node.test)
            )
        )
        readonly = Mock(
            return_value={
                "schema_version": 1,
                "scope": "completed_accounted_window",
                "drain": False,
            }
        )
        native = Mock(return_value={"ranks": [{"generation": 3}]})
        cache = SimpleNamespace(cache_controller=object())
        scheduler = SimpleNamespace(
            tree_cache=cache,
            ps=SimpleNamespace(
                tp_rank=2,
                tp_size=8,
                pp_rank=0,
                pp_size=1,
                dp_rank=0,
                dp_size=1,
                attn_cp_size=1,
                attn_dcp_size=1,
            ),
            tp_cpu_group=object(),
            enable_hierarchical_cache=True,
            is_fully_idle=Mock(
                side_effect=AssertionError("readonly must not poll idle")
            ),
        )
        code = compile(
            ast.Module(body=[selection], type_ignores=[]),
            "<scheduler-selector>",
            "exec",
        )
        modules = {
            "sglang.srt.observability.hicache_io": SimpleNamespace(
                collect_hicache_io=readonly
            ),
            "sglang.srt.managers.hicache_io_state": SimpleNamespace(
                collect_hicache_io_state=native
            ),
        }
        with patch.dict("sys.modules", modules):
            scope = dict(
                self=scheduler,
                recv_req=SimpleNamespace(hicache_io_mode="readonly"),
                ret={},
            )
            exec(code, scope)
            self.assertEqual(
                scope["ret"]["hicache_io"]["scope"], "completed_accounted_window"
            )
            native.assert_not_called()
            scheduler.is_fully_idle.assert_not_called()
            readonly.assert_called_once_with(
                cache=cache,
                tp_rank=2,
                tp_size=8,
                pp_rank=0,
                pp_size=1,
                dp_rank=0,
                dp_size=1,
                attn_cp_size=1,
                attn_dcp_size=1,
                tp_cpu_group=scheduler.tp_cpu_group,
            )
            readonly.reset_mock()
            scheduler.is_fully_idle.side_effect = None
            scheduler.is_fully_idle.return_value = True
            scope = dict(
                self=scheduler, recv_req=SimpleNamespace(hicache_io_mode=None), ret={}
            )
            exec(code, scope)
            self.assertEqual(scope["ret"]["hicache_io"], {"ranks": [{"generation": 3}]})
            native.assert_called_once_with(cache=cache, idle=True)
            scheduler.is_fully_idle.assert_called_once_with(include_storage=False)
            readonly.assert_not_called()


if __name__ == "__main__":
    unittest.main()
