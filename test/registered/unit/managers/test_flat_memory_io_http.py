"""Flat I/O control accepts the native client's JSON request body."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated
from unittest.mock import Mock

from fastapi import Body, FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.testclient import TestClient
from sglang.srt.managers.flat_memory_io_window import (
    collect_flat_memory_state,
    flat_io_window_response,
    validate_flat_window_request,
)
from sglang.srt.managers.io_struct import FlatMemoryIOWindowReq
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFlatMemoryIOHTTP(CustomTestCase):
    def setUp(self):
        self.calls = []

        async def get_state(**request):
            self.calls.append(request)
            return [
                {
                    "flat_io_control": {"error": None},
                    "flat_memory": {
                        "tp_size": 1,
                        "ranks": [
                            {
                                "tp_rank": 0,
                                "tp_size": 1,
                                "pid": 12345,
                                "gpu_id": 3,
                                "io_window": {
                                    "window_id": request["flat_io_window_id"],
                                    "enabled": True,
                                    "active": request["flat_io_action"]
                                    in ("begin", "collect"),
                                    "aborted": False,
                                    "overflowed": False,
                                    "io_errors": 0,
                                },
                            }
                        ],
                    },
                }
            ]

        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/entrypoints/http_server.py"
        )
        function = next(
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "flat_memory_io_window"
        )
        # FLAT_MEMORY: Bind the real handler without importing server lifecycle hooks.
        function.decorator_list = []
        namespace = {
            "Annotated": Annotated,
            "Body": Body,
            "FlatMemoryIOWindowReq": FlatMemoryIOWindowReq,
            "ORJSONResponse": ORJSONResponse,
            "validate_flat_window_request": validate_flat_window_request,
            "flat_io_window_response": flat_io_window_response,
            "_global_state": SimpleNamespace(
                tokenizer_manager=SimpleNamespace(get_internal_state=get_state)
            ),
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        self.app = FastAPI()
        self.app.post("/flat_memory/io_window")(namespace["flat_memory_io_window"])
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def test_json_body_reaches_window_control(self):
        """Body fields must not be mistaken for a required query parameter named obj."""
        for action in ("begin", "collect", "end"):
            response = self.client.post(
                "/flat_memory/io_window",
                json={"action": action, "window_id": "test-owned-window"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(
                response.json()["ranks"][0]["io_window"]["active"],
                action in ("begin", "collect"),
            )
        self.assertEqual(
            [call["flat_io_action"] for call in self.calls], ["begin", "collect", "end"]
        )
        operation = self.app.openapi()["paths"]["/flat_memory/io_window"]["post"]
        self.assertIn("application/json", operation["requestBody"]["content"])
        self.assertFalse(
            any(param["name"] == "obj" for param in operation.get("parameters", []))
        )

    def test_invalid_action_does_not_reach_scheduler(self):
        response = self.client.post(
            "/flat_memory/io_window", json={"action": "invalid", "window_id": "test"}
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.calls, [])

    def test_collect_does_not_require_idle_or_drain(self):
        """Live telemetry must not drain requests, and must return the consumed chunk."""
        current = {
            "schema_version": 3,
            "window_id": "owned",
            "active": True,
            "enabled": True,
            "aborted": False,
            "overflowed": False,
            "io_errors": 0,
            "start_ns": 1,
            "end_ns": 0,
            "bucket_begin": 0,
            "bucket_end": 0,
        }
        chunk = {
            **current,
            "bucket_end": 10,
            "buckets": [{"index": 1, "read_bytes": 42}],
        }
        manager = SimpleNamespace(
            get_io_window=Mock(return_value=current),
            io_clock_ns=Mock(return_value=1_000_000_001),
            collect_io_window=Mock(return_value=chunk),
        )
        drain = Mock(side_effect=AssertionError("collection must not drain"))
        state = collect_flat_memory_state(
            manager=manager,
            tp_rank=0,
            tp_size=1,
            gather=lambda value: [value],
            snapshot=lambda: {"pending_backups": 1, "pending_prefetches": 2},
            idle=False,
            drain=drain,
            action="collect",
            window_id="owned",
        )
        response = flat_io_window_response([state], action="collect", window_id="owned")
        self.assertEqual(response["ranks"][0]["io_window"], chunk)
        manager.collect_io_window.assert_called_once_with("owned", 1_000_000_001)
        drain.assert_not_called()


if __name__ == "__main__":
    unittest.main()
