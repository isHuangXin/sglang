"""Serving reports retain cumulative metadata and exclude collector control time."""

import __future__
import ast
import asyncio
import contextlib
import dataclasses
import io
import json
import os
import sys
import time
import traceback
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, mock_open, patch

import requests

from sglang.benchmark import flat_memory_metrics as flat_metrics
from sglang.benchmark.flat_memory_report import (
    REPORT_FIELD_NAMES,
    format_cache_reports,
    project_cache_reports,
    resolve_metrics_mode,
)
from sglang.benchmark.native_io_metrics import sanitize_server_info
from sglang.benchmark.tiered_cache_metrics import consume_tiered_cache_metadata, summarize_tiered_cache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _serving_namespace():
    # Execute the production functions without importing GPU-serving dependencies.
    path = Path(__file__).resolve().parents[4] / "python/sglang/benchmark/serving.py"
    names = {
        "RequestFuncInput",
        "RequestFuncOutput",
        "BenchmarkMetrics",
        "benchmark",
        "_extract_cache_from_sglext",
        "_response_chunks",
        "_fetch_cache_server_info",
        "_combine_openai_chat_content",
        "_validate_cache_io_options",
        "async_request_sglang_generate",
        "async_request_openai_completions",
        "async_request_openai_chat_completions",
    }
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    namespace = dict(
        __name__=__name__,
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
        asyncio=asyncio,
        json=json,
        orjson=json,
        time=time,
        sys=sys,
        traceback=traceback,
        os=os,
        warnings=warnings,
        requests=requests,
        consume_native_usage=flat_metrics.consume_native_usage,
        consume_tiered_cache_metadata=consume_tiered_cache_metadata,
        summarize_tiered_cache=summarize_tiered_cache,
        sanitize_server_info=sanitize_server_info,
        summarize_cache_usage=flat_metrics.summarize_cache_usage,
        summarize_flat_cache=flat_metrics.summarize_flat_cache,
        project_cache_reports=project_cache_reports,
        format_cache_reports=format_cache_reports,
        resolve_metrics_mode=resolve_metrics_mode,
        FlatMemoryIOWindow=flat_metrics.FlatMemoryIOWindow,
        FLAT_DETAIL_FIELDS=flat_metrics.FLAT_DETAIL_FIELDS,
        get_auth_headers=lambda: {},
        get_request_headers=lambda: {},
        remove_prefix=lambda value, prefix: value.removeprefix(prefix),
        _DEFAULT_SGLANG_FLUSH_CACHE_TIMEOUT=150.0,
        _EMBEDDING_BACKENDS=set(),
    )
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


def _options(**overrides):
    options = dict(
        backend="sglang",
        cache_report=False,
        disable_stream=False,
        disable_ignore_eos=False,
        return_logprob=False,
        return_routed_experts=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        logprob_start_len=-1,
        temperature=0,
        top_p=1,
        print_requests=False,
        collect_flat_memory_io=True,
        flat_memory_tp_size=1,
        collect_hicache_io=False,
        collect_hicache_io_metrics=False,
        collect_mooncake_io_metrics=False,
        mooncake_owner_metrics_url=None,
        mooncake_master_host=None,
        mooncake_metrics_port=9003,
        mooncake_client_host="localhost",
        mooncake_client_metrics_port=9301,
        dataset_name="random",
        warmup_requests=0,
        plot_throughput=False,
        output_file="mock.jsonl",
        output_details=True,
        sharegpt_output_len=2,
        random_input_len=500,
        random_output_len=2,
        random_range_ratio=1,
    )
    return SimpleNamespace(**(options | overrides))


class TestServingCacheReports(CustomTestCase):
    def test_cumulative_metadata_only_and_nonstream_usage(self):
        """Missing text/choices must not discard final usage or depend on cache-report."""
        ns = _serving_namespace()
        details = {"device": 80, "producer_extension": {"kept": True}}
        for name in (
            "sglang_generate",
            "openai_completions",
            "openai_chat_completions",
        ):
            for stream in (False, True):
                with self.subTest(handler=name, stream=stream):
                    ns["args"] = _options(disable_stream=not stream)
                    usage = {
                        "prompt_tokens": 100,
                        "completion_tokens": 2,
                        "cached_tokens": 80,
                        "cached_tokens_details": details,
                    }
                    if name == "sglang_generate":
                        body = {"text": "answer", "meta_info": usage}
                        chunks = [body, {"meta_info": usage}]
                    else:
                        choice = {
                            "text": "answer",
                            "delta": {"content": "answer"},
                            "message": {"content": "answer"},
                        }
                        body = {
                            "choices": [choice],
                            "usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 2,
                                "prompt_tokens_details": {"cached_tokens": 80},
                            },
                            "sglext": {"cached_tokens_details": details},
                        }
                        chunks = [body, {**body, "choices": []}]
                    response = MagicMock(status=200)
                    response.__aenter__ = AsyncMock(return_value=response)
                    response.read = AsyncMock(
                        return_value=json.dumps(body, indent=2).encode()
                    )
                    response.json = AsyncMock(return_value=body)
                    response.content.__aiter__.return_value = [
                        b"data: " + json.dumps(chunk).encode() for chunk in chunks
                    ] + [b"data: [DONE]"]
                    session = MagicMock()
                    session.__aenter__ = AsyncMock(return_value=session)
                    session.post.return_value = response
                    ns["_create_bench_client_session"] = lambda: session
                    request = ns["RequestFuncInput"](
                        prompt="prompt",
                        api_url="http://server/v1/chat/completions",
                        prompt_len=500,
                        output_len=9,
                        model="test",
                        lora_name=None,
                        image_data=None,
                        extra_request_body={},
                    )
                    output = asyncio.run(ns["async_request_" + name](request))
                    self.assertTrue(output.success, output.error)
                    if name != "sglang_generate" and stream:
                        self.assertTrue(
                            session.post.call_args.kwargs["json"]["stream_options"][
                                "include_usage"
                            ]
                        )
                    self.assertEqual(
                        (
                            output.server_prompt_tokens,
                            output.server_cached_tokens,
                            output.output_len,
                        ),
                        (100, 80, 2),
                    )
                    self.assertEqual(output.cached_tokens_details, details)
                    self.assertIsNone(output.cached_tokens_host)
                    self.assertEqual(
                        flat_metrics.summarize_cache_usage([output])["cache_hit_rate"],
                        0.8,
                    )
        self.assertIsNone(ns["RequestFuncOutput"]().cached_tokens)

    def test_tiered_summary_with_successful_and_failed_requests(self):
        """Tiered summaries must survive numeric counters and exclude failed requests."""
        output_type = _serving_namespace()["RequestFuncOutput"]
        output = output_type(success=True)
        meta = {
            "prompt_tokens": 1024,
            "cached_tokens": 768,
            "cached_tokens_details": {
                "device": 256,
                "host": 128,
                "storage": 384,
                "cache_source_mode": "mooncake_tiered_gds",
                "tiered_cache": {
                    "device": 256,
                    "host": 256,
                    "l2_host": 128,
                    "mooncake_dram": 128,
                    "ssd": 256,
                    "mixed": 128,
                },
            },
        }
        flat_metrics.consume_native_usage(output, meta)
        consume_tiered_cache_metadata(output, meta)
        outputs = [output, output_type(success=False, error="Connection refused")]
        summary = summarize_tiered_cache(outputs)
        self.assertEqual(summary["tiered_cache_status"], "ok")
        self.assertEqual(summary["total_prompt_tokens"], 1024)
        self.assertEqual(summary["total_cached_tokens"], 768)
        self.assertEqual(summary["cache_hit_rate"], 0.75)
        for tier in ("device", "host", "storage"):
            self.assertEqual(summary[f"{tier}_hit_rate_pct"], 25.0)
        self.assertEqual(summary["tiered_cached_tokens_mixed"], 128)

        output.server_cached_tokens = None
        summary = summarize_tiered_cache(outputs)
        self.assertEqual(summary["tiered_cache_status"], "unavailable")
        self.assertIsNone(summary["total_cached_tokens"])
        self.assertIsNone(summary["cache_hit_rate"])

    def test_legacy_metrics_parse_without_suppressing_valid_counters(self):
        """Valid legacy metrics must not disappear behind an import-error warning."""
        response = Mock(
            text='master_successful_evictions_total 3\n'
            'sglang:prefetch_bandwidth_sum{tp_rank="0"} 4\n'
            'sglang:prefetch_bandwidth_sum{tp_rank="1"} 6\n'
        )
        patterns = {
            "evictions": "master_successful_evictions_total",
            "prefetch_sum": "sglang:prefetch_bandwidth_sum",
        }
        with patch.object(
            flat_metrics.requests, "get", return_value=response
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(
                flat_metrics.fetch_legacy_metrics("server", 9006, patterns),
                {"evictions": 3.0, "prefetch_sum": 10.0},
            )
            self.assertEqual(caught, [])
            for value in ("NaN", "+Inf", "-1"):
                with self.subTest(value=value):
                    response.text = f"master_successful_evictions_total {value}\n"
                    self.assertEqual(
                        flat_metrics.fetch_legacy_metrics("server", 9006, patterns), {}
                    )

    def _benchmark(self, *, mode, cache_report=False, request_error=False):
        ns = _serving_namespace()
        ns["args"] = _options(cache_report=cache_report)
        clock, events = [0.0], []
        info = {"radix_cache_backend": mode} if mode != "unknown" else {}
        info.update(api_key="private-api-key", secret="private-server-secret")
        ns["_fetch_cache_server_info"] = Mock(return_value=info)
        ns["_get_bool_env_var"] = lambda *_: False
        ns["time"] = SimpleNamespace(
            perf_counter=lambda: clock[0], sleep=lambda _: None
        )
        native = MagicMock(
            enabled=False,
            storage_enabled=False,
            after=None,
            before=None,
            host_metrics={},
            storage_metrics={},
            collect_host=False,
            collect_mooncake=False,
            host_result={},
            mooncake_result={},
            gds_result={},
        )
        native.__aenter__ = AsyncMock(return_value=native)
        native.__aexit__ = AsyncMock(return_value=False)
        ns["NativeIOWindow"] = Mock(return_value=native)
        hicache = MagicMock(enabled=False)
        hicache.__aenter__ = AsyncMock(return_value=hicache)
        hicache.__aexit__ = AsyncMock(return_value=False)
        ns["HiCacheIOCollector"] = Mock(return_value=hicache)
        ns["project_cache_reports"] = Mock(wraps=project_cache_reports)
        details = {
            "device": 40,
            "host": 20,
            "storage": 20,
            "unknown_detail": 7,
            "flat_prefetch_dram_ms": 12,
            "flat_prefetch_dram_ops": 2,
            "flat_prefetch_ssd_ms": 18,
            "flat_prefetch_ssd_ops": 1,
        }
        output = ns["RequestFuncOutput"](success=True, prompt_len=500, output_len=2)
        flat_metrics.consume_native_usage(
            output,
            {
                "prompt_tokens": 100,
                "cached_tokens": 80,
                "cached_tokens_details": details,
            },
        )
        metrics = ns["BenchmarkMetrics"](
            **{field.name: 1 for field in dataclasses.fields(ns["BenchmarkMetrics"])}
        )
        for name, value in flat_metrics.summarize_cache_usage([output]).items():
            setattr(metrics, name, value)
        ns["calculate_metrics"] = lambda **_: (metrics, [2])
        row = SimpleNamespace(
            prompt="prompt",
            prompt_len=500,
            output_len=2,
            image_data=None,
            extra_request_body={},
            timestamp=None,
            routing_key=None,
        )

        async def generate(*_):
            yield row

        async def request(**_):
            events.append("request")
            clock[0] += 2
            if request_error:
                raise RuntimeError("request failed")
            return output

        async def window_request(collector, action=None):
            events.append(action)
            clock[0] += 10
            if action == "end":
                raise TimeoutError("telemetry failed after requests")
            return {}

        ns["get_request"] = generate
        ns["ASYNC_REQUEST_FUNCS"] = {"sglang": request}
        file = mock_open()
        console = io.StringIO()
        with patch.object(
            flat_metrics.FlatMemoryIOWindow, "_wait_idle", new=AsyncMock()
        ), patch.object(
            flat_metrics.FlatMemoryIOWindow, "_request", new=window_request
        ), patch.object(
            flat_metrics, "_validate_flat_window"
        ), patch(
            "builtins.open", file
        ), contextlib.redirect_stdout(
            console
        ), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = asyncio.run(
                ns["benchmark"](
                    backend="sglang",
                    api_url="http://server/generate",
                    base_url="http://server",
                    model_id="test",
                    tokenizer=None,
                    input_requests=[row],
                    request_rate=1,
                    max_concurrency=None,
                    disable_tqdm=True,
                    lora_names=None,
                    lora_request_distribution=None,
                    lora_zipf_alpha=None,
                    extra_request_body={},
                    profile=False,
                    warmup_requests=0,
                )
            )
        dumped = json.loads(file().write.call_args.args[0])
        self.assertEqual(result, dumped)
        new_exports = json.dumps(
            {key: value for key, value in dumped.items() if key != "server_info"}
        )
        self.assertNotIn("private-api-key", new_exports)
        self.assertNotIn("private-server-secret", new_exports)
        ns["project_cache_reports"].assert_called_once()
        self.assertEqual(result["duration"], 2)
        self.assertEqual(result["cached_tokens_details"], [details])
        for heading in (
            "Cache Hit Statistics",
            "Flat Prefetch Arrival-to-All-TP-GPU-Ready",
            "Flat Completed/Durable I/O Window",
        ):
            self.assertEqual(console.getvalue().count(heading), 1)
        return result, console.getvalue(), events

    def test_projection_flag_parity_mode_gating_and_telemetry_failure(self):
        """All output paths share one projection; Flat telemetry errors keep requests."""
        for mode in ("native", "flat_memory", "unknown"):
            with self.subTest(mode=mode):
                result, console, events = self._benchmark(mode=mode)
                flagged, _, _ = self._benchmark(mode=mode, cache_report=True)
                self.assertEqual(
                    {key: result[key] for key in REPORT_FIELD_NAMES},
                    {key: flagged[key] for key in REPORT_FIELD_NAMES},
                )
                self.assertEqual(result["cache_report"], flagged["cache_report"])
                if mode == "native":
                    self.assertEqual(result["cache_hit_rate"], 0.8)
                    self.assertEqual(result["device_hit_rate_pct"], 40)
                    self.assertIsNone(result["flat_prefetch_latency_ms"])
                    self.assertEqual(events, ["request"])
                elif mode == "flat_memory":
                    self.assertIsNone(result["total_prompt_tokens"])
                    self.assertIsNone(result["total_cached_tokens"])
                    self.assertEqual(result["flat_prefetch_latency_ms"], 10)
                    self.assertEqual(events, ["begin", "request", "end", "abort"])
                    self.assertEqual(result["flat_io_status"], "unavailable")
                    self.assertNotIn("Transfer Metrics", console)
                    self.assertNotIn("Flat Memory System Statistics", console)
                else:
                    self.assertEqual(events, ["request"])
                    self.assertTrue(
                        all(result[key] is None for key in REPORT_FIELD_NAMES)
                    )
        with self.assertRaisesRegex(RuntimeError, "request failed"):
            self._benchmark(mode="flat_memory", request_error=True)

    def test_readonly_probe_failure_and_exclusive_options(self):
        ns = _serving_namespace()
        with patch.object(
            requests, "get", side_effect=requests.Timeout
        ), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIsNone(ns["_fetch_cache_server_info"]("http://server"))
            self.assertEqual(requests.get.call_args.kwargs["timeout"], 5)
            self.assertEqual(
                requests.get.call_args.kwargs["params"], {"hicache_io_mode": "readonly"}
            )
        for flag in (
            "collect_hicache_io",
            "collect_hicache_io_metrics",
            "collect_mooncake_io_metrics",
        ):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                ns["_validate_cache_io_options"](_options(**{flag: True}))


if __name__ == "__main__":
    unittest.main()
