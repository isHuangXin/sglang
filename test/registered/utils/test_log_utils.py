import io
import json
import logging
import re
import tempfile
import threading
import unittest
import uuid
from contextlib import redirect_stdout
from itertools import count
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.srt.utils.log_utils import SlowStageLogger, create_log_targets, log_json
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=4, suite="base-c-test-cpu")

_LOG_PREFIX_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ")


class TestLogUtils(unittest.TestCase):
    def test_stdout(self):
        for targets in [["stdout"], None]:
            with self.subTest(targets=targets):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    loggers = create_log_targets(
                        targets=targets, name_prefix=f"test_stdout_{uuid.uuid4()}"
                    )
                    self.assertEqual(len(loggers), 1)
                    log_json(loggers[0], "test.event", {"key": "value"})
                data = _parse_log_json(buf.getvalue().strip())
                self.assertIn("timestamp", data)
                self.assertEqual(data["event"], "test.event")
                self.assertEqual(data["key"], "value")

    def test_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loggers = create_log_targets(
                targets=[temp_dir], name_prefix=f"test_file_{uuid.uuid4()}"
            )
            self.assertEqual(len(loggers), 1)
            log_json(loggers, "file.event", {"data": 123})
            _flush_all(loggers)
            data = _read_log_file(temp_dir)
            self.assertIn("timestamp", data)
            self.assertEqual(data["event"], "file.event")
            self.assertEqual(data["data"], 123)

    def test_multiple_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            buf = io.StringIO()
            with redirect_stdout(buf):
                loggers = create_log_targets(
                    targets=["stdout", temp_dir],
                    name_prefix=f"test_multi_{uuid.uuid4()}",
                )
                self.assertEqual(len(loggers), 2)
                log_json(loggers, "multi.event", {"x": 1})
            _flush_all(loggers)
            stdout_data = _parse_log_json(buf.getvalue().strip())
            file_data = _read_log_file(temp_dir)
            self.assertEqual(stdout_data["event"], "multi.event")
            self.assertEqual(file_data["event"], "multi.event")
            self.assertEqual(stdout_data["x"], file_data["x"])


class TestSlowStageLogger(CustomTestCase):
    def test_disabled_skips_clocks_and_logging(self):
        logger = Mock(spec=logging.Logger)
        with (
            patch("sglang.srt.utils.log_utils.perf_counter_ns") as monotonic,
            patch("sglang.srt.utils.log_utils.time_ns") as wall,
        ):
            for threshold in (0, -1):
                recorder = SlowStageLogger(threshold, logger=logger)
                with recorder.scope(request_ids=("unused",)):
                    with recorder.stage("disabled") as details:
                        self.assertIsNone(details)
            monotonic.assert_not_called()
            wall.assert_not_called()
        logger.info.assert_not_called()

    def test_threshold_and_bounded_snapshot(self):
        logger = Mock(spec=logging.Logger)
        recorder = SlowStageLogger(1, logger=logger, tp_rank=2)
        ids = ["request\n" + "x" * 200] + [str(i) for i in range(9)]
        metadata = recorder.request_metadata(ids, len(ids))
        ids[0] = "changed"
        with patch(
            "sglang.srt.utils.log_utils.perf_counter_ns",
            side_effect=[0, 999_999, 2_000_000, 3_000_000],
        ):
            with recorder.stage("fast"):
                pass
            with recorder.stage("slow", **metadata):
                pass
        logger.info.assert_called_once()
        message = logger.info.call_args.args[0]
        data = json.loads(message)
        self.assertNotIn("\n", message)
        self.assertEqual(data["request_ids"][0], ("request\n" + "x" * 200)[:128])
        self.assertEqual(len(data["request_ids"]), 8)
        self.assertEqual(data["request_count"], 10)
        self.assertEqual(data["request_ids_omitted"], 2)
        self.assertEqual(data["request_ids_truncated"], 1)
        self.assertEqual(data["tp_rank"], 2)
        self.assertEqual(data["elapsed_ms"], 1)

    def test_nested_exception_restores_previous_context(self):
        logger = Mock(spec=logging.Logger)
        recorder = SlowStageLogger(1, logger=logger)
        with patch(
            "sglang.srt.utils.log_utils.perf_counter_ns",
            side_effect=count(0, 1_000_000),
        ):
            with recorder.scope(request_ids=("previous-batch",), forward_iter=1):
                with recorder.stage("outer"):
                    with self.assertRaisesRegex(ValueError, "private error"):
                        with recorder.stage(
                            "child", request_ids=("current-batch",), forward_iter=2
                        ):
                            raise ValueError("private error")
                    with recorder.stage("after-child"):
                        pass
            with recorder.stage("unbound"):
                pass
        records = {
            d["stage"]: d
            for call in logger.info.call_args_list
            for d in [json.loads(call.args[0])]
        }
        self.assertEqual(records["child"]["exception_type"], "ValueError")
        self.assertNotIn("private error", str(logger.info.call_args_list))
        self.assertEqual(records["child"]["parent_stage"], "outer")
        self.assertEqual(records["after-child"]["request_ids"], ["previous-batch"])
        self.assertEqual(records["after-child"]["forward_iter"], 1)
        self.assertNotIn("request_ids", records["unbound"])
        self.assertIsNone(records["unbound"]["parent_stage"])

    def test_context_does_not_leak_to_new_thread(self):
        logger = Mock(spec=logging.Logger)
        recorder = SlowStageLogger(1, logger=logger)

        def worker():
            with recorder.stage("worker"):
                pass

        with patch(
            "sglang.srt.utils.log_utils.perf_counter_ns",
            side_effect=count(0, 1_000_000),
        ):
            with recorder.scope(request_ids=("scheduler-request",)):
                thread = threading.Thread(target=worker)
                thread.start()
                thread.join()
                with recorder.stage("scheduler"):
                    pass
        records = [json.loads(call.args[0]) for call in logger.info.call_args_list]
        self.assertNotIn("request_ids", records[0])
        self.assertEqual(records[1]["request_ids"], ["scheduler-request"])


def _parse_log_json(line: str) -> dict:
    """Strip the ``[YYYY-MM-DD HH:MM:SS] `` prefix added by the formatter."""
    return json.loads(_LOG_PREFIX_RE.sub("", line))


def _flush_all(loggers: list) -> None:
    for logger in loggers:
        for handler in logger.handlers:
            handler.flush()


def _read_log_file(temp_dir: str) -> dict:
    log_files = list(Path(temp_dir).glob("*.log"))
    assert len(log_files) == 1
    return _parse_log_json(log_files[0].read_text().strip())


if __name__ == "__main__":
    unittest.main()
