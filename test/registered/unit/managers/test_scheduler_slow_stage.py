"""Keep diagnostic context aligned with overlap result snapshots."""

import json
import logging
import unittest
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.utils.log_utils import SlowStageLogger

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _batch(rid):
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid=rid)],
        forward_mode=SimpleNamespace(name="EXTEND"),
        forward_iter=999,
        extend_num_tokens=16,
    )
    batch.copy = lambda: SimpleNamespace(
        reqs=batch.reqs[:],
        forward_mode=batch.forward_mode,
        forward_iter=batch.forward_iter,
        extend_num_tokens=batch.extend_num_tokens,
    )
    return batch


def _run_overlap(*, enabled, disable_overlap):
    calls = []
    logger = Mock(spec=logging.Logger)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.slow_stage_logger = (
        SlowStageLogger(1, logger=logger, tp_rank=2) if enabled else None
    )
    scheduler.forward_ct = 0
    scheduler.waiting_queue = []
    scheduler.gracefully_exit = False
    scheduler._engine_paused = False
    scheduler.running_batch = None
    scheduler.last_batch = None
    scheduler.is_generation = True
    scheduler.enable_unified_memory = False
    batches = iter([_batch("A"), _batch("B")])

    def receive():
        calls.append("receive")
        return []

    def prepare(**kwargs):
        calls.append("prepare")
        return SimpleNamespace(running_batch=None, batch_to_run=next(batches))

    def forward(batch):
        scheduler.forward_ct += 1
        batch.forward_iter = scheduler.forward_ct
        calls.append(f"forward:{batch.forward_iter}")
        return SimpleNamespace(forward_iter=batch.forward_iter)

    def process(batch, result):
        calls.append(f"result:{batch.forward_iter}")
        batch.reqs.clear()
        if scheduler.slow_stage_logger is not None:
            with scheduler.slow_stage_logger.stage("processor.marker"):
                pass
        scheduler.gracefully_exit = True

    scheduler.request_receiver = SimpleNamespace(recv_requests=receive)
    scheduler.process_input_requests = lambda reqs: calls.append("input")
    scheduler.get_next_batch_to_run = prepare
    scheduler.is_disable_overlap_for_batch = lambda batch, *, last_batch: (
        disable_overlap and last_batch is not None
    )
    scheduler.run_batch = forward
    scheduler._apply_war_barrier = lambda: calls.append("fence")
    scheduler.process_batch_result = process
    scheduler.launch_batch_sample_if_needed = lambda result, batch: calls.append(
        f"sample:{batch.forward_iter}"
    )
    with (
        envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(0),
        patch(
            "sglang.srt.utils.log_utils.perf_counter_ns",
            side_effect=count(0, 1_000_000),
        ),
    ):
        scheduler.event_loop_overlap()
    records = [json.loads(call.args[0]) for call in logger.info.call_args_list]
    return calls, records


class TestSchedulerSlowStage(CustomTestCase):
    def test_overlap_identity_and_order_with_mutated_result(self):
        for disable_overlap in (False, True):
            with self.subTest(disable_overlap=disable_overlap):
                original_calls, original_records = _run_overlap(
                    enabled=False, disable_overlap=disable_overlap
                )
                calls, records = _run_overlap(
                    enabled=True, disable_overlap=disable_overlap
                )
                self.assertEqual(calls, original_calls)
                self.assertFalse(original_records)
                self.assertEqual(calls.count("fence"), 2)
                self.assertEqual(calls.count("result:1"), 1)
                self.assertEqual(
                    calls.index("result:1") < calls.index("forward:2"),
                    disable_overlap,
                )
                result_records = [
                    record
                    for record in records
                    if record["stage"] in ("scheduler.result", "processor.marker")
                ]
                self.assertEqual(len(result_records), 2)
                for record in result_records:
                    self.assertEqual(record["forward_iter"], 1)
                    self.assertEqual(record["request_ids"], ["A"])
                    self.assertEqual(record["request_count"], 1)
                forward_records = [
                    record
                    for record in records
                    if record["stage"] == "scheduler.forward"
                ]
                self.assertEqual(
                    [record["next_forward_iter"] for record in forward_records], [1, 2]
                )
                self.assertTrue(
                    all(record["forward_iter"] is None for record in forward_records)
                )

    def test_copy_wait_does_not_add_synchronization(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                processor = SchedulerBatchResultProcessor.__new__(
                    SchedulerBatchResultProcessor
                )
                recorder = (
                    SlowStageLogger(1, logger=Mock(spec=logging.Logger))
                    if enabled
                    else None
                )
                object.__setattr__(processor, "slow_stage_logger", recorder)
                result = SimpleNamespace(copy_done=Mock())
                with patch(
                    "sglang.srt.utils.log_utils.perf_counter_ns",
                    side_effect=[0, 1_000_000],
                ):
                    processor._synchronize_result_copy(result)
                result.copy_done.synchronize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
