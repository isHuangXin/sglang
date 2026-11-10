"""The shared-memory Flat control reduce matches a true cross-rank MIN/MAX."""

import os
import random
import socket
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.mem_cache.flat_memory_control_reduce import (
    _PAYLOAD_WORDS,
    ShmControlReducer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_WORLD = 8
_ROUNDS = 400


def _round_values(rank: int, step: int) -> tuple[list[int], object]:
    """Every rank derives all ranks' inputs, so each can check the exact answer."""
    rng = random.Random(step)
    count = rng.choice([0, 1, 2, 7, 64, 513, 2048])
    op = rng.choice([dist.ReduceOp.MIN, dist.ReduceOp.MAX])
    values = [
        random.Random(step * 1000 + rank * 7919 + i).randint(-(1 << 40), 1 << 40)
        for i in range(count)
    ]
    return values, op


def _worker(rank: int, port: int, failures):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD)
    try:
        reducer = ShmControlReducer.create(group=dist.group.WORLD, timeout=60.0)
        assert reducer is not None, "single-host x86 group must use shared memory"
        jitter = random.Random(rank)
        for step in range(_ROUNDS):
            # Uneven arrival lets fast ranks run a round ahead and reuse a parity.
            if jitter.random() < 0.2:
                time.sleep(jitter.random() * 0.002)
            values, op = _round_values(rank, step)
            peers = [_round_values(peer, step)[0] for peer in range(_WORLD)]
            pick = min if op == dist.ReduceOp.MIN else max
            expected = [pick(column) for column in zip(*peers)] if values else []
            got = reducer.all_reduce(values, op)
            if got != expected:
                failures.put(f"rank {rank} step {step}: {got[:4]} != {expected[:4]}")
                return
        # Oversized vectors are declined on every rank alike, so gloo takes over.
        if reducer.all_reduce([0] * (_PAYLOAD_WORDS + 1), dist.ReduceOp.MIN) is not None:
            failures.put(f"rank {rank}: oversized vector was not declined")
        reducer.close()
    except Exception as error:  # Surface any rank's failure to the parent.
        failures.put(f"rank {rank}: {error!r}")
    finally:
        dist.destroy_process_group()


def _length_mismatch_worker(rank: int, port: int, failures):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        reducer = ShmControlReducer.create(group=dist.group.WORLD, timeout=60.0)
        try:
            reducer.all_reduce([1] * (rank + 1), dist.ReduceOp.MIN)
            failures.put(f"rank {rank}: diverged lengths were not rejected")
        except RuntimeError as error:
            if "lengths differ" not in str(error):
                failures.put(f"rank {rank}: wrong error {error!r}")
        reducer.close()
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestFlatMemoryControlReduce(CustomTestCase):
    def _spawn(self, target, nprocs):
        context = mp.get_context("spawn")
        failures = context.SimpleQueue()
        mp.start_processes(
            target,
            args=(_free_port(), failures),
            nprocs=nprocs,
            start_method="spawn",
        )
        messages = []
        while not failures.empty():
            messages.append(failures.get())
        return messages

    def test_matches_exact_min_max_under_uneven_arrival(self):
        """Double-buffer reuse must never let a fast rank overwrite unread data."""
        self.assertEqual(self._spawn(_worker, _WORLD), [])

    def test_diverged_lengths_fail_instead_of_mixing_rounds(self):
        self.assertEqual(self._spawn(_length_mismatch_worker, 2), [])


if __name__ == "__main__":
    unittest.main()
