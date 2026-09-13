"""Age-per-byte SSD preparation with deterministic FIFO tie breaking."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from typing import Any

import msgspec


def ssd_order_key(*, queued_at: float, io_bytes: int, sequence: int, now: float):
    return (-max(0.0, now - queued_at) / max(1, io_bytes), sequence)


class _ReadTask(msgspec.Struct):
    future: Future
    function: Any
    args: tuple
    queued_at: float
    io_bytes: int
    sequence: int


class SSDReadExecutor:
    def __init__(self, workers: int = 2):
        self._pending: list[_ReadTask] = []
        self._condition = threading.Condition()
        self._closed = False
        self._sequence = 0
        self._threads = [
            threading.Thread(target=self._run, name=f"flat-ssd-{index}")
            for index in range(workers)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, function, *args, io_bytes: int = 0) -> Future:
        with self._condition:
            if self._closed:
                raise RuntimeError("Flat SSD executor is closed")
            future = Future()
            self._pending.append(
                _ReadTask(
                    future, function, args, time.monotonic(), io_bytes, self._sequence
                )
            )
            self._sequence += 1
            self._condition.notify()
            return future

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._closed)
                if not self._pending:
                    return
                now = time.monotonic()
                # FLAT_MEMORY: Recompute age at dequeue so large old reads cannot starve.
                index = min(
                    range(len(self._pending)),
                    key=lambda position: ssd_order_key(
                        queued_at=self._pending[position].queued_at,
                        io_bytes=self._pending[position].io_bytes,
                        sequence=self._pending[position].sequence,
                        now=now,
                    ),
                )
                task = self._pending.pop(index)
            if not task.future.set_running_or_notify_cancel():
                continue
            try:
                result = task.function(*task.args)
            except BaseException as error:
                task.future.set_exception(error)
            else:
                task.future.set_result(result)

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if wait:
            for thread in self._threads:
                thread.join()
