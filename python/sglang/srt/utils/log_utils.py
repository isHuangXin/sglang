from __future__ import annotations

import json
import logging
import os
import socket
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from itertools import islice
from logging.handlers import TimedRotatingFileHandler
from time import perf_counter_ns, time_ns
from typing import Iterable, List, Optional, Union

import torch.distributed as dist


def create_log_targets(
    *, targets: Optional[List[str]], name_prefix: str
) -> List[logging.Logger]:
    if not targets:
        return [_create_log_target_stdout(name_prefix)]
    return [_create_log_target(t, name_prefix) for t in targets]


def _create_log_target(target: str, name_prefix: str) -> logging.Logger:
    if target.lower() == "stdout":
        return _create_log_target_stdout(name_prefix)
    return _create_log_target_file(target, name_prefix)


def _create_log_target_stdout(name_prefix: str) -> logging.Logger:
    return _create_logger_with_handler(
        f"{name_prefix}.stdout", logging.StreamHandler(sys.stdout)
    )


def _create_log_target_file(directory: str, name_prefix: str) -> logging.Logger:
    os.makedirs(directory, exist_ok=True)
    hostname = socket.gethostname()
    rank = dist.get_rank() if dist.is_initialized() else 0
    filename = os.path.join(directory, f"{hostname}_{rank}.log")
    handler = TimedRotatingFileHandler(
        filename, when="H", backupCount=0, encoding="utf-8"
    )
    return _create_logger_with_handler(
        f"{name_prefix}.file.{directory}.{hostname}_{rank}", handler
    )


def _create_logger_with_handler(name: str, handler: logging.Handler) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def log_json(
    loggers: Union[logging.Logger, List[logging.Logger]], event: str, data: dict
) -> None:
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        **data,
    }
    msg = json.dumps(log_data, ensure_ascii=False)

    if not isinstance(loggers, list):
        loggers = [loggers]

    for logger in loggers:
        logger.info(msg)


class SlowStageLogger:
    def __init__(self, threshold_ms: float, *, logger: logging.Logger, **metadata):
        self.threshold_ns = int(threshold_ms * 1_000_000)
        self.logger = logger
        self.metadata = {"pid": os.getpid(), **metadata}
        self._context: ContextVar[Optional[dict]] = ContextVar(
            "slow_stage_context", default=None
        )
        self._stage: ContextVar[Optional[str]] = ContextVar(
            "slow_stage_parent", default=None
        )

    @staticmethod
    def request_metadata(request_ids: Iterable[str], request_count: int) -> dict:
        ids = tuple(str(rid) for rid in islice(request_ids, 8))
        return {
            "request_ids": tuple(rid[:128] for rid in ids),
            "request_count": request_count,
            "request_ids_omitted": max(0, request_count - len(ids)),
            "request_ids_truncated": sum(len(rid) > 128 for rid in ids),
        }

    @contextmanager
    def scope(self, **metadata):
        if self.threshold_ns <= 0:
            yield None
            return
        context = {**(self._context.get() or {}), **metadata}
        token = self._context.set(context)
        try:
            yield context
        finally:
            self._context.reset(token)

    @contextmanager
    def stage(self, name: str, **metadata):
        if self.threshold_ns <= 0:
            yield None
            return
        with self.scope(**metadata) as details:
            parent_stage = self._stage.get()
            token = self._stage.set(name)
            started_wall_ns = time_ns()
            started_ns = perf_counter_ns()
            exception_type = None
            try:
                yield details
            except BaseException as exc:
                exception_type = type(exc).__name__
                raise
            finally:
                finished_ns = perf_counter_ns()
                finished_wall_ns = time_ns()
                self._stage.reset(token)
                elapsed_ns = finished_ns - started_ns
                if elapsed_ns >= self.threshold_ns:
                    # Host intervals include waits; parent intervals include children.
                    log_json(
                        self.logger,
                        "slow_stage",
                        {
                            **self.metadata,
                            **details,
                            "stage": name,
                            "parent_stage": parent_stage,
                            "started_wall_ns": started_wall_ns,
                            "finished_wall_ns": finished_wall_ns,
                            "elapsed_ms": elapsed_ns / 1_000_000,
                            "inclusive": True,
                            "exception_type": exception_type,
                        },
                    )
