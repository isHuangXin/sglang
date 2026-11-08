# SPDX-License-Identifier: Apache-2.0
"""GVA master address for a Flat store: explicit, off, or started per node."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Callable

logger = logging.getLogger(__name__)

AUTO = "auto"
OFF = "off"
_WAIT_SECONDS = 60.0


def resolve_master_addr(
    *,
    mode: str,
    token: str,
    is_starter: bool,
    barrier: Callable[[], None] | None,
) -> tuple[str, subprocess.Popen | None]:
    """Return (master "host:port" or "", daemon process this rank started or None).

    mode is a "host:port", "off" (or ""), or "auto". Under "auto" the node's
    starter rank launches gva_master_daemon and every rank of the node reads
    its address from a /dev/shm file keyed by `token`.
    """
    if mode in (OFF, ""):
        return "", None
    if mode != AUTO:
        return mode, None
    path = f"/dev/shm/flat_master_{token}"
    if is_starter:
        proc, addr = _start_daemon()
        _publish(path, addr)
    if barrier is not None:
        # Every rank calls this exactly once, after the starter has published.
        barrier()
    if is_starter:
        return addr, proc
    return _read_published(path, wait=barrier is None), None


def stop_daemon(proc: subprocess.Popen | None, *, token: str) -> None:
    if proc is None:
        return
    from flat_memory_system import master_daemon

    master_daemon.stop(proc)
    try:
        os.unlink(f"/dev/shm/flat_master_{token}")
    except FileNotFoundError:
        pass


def _start_daemon() -> tuple[subprocess.Popen | None, str]:
    from flat_memory_system import master_daemon

    try:
        proc, addr = master_daemon.start(host="127.0.0.1", port=0)
    except (OSError, RuntimeError) as exc:
        # Fail open, like the native client: running without a master is
        # today's single-node behaviour, not a correctness loss.
        logger.warning("Flat GVA master not started (%s); continuing without it", exc)
        return None, ""
    logger.info("Flat GVA master started at %s (pid %d)", addr, proc.pid)
    return proc, addr


def _publish(path: str, addr: str) -> None:
    # pid lets a reader reject a file left by a dead server on the same port;
    # an empty address means the starter ran without a master.
    tmp = f"{path}.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(f"{os.getpid()} {addr or OFF}\n")
    os.replace(tmp, path)


def _read_published(path: str, *, wait: bool) -> str:
    deadline = time.monotonic() + (_WAIT_SECONDS if wait else 0.0)
    while True:
        try:
            with open(path) as f:
                pid_text, addr = f.read().split()
            os.kill(int(pid_text), 0)
            return "" if addr == OFF else addr
        except (FileNotFoundError, ValueError, ProcessLookupError):
            if time.monotonic() >= deadline:
                logger.warning("Flat GVA master address not published at %s; "
                               "continuing without a master", path)
                return ""
            time.sleep(0.1)
