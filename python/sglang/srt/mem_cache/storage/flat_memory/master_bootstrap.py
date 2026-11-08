# SPDX-License-Identifier: Apache-2.0
"""GVA master address for a Flat store: explicit, off, or started automatically.

Under "auto" one node runs the master: with nnodes == 1 that node's starter
rank, with nnodes > 1 the starter rank of node 0, bound on the host of
dist_init_addr so every node reaches the same master. The master journals to
/dev/shm and is restarted on the same port if it dies, so a crash loses no
key->GVA or allocation state.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

AUTO = "auto"
OFF = "off"
DEFAULT_MASTER_PORT = 50151
_WAIT_SECONDS = 60.0


def resolve_master_addr(
    *,
    mode: str,
    token: str,
    is_starter: bool,
    barrier: Callable[[], None] | None,
    node_rank: int = 0,
    nnodes: int = 1,
    dist_init_addr: str | None = None,
    master_port: int = DEFAULT_MASTER_PORT,
) -> tuple[str, Any]:
    """Return (master "host:port" or "", master handle this rank started or None).

    mode is a "host:port", "off" (or ""), or "auto". `token` must be equal on
    every rank of the cluster for nnodes > 1 and of the node otherwise.
    """
    if mode in (OFF, ""):
        return "", None
    if mode != AUTO:
        return mode, None
    if nnodes > 1:
        host = _dist_host(dist_init_addr)
        if host is not None:
            return _resolve_cluster(
                host=host, port=master_port, token=token,
                is_starter=is_starter and node_rank == 0, barrier=barrier)
        logger.warning("Flat GVA master: nnodes=%d without dist_init_addr; each "
                       "node runs its own master, so nodes do not dedup", nnodes)
    return _resolve_node(token=token, is_starter=is_starter, barrier=barrier)


def stop_daemon(master: Any, *, token: str) -> None:
    if master is None:
        return
    master.stop()
    try:
        os.unlink(_addr_path(token))
    except FileNotFoundError:
        pass


def _addr_path(token: str) -> str:
    return f"/dev/shm/flat_master_{token}"


def _resolve_node(*, token, is_starter, barrier) -> tuple[str, Any]:
    path = _addr_path(token)
    master, addr = None, ""
    if is_starter:
        master, addr = _start_daemon(host="127.0.0.1", port=0, token=token)
        _publish(path, addr)
    if barrier is not None:
        # Every rank calls this exactly once, after the starter has published.
        barrier()
    if is_starter:
        return addr, master
    return _read_published(path, wait=barrier is None), None


def _resolve_cluster(*, host, port, token, is_starter, barrier) -> tuple[str, Any]:
    # A fixed address replaces the /dev/shm hand-off, which no other node sees.
    addr = f"{host}:{port}"
    master = None
    if is_starter:
        master, bound = _start_daemon(host="0.0.0.0", port=port, token=token)
        if not bound:
            addr = ""
    if barrier is not None:
        barrier()
    if is_starter:
        return addr, master
    return (addr if _wait_reachable(host, port) else ""), None


def _start_daemon(*, host: str, port: int, token: str) -> tuple[Any, str]:
    from flat_memory_system import master_daemon

    state = f"{_addr_path(token)}.state"
    # A new server starts a new table; only a restart of this one may reuse it.
    for stale in (state, f"{state}.tmp"):
        try:
            os.unlink(stale)
        except FileNotFoundError:
            pass
    try:
        master = master_daemon.SupervisedMaster(host=host, port=port, state=state)
    except (OSError, RuntimeError) as exc:
        # Fail open, like the native client: running without a master is
        # today's single-node behaviour, not a correctness loss.
        logger.warning("Flat GVA master not started (%s); continuing without it", exc)
        return None, ""
    logger.info("Flat GVA master started at %s (pid %d)", master.addr, master.pid)
    return master, master.addr


def _dist_host(dist_init_addr: str | None) -> str | None:
    if not dist_init_addr:
        return None
    host = dist_init_addr.rsplit(":", 1)[0]
    return host.strip("[]") or None


def _wait_reachable(host: str, port: int) -> bool:
    deadline = time.monotonic() + _WAIT_SECONDS
    while True:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            if time.monotonic() >= deadline:
                logger.warning("Flat GVA master %s:%d unreachable; continuing "
                               "without a master", host, port)
                return False
            time.sleep(0.2)


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
