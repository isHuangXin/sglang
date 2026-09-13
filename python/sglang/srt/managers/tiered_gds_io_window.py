"""FLAT_MEMORY: All-TP control for dedicated native consumer GDS windows."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class GDSWindowStorage(Protocol):
    def get_gds_io_snapshot(self) -> dict: ...
    def begin_gds_io_window(self, window_id: str, start_ns: int) -> bool: ...
    def end_gds_io_window(
        self, window_id: str, end_ns: int, abort: bool = False
    ) -> bool: ...
    def gds_io_clock_ns(self) -> int: ...


def validate_gds_window_request(action: str, window_id: str) -> int:
    if (
        action not in ("begin", "end", "abort")
        or not isinstance(window_id, str)
        or not 1 <= len(window_id) <= 19
        or not window_id.isascii()
        or not window_id.isdecimal()
        or not 0 < int(window_id) < 2**63
    ):
        raise ValueError("Invalid GDS I/O window action or ID")
    return int(window_id)


def _owned_snapshot(storage: GDSWindowStorage) -> dict:
    snapshot = storage.get_gds_io_snapshot()
    if not isinstance(snapshot, dict) or snapshot.get("enabled") is not True:
        raise ValueError("GDS I/O counters are missing or disabled")
    if snapshot.get("schema_version") != 1 or snapshot.get("pid") != os.getpid():
        raise ValueError("GDS I/O snapshot schema or process owner does not match")
    window = snapshot.get("window")
    if not isinstance(window, dict) or type(window.get("active")) is not bool:
        raise ValueError("Missing GDS I/O window state")
    for name in ("id", "start_ns", "end_ns"):
        if type(window.get(name)) is not int or window[name] < 0:
            raise ValueError(f"Invalid GDS I/O window {name}")
    return window


def control_gds_io_window(
    action: str,
    window_id: str,
    *,
    storage: GDSWindowStorage,
    gather: Callable[[Any], list],
    is_idle: Callable[[], bool],
) -> dict:
    error, current = None, None
    try:
        ident = validate_gds_window_request(action, window_id)
        current = _owned_snapshot(storage)
        if action == "begin":
            if current["active"] and current["id"] != ident:
                raise ValueError("Another GDS I/O window is active")
            if not current["active"] and current["id"] == ident:
                raise ValueError("Use a fresh ID for a new GDS I/O window")
        elif current["id"] != ident:
            raise ValueError("GDS I/O window owner does not match")
        if action != "abort" and current.get("aborted", False):
            if action == "end" or current["active"]:
                raise ValueError("GDS I/O window was aborted")
        needs_idle = (action == "begin" and not current["active"]) or (
            action == "end" and current["active"]
        )
        if needs_idle and not is_idle():
            raise ValueError("GDS I/O is not idle")
        current = {key: current[key] for key in ("id", "active", "start_ns", "end_ns")}
    except Exception as exc:
        error = str(exc)
    states = gather((error, current))
    if any(state[0] for state in states):
        return {"error": str([state[0] for state in states])}
    windows = [state[1] for state in states]
    if action != "abort":
        active = [window["active"] for window in windows]
        if any(active) and not all(active):
            return {"error": "GDS TP window states diverged"}
        if (all(active) or action == "end") and any(
            window != windows[0] for window in windows
        ):
            return {"error": "GDS TP window boundaries diverged"}
        if (action == "begin" and all(active)) or (action == "end" and not any(active)):
            return {"error": None}

    # FLAT_MEMORY: Negotiate a shared monotonic boundary, not per-rank HTTP times.
    clock = None
    try:
        clock = storage.gds_io_clock_ns()
        if type(clock) is not int or clock <= 0:
            raise ValueError("Invalid native GDS clock")
    except Exception as exc:
        error = str(exc)
    clocks = gather((error, clock))
    if any(state[0] for state in clocks):
        return {"error": str([state[0] for state in clocks])}
    boundary = clocks[0][1]
    try:
        accepted = (
            storage.begin_gds_io_window(window_id, boundary)
            if action == "begin"
            else storage.end_gds_io_window(window_id, boundary, abort=action == "abort")
        )
        if not accepted:
            raise RuntimeError("GDS I/O window operation was rejected")
    except Exception as exc:
        error = str(exc)
    errors = gather(error)
    if any(errors):
        if action == "begin":
            _abort_partial_begin(storage=storage, window_id=window_id)
        return {"error": str(errors)}
    return {"error": None}


def _abort_partial_begin(*, storage: GDSWindowStorage, window_id: str) -> None:
    try:
        owned = _owned_snapshot(storage)
        if owned["id"] == int(window_id):
            storage.end_gds_io_window(window_id, storage.gds_io_clock_ns(), abort=True)
    except Exception:
        logger.exception("Could not abort a partially opened GDS I/O window")


def gds_io_window_response(states: list, *, action: str, window_id: str) -> dict:
    ident = validate_gds_window_request(action, window_id)
    if (
        not isinstance(states, list)
        or len(states) != 1
        or not isinstance(states[0], dict)
    ):
        raise ValueError("GDS I/O windows require exactly one scheduler DP state")
    state = states[0]
    control = state.get("gds_io_control")
    if (
        not isinstance(control, dict)
        or "error" not in control
        or control["error"] is not None
    ):
        raise ValueError(f"GDS I/O control failed: {control}")
    telemetry = state.get("gds_io")
    ranks = telemetry.get("ranks") if isinstance(telemetry, dict) else None
    if not isinstance(ranks, list) or not ranks:
        raise ValueError("Missing GDS I/O rank snapshots")
    seen, pids = set(), set()
    for rank in ranks:
        if not isinstance(rank, dict):
            raise ValueError("Invalid GDS I/O rank snapshot")
        rank_id, pid = rank.get("tp_rank"), rank.get("pid")
        if (
            type(rank_id) is not int
            or not 0 <= rank_id < len(ranks)
            or rank_id in seen
            or rank.get("tp_size") != len(ranks)
            or type(pid) is not int
            or pid <= 0
            or pid in pids
            or rank.get("enabled") is not True
        ):
            raise ValueError("Incomplete or duplicate GDS I/O rank ownership")
        seen.add(rank_id)
        pids.add(pid)
        window = rank.get("window")
        if (
            not isinstance(window, dict)
            or window.get("id") != ident
            or window.get("active") is not (action == "begin")
            or window.get("aborted") is not (action == "abort")
        ):
            raise ValueError("GDS I/O response window owner or state does not match")
    return {"ranks": ranks}
