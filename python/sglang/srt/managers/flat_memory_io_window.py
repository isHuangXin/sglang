"""Scheduler-thread control of rank-local Flat I/O measurement windows."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol


class FlatIOManager(Protocol):
    def get_io_window(self) -> dict: ...
    def io_clock_ns(self) -> int: ...
    def start_io_window(self, window_id: str, start_ns: int) -> bool: ...
    def end_io_window(
        self, window_id: str, end_ns: int, abort: bool = False
    ) -> bool: ...


def validate_flat_window_request(action: str, window_id: str) -> None:
    if (
        action not in ("begin", "end", "abort")
        or not isinstance(window_id, str)
        or not 1 <= len(window_id) <= 128
    ):
        raise ValueError("Invalid Flat I/O window action or ID")


def validate_flat_rank_snapshots(
    ranks: Sequence[Mapping[str, Any]], *, tp_size: int
) -> None:
    if type(tp_size) is not int or tp_size not in (1, 4, 8) or len(ranks) != tp_size:
        raise ValueError("Flat I/O metrics require every configured TP1/4/8 rank")
    if any(not isinstance(rank, Mapping) for rank in ranks):
        raise ValueError("Flat I/O rank snapshot is malformed")
    identities = [(rank.get("tp_rank"), rank.get("pid")) for rank in ranks]
    if any(type(rank) is not int for rank, _ in identities) or {
        rank for rank, _ in identities
    } != set(range(tp_size)):
        raise ValueError("Flat I/O ranks are missing or duplicated")
    pids = [pid for _, pid in identities]
    if (
        any(type(pid) is not int or pid <= 0 for pid in pids)
        or len(set(pids)) != tp_size
    ):
        raise ValueError("Flat I/O rank process identities are missing or duplicated")
    if any(rank.get("tp_size", tp_size) != tp_size for rank in ranks):
        raise ValueError("Flat I/O tensor-parallel topology changed")


def _errors(states: Sequence[Mapping[str, Any]]) -> str | None:
    errors = [
        f"rank {state['tp_rank']}: {state['error']}"
        for state in states
        if state.get("error")
    ]
    return "; ".join(errors) or None


def _validate_owner(action: str, window_id: str, current: Mapping[str, Any]) -> None:
    if action == "begin":
        if current["active"] and current["window_id"] != window_id:
            raise ValueError("Another Flat I/O window is active")
        if not current["active"] and current["window_id"] == window_id:
            raise ValueError("Use a fresh ID for a new Flat I/O window")
    elif current["window_id"] != window_id:
        raise ValueError("Flat I/O window owner does not match")


def _control_window(
    *,
    manager: FlatIOManager,
    tp_rank: int,
    tp_size: int,
    gather: Callable[[Any], list[Any]],
    action: str,
    window_id: str,
    idle: bool,
    drain: Callable[[], bool],
) -> str | None:
    state = {"tp_rank": tp_rank, "tp_size": tp_size, "pid": os.getpid(), "error": None}
    try:
        validate_flat_window_request(action, window_id)
        current = manager.get_io_window()
        _validate_owner(action, window_id, current)
        state["current"] = current
        needs_idle = (
            action == "begin"
            and not current["active"]
            or action == "end"
            and current["active"]
        )
        state["needs_drain"] = needs_idle
        if needs_idle and not idle:
            raise RuntimeError("Flat I/O is not idle: requests are still active")
    except Exception as exc:
        state["error"] = str(exc)
    states = gather(state)
    try:
        validate_flat_rank_snapshots(states, tp_size=tp_size)
    except ValueError as exc:
        return str(exc)
    error = _errors(states)
    if error:
        return error
    if action != "begin" or any(item["current"]["active"] for item in states):
        owners = {
            (
                item["current"]["window_id"],
                item["current"]["active"],
                item["current"].get("start_ns"),
            )
            for item in states
        }
        if len(owners) != 1:
            return "Flat I/O window ownership differs across TP ranks"
    if len({item["needs_drain"] for item in states}) != 1:
        return "Flat I/O drain state differs across TP ranks"
    if states[0]["needs_drain"]:
        # FLAT_MEMORY: Validate globally before drain, which may itself use TP collectives.
        drained = {"tp_rank": tp_rank, "error": None}
        try:
            if not drain():
                raise RuntimeError("Flat I/O drain did not complete")
        except Exception as exc:
            drained["error"] = str(exc)
        error = _errors(gather(drained))
        if error:
            return error

    clock = {"tp_rank": tp_rank, "error": None}
    try:
        clock["ns"] = manager.io_clock_ns()
    except Exception as exc:
        clock["error"] = str(exc)
    clocks = gather(clock)
    error = _errors(clocks)
    if error:
        return error
    boundary = next(item["ns"] for item in clocks if item["tp_rank"] == 0)
    result = {"tp_rank": tp_rank, "error": None}
    try:
        accepted = (
            manager.start_io_window(window_id, boundary)
            if action == "begin"
            else manager.end_io_window(window_id, boundary, abort=action == "abort")
        )
        if accepted is False:
            raise RuntimeError("Flat I/O window operation was rejected")
    except Exception as exc:
        result["error"] = str(exc)
    error = _errors(gather(result))
    if error and action != "abort":
        rollback = {"tp_rank": tp_rank, "error": None}
        try:
            current = manager.get_io_window()
            if current["active"] and current["window_id"] == window_id:
                if (
                    manager.end_io_window(window_id, manager.io_clock_ns(), abort=True)
                    is False
                ):
                    raise RuntimeError(
                        "Could not invalidate partially controlled Flat I/O window"
                    )
        except Exception as exc:
            rollback["error"] = str(exc)
        rollback_error = _errors(gather(rollback))
        if rollback_error:
            error += "; " + rollback_error
    return error


def collect_flat_memory_state(
    *,
    manager: FlatIOManager,
    tp_rank: int,
    tp_size: int,
    gather: Callable[[Any], list[Any]],
    snapshot: Callable[[], dict[str, Any]],
    idle: bool,
    drain: Callable[[], bool],
    action: str | None = None,
    window_id: str | None = None,
) -> dict[str, Any]:
    """Called on every scheduler rank in the same control-request order.

    ``gather`` uses the scheduler's CPU TP group. ``snapshot`` supplies backend
    metrics and ownership counters; this function adds stable rank/window fields.
    ``idle`` covers requests, and ``drain`` waits for owned I/O without cancelling it.
    """
    error = None
    if action is not None:
        error = _control_window(
            manager=manager,
            tp_rank=tp_rank,
            tp_size=tp_size,
            gather=gather,
            action=action,
            window_id=window_id,
            idle=idle,
            drain=drain,
        )
    local = {}
    try:
        local.update(snapshot())
        local["io_window"] = manager.get_io_window()
    except Exception as exc:
        local["error"] = str(exc)
    # FLAT_MEMORY: Backend statistics cannot override the process owning this reply.
    local.update(pid=os.getpid(), tp_rank=tp_rank, tp_size=tp_size)
    ranks = gather(local)
    try:
        validate_flat_rank_snapshots(ranks, tp_size=tp_size)
    except ValueError as exc:
        error = error or str(exc)
    error = error or _errors(ranks)
    result = {"flat_memory": {"tp_size": tp_size, "ranks": ranks}}
    if action is not None or error:
        result["flat_io_control"] = {"error": error}
    return result


def flat_io_window_response(
    states: Sequence[Mapping[str, Any]], *, action: str, window_id: str
) -> dict[str, Any]:
    """Validate the tokenizer fan-out result before returning an HTTP success."""
    if len(states) != 1:
        raise ValueError("Flat I/O windows require a single DP service group")
    state = states[0]
    control = state.get("flat_io_control")
    if not isinstance(control, Mapping) or control.get("error"):
        raise ValueError(
            str(
                control.get("error")
                if isinstance(control, Mapping)
                else "Flat I/O window control unavailable"
            )
        )
    flat = state.get("flat_memory")
    if not isinstance(flat, Mapping) or not isinstance(flat.get("ranks"), list):
        raise ValueError("Flat I/O rank snapshots are unavailable")
    validate_flat_rank_snapshots(flat["ranks"], tp_size=flat.get("tp_size"))
    for rank in flat["ranks"]:
        window = rank.get("io_window")
        if not isinstance(window, Mapping) or window.get("window_id") != window_id:
            raise ValueError("Flat I/O response belongs to a different window")
        if window.get("active") != (action == "begin"):
            raise ValueError("Flat I/O window did not reach the requested boundary")
        if action != "abort" and (
            not window.get("enabled")
            or window.get("aborted")
            or window.get("overflowed")
            or window.get("io_errors")
        ):
            raise ValueError("Flat I/O window measurements are invalid")
    return {"tp_size": flat["tp_size"], "ranks": flat["ranks"]}
