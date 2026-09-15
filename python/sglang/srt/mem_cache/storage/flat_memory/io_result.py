"""Flat admission and I/O outcomes shared by workers and the scheduler."""

import msgspec


class FlatCapacityError(RuntimeError):
    """The configured storage cannot admit a batch; no failed I/O is implied."""


class FlatUnsafeIOError(RuntimeError):
    """I/O quiescence is unproven; its buffers must not be recycled."""


class FlatWriteResult(msgspec.Struct, frozen=True):
    success: bool
    capacity_rejected: bool = False
    error: str | None = None
    unsafe: bool = False
