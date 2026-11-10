"""Node-local shared-memory all-reduce for Flat control-plane integers.

FlatMemoryCache agrees small integer vectors across TP ranks several times per
scheduler pass. Over gloo each one costs ~0.3-1.1 ms at TP8; when every rank of
the group shares one host, a shared-memory exchange takes tens of microseconds.
Any other topology keeps gloo.
"""

from __future__ import annotations

import logging
import mmap
import os
import platform
import socket
import time
import uuid

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Per rank and parity: [sequence, length, padding to one cache line, payload].
_HEADER_WORDS = 8
# Arbitrary; 256 KiB per slot keeps glibc memcpy off non-temporal stores, which
# x86 does not order before the sequence store. Longer vectors fall back to gloo.
_PAYLOAD_WORDS = 1 << 15
# Busy-wait iterations before each wait yields the GIL; arbitrary.
_SPINS_BEFORE_YIELD = 64


def control_reduce_group(cache):
    """The one process group cache._all_reduce_attn_groups reduces over, if any."""
    if not torch.distributed.is_initialized():
        return None
    groups = [
        group
        for group in (cache.attn_cp_group, cache.attn_tp_group)
        if group is not None and torch.distributed.get_world_size(group=group) > 1
    ]
    if not groups and cache.tp_world_size > 1:
        groups = [cache.tp_group]
    # Two reduce stages compose MIN/MAX across groups; keep gloo for that case.
    return groups[0] if len(groups) == 1 else None


class ShmControlReducer:
    """Sequence-numbered MIN/MAX all-reduce over a double-buffered shm segment.

    A rank reuses a parity buffer only two rounds later, and it can only reach
    that round after every rank published the round in between, which each does
    after it finished reading the round that used this buffer.
    """

    def __init__(self, *, region: mmap.mmap, rank: int, world: int, timeout: float):
        self._region = region
        self._rank = rank
        self._world = world
        self._timeout = timeout
        self._seq = 0
        self._slots = np.frombuffer(region, dtype=np.int64).reshape(
            2, world, _HEADER_WORDS + _PAYLOAD_WORDS
        )

    @staticmethod
    def region_bytes(world: int) -> int:
        return 2 * world * (_HEADER_WORDS + _PAYLOAD_WORDS) * 8

    @classmethod
    def create(cls, *, group, timeout: float) -> "ShmControlReducer | None":
        """Collective over `group`; every rank returns a reducer or every rank None."""
        world = torch.distributed.get_world_size(group=group)
        rank = torch.distributed.get_rank(group=group)
        hosts = [None] * world
        torch.distributed.all_gather_object(
            hosts, (socket.gethostname(), platform.machine()), group=group
        )
        # x86 total store order makes the payload visible before the sequence.
        if len(set(hosts)) != 1 or hosts[0][1] not in ("x86_64", "AMD64"):
            return None
        size = cls.region_bytes(world)
        path = [None]
        if rank == 0:
            path[0] = cls._create_file(size)
        torch.distributed.broadcast_object_list(
            path, src=torch.distributed.get_global_rank(group, 0), group=group
        )
        region = cls._map(path[0], size) if path[0] is not None else None
        mapped = [None] * world
        torch.distributed.all_gather_object(mapped, region is not None, group=group)
        # Every rank has mapped it (or given up), so the name is no longer needed.
        if rank == 0 and path[0] is not None:
            os.unlink(path[0])
        if not all(mapped):
            if region is not None:
                region.close()
            return None
        return cls(region=region, rank=rank, world=world, timeout=timeout)

    @staticmethod
    def _create_file(size: int) -> str | None:
        path = f"/dev/shm/flat_tp_control_{os.getpid()}_{uuid.uuid4().hex}"
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.ftruncate(fd, size)
            finally:
                os.close(fd)
        except OSError as error:
            logger.warning("Flat TP control shm unavailable, using gloo: %s", error)
            return None
        return path

    @staticmethod
    def _map(path: str, size: int) -> mmap.mmap | None:
        try:
            fd = os.open(path, os.O_RDWR)
        except OSError as error:
            logger.warning("Flat TP control shm attach failed, using gloo: %s", error)
            return None
        try:
            return mmap.mmap(fd, size)
        finally:
            os.close(fd)

    def all_reduce(self, values: list[int], op) -> list[int] | None:
        """None when `values` exceeds a slot; the length is rank-agreed, so all fall back."""
        count = len(values)
        if count > _PAYLOAD_WORDS:
            return None
        if op == torch.distributed.ReduceOp.MIN:
            reduce = np.min
        elif op == torch.distributed.ReduceOp.MAX:
            reduce = np.max
        else:
            raise ValueError(f"Unsupported Flat control reduce op: {op}")
        self._seq += 1
        slots = self._slots[self._seq & 1]
        mine = slots[self._rank]
        mine[_HEADER_WORDS : _HEADER_WORDS + count] = values
        mine[1] = count
        mine[0] = self._seq  # Published last; see the TSO note in create().
        self._wait_all(slots)
        if (slots[:, 1] != count).any():
            raise RuntimeError(
                "Flat TP control reduce lengths differ across ranks: "
                f"{slots[:, 1].tolist()}"
            )
        if count == 0:
            return []
        return reduce(slots[:, _HEADER_WORDS : _HEADER_WORDS + count], axis=0).tolist()

    def _wait_all(self, slots: np.ndarray) -> None:
        deadline = None
        spins = 0
        for rank in range(self._world):
            while slots[rank, 0] != self._seq:
                spins += 1
                if spins < _SPINS_BEFORE_YIELD:
                    continue
                # Releases the GIL so this process's I/O threads keep running.
                time.sleep(0)
                if spins % 1024:
                    continue
                now = time.monotonic()
                if deadline is None:
                    deadline = now + self._timeout
                elif now >= deadline:
                    raise RuntimeError(
                        f"Flat TP control reduce timed out after {self._timeout:.0f}s "
                        f"waiting for rank {rank} (round {self._seq})"
                    )

    def close(self) -> None:
        self._slots = None
        try:
            self._region.close()
        except BufferError:
            # A caller still holds a view; the mapping goes when that view does.
            pass
