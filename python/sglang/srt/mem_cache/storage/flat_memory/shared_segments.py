"""FLAT_MEMORY: node-shared DRAM segments for rank-replicated pools.

The node keeps one DRAM segment and one published-block index rather than one
per TP rank. Rank 0 creates, sizes and seals anonymous memfds, then hands the
descriptors to every other rank over a Unix socket with SCM_RIGHTS, along with
a small text payload the owner can only produce after its own startup.

memfd rather than shm_open because tmpfs pages are charged to the /dev/shm
mount (100 GiB on the B200 node) while memfd pages are charged only to the
process cgroup, so the node DRAM budget is bounded by physical memory the way
Mooncake's aligned_alloc is. There is also no name to collide between runs or
leak after a crash: the kernel frees the segment once the last rank closes its
fd.

share_dram=False is the aligned_alloc + RDMA mode: the owner's DRAM is an
ordinary heap allocation registered with the transfer engine, so there is no fd
to hand over and peers read it over RDMA instead of mapping it. Only the index
travels as a memfd then, and the payload carries the owner's engine name and
segment base address.
"""

import fcntl
import logging
import mmap
import os
import socket

logger = logging.getLogger(__name__)

# Mirrors kSharedIndexSlots and SharedBlockIndex's 64-byte header plus 40-byte
# slots in src/flat_memory/include/shared_block_index.h. The native side
# fstats the fd and refuses to bind a short region, so a drift here fails at
# startup instead of corrupting the table.
_INDEX_SLOTS = 1 << 21
_INDEX_BYTES = 64 + _INDEX_SLOTS * 40

_CONNECT_TIMEOUT_S = 120.0

# Pinning 512 GiB of 4 KiB pages takes each rank minutes; 2 MiB pages cut the
# page count 512x. They come from the reserved pool (vm.nr_hugepages).
_HUGE_PAGE_BYTES = 2 << 20
_FREE_HUGE_PAGES = "/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages"


def _cgroup_huge_page_headroom() -> int:
    # A Kubernetes pod without a hugepages-2Mi request has hugetlb.2MB.max = 0 on
    # its pod slice, so free host pages are still unusable here (mmap ENOMEM).
    try:
        with open("/proc/self/cgroup") as f:
            path = next(line.split(":", 2)[2].strip() for line in f if line.startswith("0::"))
    except (OSError, StopIteration):
        return 1 << 62
    headroom = 1 << 62
    while True:
        base = "/sys/fs/cgroup" + (path if path != "/" else "")
        # Both the usage and the reservation limit gate a MAP_SHARED hugetlb mmap.
        for kind in ("", ".rsvd"):
            try:
                with open(f"{base}/hugetlb.2MB{kind}.max") as f:
                    limit = f.read().strip()
                with open(f"{base}/hugetlb.2MB{kind}.current") as f:
                    current = int(f.read())
                if limit != "max":
                    headroom = min(headroom, max(int(limit) - current, 0))
            except (OSError, ValueError):
                pass
        if path in ("", "/"):
            return headroom
        path = os.path.dirname(path)


def _free_huge_page_bytes() -> int:
    try:
        with open(_FREE_HUGE_PAGES) as f:
            free = int(f.read())
        # Free pages still count the ones other mappings have reserved.
        with open(_FREE_HUGE_PAGES.replace("free_hugepages", "resv_hugepages")) as f:
            free = (free - int(f.read())) * _HUGE_PAGE_BYTES
    except (OSError, ValueError):
        return 0
    return min(free, _cgroup_huge_page_headroom())


class FlatSharedSegments:
    """Owns this rank's fds for the node-shared segment and index."""

    def __init__(
        self,
        *,
        token: str,
        peers: int,
        dram_bytes: int,
        owner: bool,
        share_dram: bool = True,
        hugepages: str = "auto",
    ):
        if hugepages not in ("auto", "on", "off"):
            raise ValueError(f"dram_hugepages must be auto, on or off, got {hugepages!r}")
        # Abstract socket namespace (leading NUL): no filesystem entry, so a
        # crashed run leaves nothing behind for the next one to trip over.
        self._address = "\0" + token[:100]
        self._peers = peers
        # Whole huge pages either way, so owner and peers agree on the size.
        self._dram_bytes = -(-dram_bytes // _HUGE_PAGE_BYTES) * _HUGE_PAGE_BYTES
        self._hugepages = hugepages
        self._owner = owner
        self._share_dram = share_dram
        self._listener: socket.socket | None = None
        # Owner: a handover is owed to the peers. Peer: one was received. Both
        # sides gate on this rather than on the listener, so the owner can hold
        # one listener for the whole run and still skip the extra distribute()
        # that a manager rebuilt by clear() would otherwise trigger.
        self._pending = owner
        self._attached = False
        self.dram_fd = -1
        self.index_fd = -1
        # Owner-supplied text the peers need before they construct a manager.
        self.payload = ""
        if owner:
            self._create(dram_bytes=self._dram_bytes)

    def _create(self, *, dram_bytes: int) -> None:
        self.index_fd = os.memfd_create(
            "flat_index", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
        )
        os.ftruncate(self.index_fd, _INDEX_BYTES)
        fds = [self.index_fd]
        if self._share_dram:
            flags = os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
            if self._use_hugepages(dram_bytes):
                flags |= os.MFD_HUGETLB | os.MFD_HUGE_2MB
            self.dram_fd = os.memfd_create("flat_dram", flags)
            os.ftruncate(self.dram_fd, dram_bytes)
            fds.append(self.dram_fd)
        # Sealing makes the size part of the handed-over object: a peer cannot
        # shrink it, which would turn every mapped rank's next access into a
        # SIGBUS. The owner stays the sole writer by convention, not by seal,
        # since it writes through the same mapping.
        seals = fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL
        for fd in fds:
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
        # Bind before the caller's barrier so no peer can connect early.
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(self._address)
        self._listener.listen(max(self._peers, 1))

    def _use_hugepages(self, dram_bytes: int) -> bool:
        if self._hugepages == "off":
            return False
        free = _free_huge_page_bytes()
        if free >= dram_bytes:
            logger.info("Flat DRAM segment uses 2 MiB huge pages (%d GiB free)", free >> 30)
            return True
        if self._hugepages == "on":
            raise RuntimeError(
                f"dram_hugepages=on needs {dram_bytes >> 30} GiB of free 2 MiB huge pages "
                f"but {free >> 30} GiB are usable; reserve them with vm.nr_hugepages "
                f"(and, in a Kubernetes pod, request hugepages-2Mi)"
            )
        logger.warning(
            "Flat DRAM segment uses 4 KiB pages: %d GiB of free 2 MiB huge pages, "
            "%d GiB needed; startup pinning will be slow",
            free >> 30, dram_bytes >> 30,
        )
        return False

    def rearm(self) -> None:
        """Allow a second handover of the payload after the managers are rebuilt.

        The fds survive a clear(), but an RDMA-shared owner's segment does not:
        dropping the manager frees and unregisters the heap allocation, so the
        rebuilt one has a new base address and the peers' stale copy points at
        memory the NIC no longer knows. Call on every rank between dropping the
        old manager and building the new one; the caller's barrier orders the
        rebind against the peers' reconnect.
        """
        if self._share_dram:
            return
        if self._owner:
            # The listener outlives the first handover in this mode, so there is
            # nothing to rebind: rebinding would hit EADDRINUSE against itself.
            self._pending = True
            return
        # Keep the index fd: it is the same sealed memfd either way, and
        # attach() would otherwise leak one fd per clear().
        self._attached = False
        self.payload = ""

    def distribute(self, payload: str = "") -> None:
        """Owner: hand the fds and *payload* to every peer. Call after a barrier.

        A no-op once the handover is done: the fds outlive a manager rebuilt by
        clear(), so the segment is handed over exactly once per run -- unless
        rearm() asked for another round because the payload itself went stale.
        """
        if not self._owner or not self._pending or self._listener is None:
            return
        blob = payload.encode()
        fds = [self.index_fd] if self.dram_fd < 0 else [self.index_fd, self.dram_fd]
        for _ in range(self._peers):
            conn, _addr = self._listener.accept()
            with conn:
                socket.send_fds(conn, [blob or b"flat"], fds)
        self._pending = False
        if self._share_dram:
            # The fds are all the peers ever need here, so drop the bound
            # address. Under rdma the listener stays up for a rearm() round.
            self._listener.close()
            self._listener = None

    def attach(self) -> None:
        """Peer: receive the fds and payload. Call after the same barrier.

        A no-op once attached, for the same reason distribute() is.
        """
        if self._owner or self._attached:
            return
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(_CONNECT_TIMEOUT_S)
        try:
            sock.connect(self._address)
            # The owner's engine name plus base address is well under 4 KiB.
            msg, fds, _flags, _addr = socket.recv_fds(sock, 4096, 2)
        finally:
            sock.close()
        expected = 2 if self._share_dram else 1
        if len(fds) != expected:
            raise RuntimeError(
                f"Flat shared segment handover returned {len(fds)} fds, "
                f"expected {expected}"
            )
        self._attached = True
        self.payload = "" if msg == b"flat" else msg.decode()
        if self._share_dram:
            self.index_fd, self.dram_fd = fds
            size = os.fstat(self.dram_fd).st_size
            if size != self._dram_bytes:
                raise RuntimeError(
                    f"Flat shared DRAM segment is {size} bytes but this rank "
                    f"expects {self._dram_bytes}; the ranks disagree on the "
                    f"node budget"
                )
        else:
            # A rearm()ed peer reattaches to the same sealed index memfd, so
            # drop the previous descriptor rather than leaking one per clear().
            if self.index_fd >= 0:
                os.close(self.index_fd)
            (self.index_fd,) = fds

    def reformat_index(self) -> None:
        """Owner: zero the index header so a rebuilt manager reformats it.

        The fds outlive a manager recreated by clear(); SharedBlockIndex only
        treats a region as unformatted when its ready word is clear, and it
        relies on the slots being zero, which a fresh memfd gives for free but
        a reused one does not.
        """
        if not self._owner:
            return
        chunk = b"\0" * (1 << 20)
        with mmap.mmap(self.index_fd, _INDEX_BYTES) as region:
            for start in range(0, _INDEX_BYTES, len(chunk)):
                end = min(start + len(chunk), _INDEX_BYTES)
                region[start:end] = chunk[: end - start]

    def close(self) -> None:
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        if self.dram_fd >= 0:
            os.close(self.dram_fd)
            self.dram_fd = -1
        if self.index_fd >= 0:
            os.close(self.index_fd)
            self.index_fd = -1
