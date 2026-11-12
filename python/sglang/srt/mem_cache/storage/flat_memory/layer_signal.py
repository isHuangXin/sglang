"""GPU-side layer readiness flags for Flat layer-wise restore.

The forward stream waits on a flag instead of the launching thread blocking. Flags
live in pinned host memory and are raised by a plain CPU store in native
read_plan; a CUDA call to raise them could block behind the pending wait. The wait
is a bounded spin kernel (flat_memory_system wait_layer_flag): a wait that cannot
time out can still deadlock against a module load while the reader needs the
driver. A timed-out layer is reported, and its restore is treated as failed.
"""

from __future__ import annotations

import ctypes
import logging

import torch

logger = logging.getLogger(__name__)


class LayerSignals:
    """One uint32 flag per (slot, layer); a slot's flags only ever grow."""

    def __init__(
        self, *, device: torch.device, num_layers: int, slots: int, timeout_ms: float
    ):
        from flat_memory_system import _core

        self._wait_layer_flag = _core.wait_layer_flag
        self.device = device
        self.num_layers = num_layers
        self.slots = slots
        self._timeout_ns = int(timeout_ms * 1e6)
        # Pinned host memory is device-mapped under UVA, at the same address.
        self._flags = torch.zeros(slots * num_layers, dtype=torch.int32, pin_memory=True)
        self._timed_out = torch.zeros(slots, dtype=torch.int32, pin_memory=True)
        self._base = self._flags.data_ptr()
        # Load the kernel now: a first launch behind a pending wait would block.
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            self._wait_layer_flag(
                stream.cuda_stream, self._base, 0, self._timeout_ns, self._timed_out.data_ptr()
            )
            stream.synchronize()

    def slot_address(self, slot: int) -> int:
        """Host address of the slot's layer-0 flag; layer l is 4 * l bytes on."""
        return self._base + 4 * slot * self.num_layers

    def can_wait(self) -> bool:
        # A captured graph would replay the wait with a stale value; CPU-wait instead.
        return not torch.cuda.is_current_stream_capturing()

    def wait(self, *, slot: int, layer: int, value: int) -> None:
        # Enqueued on the current (forward) stream; the CPU returns at once.
        self._wait_layer_flag(
            torch.cuda.current_stream(self.device).cuda_stream,
            self.slot_address(slot) + 4 * layer,
            value,
            self._timeout_ns,
            self._timed_out.data_ptr() + 4 * slot,
        )

    def signal(self, *, slot: int, layer: int, value: int) -> None:
        # Backup to the native store in read_plan, for paths that skip it.
        ctypes.c_uint32.from_address(self.slot_address(slot) + 4 * layer).value = value

    def timed_out(self, *, slot: int, value: int) -> bool:
        """True when a forward stopped waiting on this restore before its data."""
        return ctypes.c_uint32.from_address(self._timed_out.data_ptr() + 4 * slot).value == value


def create_layer_signals(
    *, device: torch.device, num_layers: int, timeout_ms: float, slots: int = 4096
) -> LayerSignals | None:
    if device.type != "cuda":
        return None
    try:
        return LayerSignals(
            device=device, num_layers=num_layers, slots=slots, timeout_ms=timeout_ms
        )
    except (ImportError, AttributeError, RuntimeError) as error:
        logger.warning("Flat layer-wise GPU wait unavailable, CPU wait kept: %s", error)
        return None
