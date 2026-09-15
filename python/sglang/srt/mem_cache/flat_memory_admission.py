"""Resource offers for scheduler-owned Flat restore leases."""

import msgspec


class FlatRestoreAdmission(msgspec.Struct, frozen=True):
    can_restore: bool
    never_fits: bool = False
    full_reservation: int = 0
    swa_reservation: int = 0
    reason: str = ""
