"""FLAT_MEMORY: Logical token provenance for the tiered Mooncake GDS path."""

from collections.abc import Sequence


def tiered_cache_breakdown(
    prefix_len: int,
    host_hit_length: int,
    source_ranges: Sequence[tuple[int, int, int]],
) -> dict[str, int]:
    if type(prefix_len) is not int or prefix_len < 0:
        raise ValueError("Invalid consumed tiered-cache prefix length")
    if type(host_hit_length) is not int or host_hit_length < 0:
        raise ValueError("Invalid tiered-cache Host hit length")
    sources = bytearray(prefix_len)
    for start, end, medium in source_ranges:
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError("Invalid tiered-cache source range")
        start, end = min(prefix_len, start), min(prefix_len, end)
        if start < end:
            # FLAT_MEMORY: Native source 0 is failure, never a DRAM/SSD hit.
            # Later ranges (including Host=4) override the earlier restore source.
            sources[start:end] = bytes(
                [medium if type(medium) is int and medium in (1, 2, 3, 4) else 255]
            ) * (end - start)
    if host_hit_length:
        start = max(0, prefix_len - host_hit_length)
        sources[start:] = bytes([4]) * (prefix_len - start)
    if 255 in sources:
        raise ValueError("Unknown provenance in consumed tiered-cache prefix")
    l2_host = sources.count(4)
    dram = sources.count(1)
    mixed = sources.count(3)
    return {
        "device": sources.count(0),
        "host": l2_host + dram,
        "l2_host": l2_host,
        "mooncake_dram": dram,
        "ssd": sources.count(2) + mixed,
        "mixed": mixed,
    }
