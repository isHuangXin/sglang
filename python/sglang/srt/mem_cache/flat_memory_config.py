# SPDX-License-Identifier: Apache-2.0
"""Configuration boundary for Flat Memory's direct device-cache integration."""

from __future__ import annotations

from typing import Any

from sglang.srt.mem_cache.storage_config import load_storage_extra_config


def validate_flat_memory_direct(cfg: Any) -> dict:
    # FLAT_MEMORY: Validate before allocating device pools or starting I/O workers.
    config = load_storage_extra_config(cfg.hicache_storage_backend_extra_config)
    if config.get("gds_mode", "off") != "compat":
        raise ValueError("Flat direct cache requires gds_mode='compat'")
    if cfg.hicache_storage_backend != "flat_memory":
        raise ValueError(
            "Flat direct cache requires --hicache-storage-backend flat_memory"
        )
    if cfg.radix_cache_backend not in (None, "flat_memory"):
        raise ValueError("Flat compat cannot be combined with another radix backend")
    if cfg.tp_size not in (1, 4, 8):
        raise ValueError("Flat compat supports TP1, TP4, and TP8")
    if (
        cfg.pp_size != 1
        or cfg.dp_size != 1
        or cfg.attn_cp_size != 1
        or cfg.dcp_size != 1
        or cfg.enable_dp_attention
    ):
        raise ValueError("Flat compat requires PP1/DP1 and no CP/DCP/DP attention")
    # FLAT_MEMORY: Each node keeps its own DRAM segment, so a multi-node run
    # gets per-node stores with no sharing between them -- the native side
    # holds a single remote segment, not a pool. Stated as a limit rather than
    # left to fail later: nnodes > 1 otherwise starts and silently caches less
    # than the cluster's DRAM.
    if cfg.nnodes != 1:
        raise ValueError(
            "Flat compat is single-node; cross-node DRAM pooling is not implemented"
        )
    if cfg.hicache_write_policy != "write_through":
        raise ValueError("Flat compat requires --hicache-write-policy write_through")
    if cfg.speculative_algorithm is not None:
        raise ValueError("Flat compat does not yet restore speculative/draft state")
    if cfg.enable_hisparse or cfg.enable_unified_memory:
        raise ValueError("Flat compat does not support HiSparse or VMM pool relocation")
    if cfg.disable_radix_cache:
        raise ValueError("Flat compat requires radix prefix caching")
    if cfg.enable_lmcache or cfg.enable_flexkv or cfg.enable_lora:
        raise ValueError(
            "Flat compat cannot share storage ownership with LMCache/FlexKV/LoRA"
        )
    if cfg.enable_streaming_session or cfg.enable_session_radix_cache:
        raise ValueError(
            "Flat compat currently supports ordinary prefix reuse, not sessions"
        )
    if cfg.disaggregation_mode != "null":
        raise ValueError(
            "Flat compat currently requires a non-disaggregated server; use the host backend for PD"
        )
    # FLAT_MEMORY: remote_dram_gb is the name FlatMemoryConfig.from_dict reads;
    # the *_capacity_gb spellings stay rejected so an old config is not ignored.
    if config.get("enable_persistence", False) or any(
        config.get(name, 0)
        for name in (
            "remote_dram_gb",
            "remote_dram_capacity_gb",
            "remote_ssd_capacity_gb",
        )
    ):
        raise ValueError("Flat compat does not support persistence or remote DRAM")
    _validate_remote_ssd(config, num_segments=cfg.tp_size * cfg.nnodes)
    return config


def _validate_remote_ssd(config: dict, *, num_segments: int) -> None:
    # Another node's SSD (flat_memory_system.ssd_server), the overflow tier.
    if not config.get("remote_ssd_gb", 0):
        return
    if not config.get("remote_ssd_addr"):
        raise ValueError("remote_ssd_gb requires remote_ssd_addr (the owner's host:port)")
    segment_id = int(config.get("remote_ssd_segment_id", -1))
    # Ranks take ids 0..num_segments-1 and the native remote-DRAM default is 1,
    # so the owner's id must sit above both or the router would alias them.
    if segment_id < max(2, num_segments) or segment_id >= 4096:
        raise ValueError(
            f"remote_ssd_segment_id must be in [{max(2, num_segments)}, 4096) "
            "and equal the owner's --segment-id"
        )
