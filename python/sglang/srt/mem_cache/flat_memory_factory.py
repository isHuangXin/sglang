# SPDX-License-Identifier: Apache-2.0
"""Construct the Flat device-cache path without allocating a host cache tier."""

from __future__ import annotations

import hashlib
import json
import logging

from sglang.srt.environ import envs
from sglang.srt.mem_cache.flat_memory_config import validate_flat_memory_direct
from sglang.srt.mem_cache.storage_config import is_flat_memory_direct
from sglang.srt.runtime_context import (
    get_context,
    get_memory,
    get_observability,
    get_schedule,
)

logger = logging.getLogger(__name__)


def _make_storage_collector(*, tp_rank: int):
    from sglang.srt.observability.metrics_collector import (
        STAT_LOGGER_ROLE_STORAGE,
        StorageMetricsCollector,
        resolve_collector_class,
    )

    labels = {
        "storage_backend": "flat_memory",
        "tp_rank": tp_rank,
        "dp_rank": 0,
        "pp_rank": 0,
        "pp_size": 1,
        "attn_cp_rank": 0,
        "attn_cp_size": 1,
    }
    labels.update(get_observability().extra_metric_labels or {})
    collector = resolve_collector_class(
        STAT_LOGGER_ROLE_STORAGE, StorageMetricsCollector
    )
    return collector(labels=labels)


def create_flat_memory_cache(ctx):
    from sglang.srt.mem_cache.registry import _create_unified_radix_cache

    memory = get_memory()
    if not is_flat_memory_direct(
        memory.hicache_storage_backend, memory.hicache_storage_backend_extra_config
    ):
        if (
            memory.hicache_storage_backend != "flat_memory"
            or not ctx.enable_hierarchical_cache
        ):
            raise ValueError(
                "The host Flat backend requires --enable-hierarchical-cache"
            )
        return _create_unified_radix_cache(ctx, ctx.server_args, ctx.params)

    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.mem_cache.flat_memory_cache import FlatMemoryCache
    from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
        resolve_flat_device_pool_group,
    )
    from sglang.srt.mem_cache.storage.flat_memory.flat_memory_direct_linker import (
        FlatMemoryDirectLinker,
    )

    config = validate_flat_memory_direct(resolving_view(ctx.server_args))
    if (
        envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get() != "python"
        or envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get()
    ):
        raise ValueError("Flat compat currently requires the Python unified tree core")
    if (
        ctx.disable_radix_cache
        or ctx.is_hybrid_ssm
        or ctx.params.mtp_draft_device_pools
    ):
        raise ValueError(
            "Flat compat does not support disabled radix, SSM, or draft pools"
        )
    if ctx.model_config.is_multimodal:
        raise ValueError("Flat compat currently supports text-only prefix reuse")

    # FLAT_MEMORY: Do not route the device path through init_hicache, even for legacy flags.
    cache = _create_unified_radix_cache(
        ctx, ctx.server_args, ctx.params, attach_hicache=False
    )
    pool = ctx.params.token_to_kv_pool_allocator.get_kvcache()
    group = resolve_flat_device_pool_group(
        kvcache=pool,
        page_size=ctx.params.page_size,
        params=ctx.params,
        components=set(cache.components),
    )
    model_identity = {
        "path": ctx.server_args.model_path,
        "revision": ctx.server_args.revision,
        "config": ctx.model_config.hf_config.to_dict(),
    }
    namespace = hashlib.sha256(
        json.dumps(model_identity, sort_keys=True, default=str).encode()
    ).hexdigest()
    linker = FlatMemoryDirectLinker(
        pool_group=group,
        config=config,
        model_namespace=namespace,
        tp_rank=ctx.tp_rank,
        tp_size=ctx.tp_size,
        device=cache.device,
    )
    try:
        runtime = FlatMemoryCache(
            cache=cache,
            cache_linker=linker,
            config=config,
            ready_fcfs=(
                get_schedule().schedule_policy == "fcfs"
                and not get_schedule().enable_priority_scheduling
            ),
        )
        collector = (
            _make_storage_collector(tp_rank=ctx.tp_rank)
            if ctx.params.enable_metrics
            else None
        )
    except BaseException:
        linker.close()
        raise
    cache.flat_memory = runtime
    cache.linker = runtime
    cache.enable_storage_metrics = ctx.params.enable_metrics
    cache.storage_metrics_collector = collector
    # The scheduler's controller-independent hooks drive Flat's completion lifecycle.
    get_context().override(
        "flat_memory.factory",
        enable_hierarchical_cache=False,
        radix_cache_backend="flat_memory",
    )
    logger.info(
        "Flat device cache initialized: TP%d rank=%d host_payload_bytes=0 namespace=%s",
        ctx.tp_size,
        ctx.tp_rank,
        namespace[:16],
    )
    return cache
