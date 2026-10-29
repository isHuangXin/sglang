# SPDX-License-Identifier: Apache-2.0
"""FLAT_MEMORY: Request-local restore evidence; no device or request ownership."""

from __future__ import annotations

from time import monotonic_ns
from weakref import ReferenceType, ref

import msgspec


class RestoreSource(msgspec.Struct, frozen=True):
    mask: int = 0
    complete: bool = False


def merge_restore_sources(sources) -> RestoreSource:
    mask, complete = 0, True
    for source in sources:
        mask |= source.mask
        complete &= source.complete
    return RestoreSource(mask, complete)


def group_restore_sources(masks: list[int], multiplier: int) -> list[RestoreSource]:
    return [
        merge_restore_sources(
            RestoreSource(mask if mask in (1, 2, 3) else 0, mask in (1, 2, 3))
            for mask in masks[start : start + multiplier]
        )
        for start in range(0, len(masks), multiplier)
    ]


def pack_restore_sources(sources: list[RestoreSource]) -> list[int]:
    # FLAT_MEMORY: MIN carries validity and negated any-memory/any-disk flags.
    return [
        flag
        for source in sources
        for flag in (
            int(source.complete),
            -int(bool(source.mask & 1)),
            -int(bool(source.mask & 2)),
        )
    ]


def unpack_restore_sources(values: list[int]) -> list[RestoreSource]:
    return [
        RestoreSource(int(bool(memory)) | (int(bool(disk)) << 1), bool(complete))
        for complete, memory, disk in zip(values[::3], values[1::3], values[2::3])
    ]


class RestoreSourceBatch(msgspec.Struct):
    pools: dict[str, list[RestoreSource]] = {}

    def pages(self, *, count: int) -> list[RestoreSource]:
        if not self.pools:
            return [RestoreSource()] * count
        return [
            merge_restore_sources(
                sources[index] if index < len(sources) else RestoreSource()
                for sources in self.pools.values()
            )
            for index in range(count)
        ]


class RestoreEpisode(msgspec.Struct, kw_only=True):
    episode_id: int
    start_ns: int
    status: str = "pending"
    ready_ns: int | None = None
    restored_tokens: int = 0
    adopted_tokens: int = 0
    endpoint: str = ""
    source: RestoreSource = RestoreSource()
    dependencies: set[int] = set()
    dependency_pages: dict[int, dict[int, RestoreSource]] = {}
    dependency_ready_ns: dict[int, int] = {}
    dependency_source_required: dict[int, set[int]] = {}
    admitted: bool = False
    page_size: int = 1
    restored_pages: dict[int, RestoreSource] = {}
    storage_pages: dict[int, RestoreSource] = {}
    source_requires_full_restore: bool = False
    source_required_pages: set[int] = set()
    outcome: str = "not_applicable"

    def close(self, status: str) -> None:
        if self.status == "pending":
            self.status = status
            self.dependencies.clear()
            self.dependency_pages.clear()
            self.dependency_ready_ns.clear()
            self.dependency_source_required.clear()
            self.storage_pages.clear()
            self.restored_pages.clear()
            self.source_required_pages.clear()

    def observe_ready(self, *, ready_ns: int, endpoint: str) -> None:
        if self.status != "pending":
            return
        self.ready_ns = ready_ns
        self.endpoint = endpoint
        self._complete()

    def dependency_ready(self, *, dependency: int, ready_ns: int) -> None:
        if self.status != "pending" or dependency not in self.dependencies:
            return
        self.dependencies.remove(dependency)
        self.dependency_ready_ns[dependency] = ready_ns
        self.update_host_ready()

    def update_host_ready(self) -> None:
        self.ready_ns = None
        if self.dependency_pages and not self.dependencies:
            self.observe_ready(
                ready_ns=max(
                    self.dependency_ready_ns[dep] for dep in self.dependency_pages
                ),
                endpoint="host_all_layer_ack_all_tp_ready",
            )

    def adopt(self, *, tokens: int, source: RestoreSource | None = None) -> None:
        if self.status != "pending":
            return
        self.admitted = True
        self.adopted_tokens = min(tokens, self.restored_tokens)
        if source is not None:
            self.source = source
        if self.adopted_tokens <= 0:
            self.close("discarded" if self.restored_tokens else self.outcome)
        else:
            self._complete()

    def _complete(self) -> None:
        if (
            self.admitted
            and self.adopted_tokens > 0
            and self.ready_ns is not None
            and not self.dependencies
        ):
            self.status = "completed"
            self.dependency_pages.clear()
            self.dependency_ready_ns.clear()
            self.dependency_source_required.clear()
            self.storage_pages.clear()
            self.restored_pages.clear()
            self.source_required_pages.clear()

    def payload(self) -> dict:
        result = {
            "episode_id": self.episode_id,
            "start_ns": self.start_ns,
            "status": self.status,
        }
        if self.status == "completed":
            result.update(
                ready_ns=self.ready_ns,
                elapsed_ms=(self.ready_ns - self.start_ns) / 1e6,
                restored_tokens=self.restored_tokens,
                adopted_tokens=self.adopted_tokens,
                all_tp_ready=True,
                endpoint=self.endpoint,
                source_mask=self.source.mask,
                source_complete=self.source.complete and self.source.mask != 0,
            )
        return result


class StorageRestoreTiming(msgspec.Struct):
    tp_size: int = 1
    enabled: bool = False
    reason: str | None = None
    episodes: list[RestoreEpisode] = []
    active: RestoreEpisode | None = None

    def begin(self, *, tp_size: int, start_ns: int | None = None) -> RestoreEpisode:
        self.enabled = True
        self.tp_size = tp_size
        if self.active is None:
            self.active = RestoreEpisode(
                episode_id=len(self.episodes),
                start_ns=monotonic_ns() if start_ns is None else start_ns,
            )
            self.episodes.append(self.active)
        return self.active

    def unsupported(self, reason: str) -> None:
        self.reason = reason
        self.close("not_applicable")

    def close(self, status: str) -> None:
        if self.active is not None:
            self.active.close(status)

    def retract(self) -> None:
        self.close("cancelled")
        self.active = None

    def payload(self, *, final: bool = False) -> dict | None:
        if not self.enabled:
            return None
        reason = self.reason
        if (
            final
            and any(episode.status == "pending" for episode in self.episodes)
            and not any(episode.status == "completed" for episode in self.episodes)
        ):
            reason = reason or "all_tp_ready_not_observed_before_final_output"
        result = dict(
            schema_version=1,
            scope="scheduler_prefetch_start_to_all_tp_gpu_ready_observed",
            sample_unit="first_successful_adopted_restore_per_successful_request",
            clock="monotonic_ns",
            tp_size=self.tp_size,
            status="unsupported" if reason else "available",
            episodes=[episode.payload() for episode in self.episodes],
        )
        if reason:
            result["reason"] = reason
        return result


class RestoreValueSnapshot(msgspec.Struct, frozen=True):
    refs: tuple[ReferenceType | None, ...]

    @classmethod
    def capture(cls, values: tuple[object | None, ...]) -> RestoreValueSnapshot:
        return cls(tuple(ref(value) if value is not None else None for value in values))

    def matches(self, values: tuple[object | None, ...]) -> bool:
        return len(values) == len(self.refs) and all(
            (
                (saved() is value and value is not None)
                if saved is not None
                else value is None
            )
            for saved, value in zip(self.refs, values)
        )


class HostRestoreTracker(msgspec.Struct):
    requests: dict[str, RestoreEpisode] = {}
    waiters: dict[int, list[RestoreEpisode]] = {}
    load_sources: dict[int, dict[int, RestoreSource]] = {}
    load_full_source: dict[int, set[int]] = {}
    generations: dict[int, int] = {}
    next_generation: int = 0
    owners: dict[str, dict[tuple[int, int], RestoreValueSnapshot]] = {}

    def register(self, *, rid: str, episode: RestoreEpisode, page_size: int) -> None:
        previous = self.requests.get(rid)
        if previous is not None and previous is not episode:
            self.cancel(rid=rid)
        episode.page_size = page_size
        self.requests[rid] = episode

    def queued(
        self,
        *,
        dependency: int,
        pages: dict[int, RestoreSource],
        required_pages: set[int],
    ) -> None:
        self.load_sources[dependency] = pages
        self.load_full_source[dependency] = required_pages.copy()
        # FLAT_MEMORY: Native ACK node IDs can be reused by sequential reloads.
        self.generations[dependency] = self.next_generation
        self.next_generation += 1

    def watch(
        self,
        *,
        rid: str,
        dependency: int,
        start: int,
        end: int,
        values: tuple[object | None, ...] | None = None,
    ) -> None:
        episode = self.requests.get(rid)
        if episode is None or episode.status != "pending" or episode.admitted:
            return
        generation = self.generations[dependency]
        if generation not in episode.dependencies:
            episode.dependencies.add(generation)
            episode.ready_ns = None
            self.waiters.setdefault(generation, []).append(episode)
        pages = self.load_sources[dependency]
        snapshot = RestoreValueSnapshot.capture(values) if values is not None else None
        dependency_pages = episode.dependency_pages.setdefault(generation, {})
        for offset in range(start, end, episode.page_size):
            source = pages.get(offset, RestoreSource())
            dependency_pages[offset] = source
            previous = episode.restored_pages.get(offset)
            episode.restored_pages[offset] = (
                merge_restore_sources((previous, source))
                if previous is not None
                else source
            )
            if snapshot is not None:
                self.owners.setdefault(rid, {})[generation, offset] = snapshot
        episode.dependency_source_required[generation] = self.load_full_source[
            dependency
        ].copy()
        episode.restored_tokens = len(episode.restored_pages) * episode.page_size

    def ready(self, *, dependencies: list[int], ready_ns: int) -> None:
        for dependency in dependencies:
            generation = self.generations.pop(dependency, None)
            if generation is not None:
                for episode in self.waiters.pop(generation, ()):
                    episode.dependency_ready(dependency=generation, ready_ns=ready_ns)
            self.load_sources.pop(dependency, None)
            self.load_full_source.pop(dependency, None)
        self.requests = {
            rid: episode
            for rid, episode in self.requests.items()
            if episode.status == "pending"
        }
        self.owners = {
            rid: owners for rid, owners in self.owners.items() if rid in self.requests
        }

    def adopt(
        self,
        *,
        rid: str,
        prefix_tokens: int,
        values_by_page: dict[int, tuple[object | None, ...]] | None = None,
    ) -> None:
        episode = self.requests.get(rid)
        if episode is None or episode.admitted:
            return
        if episode.status != "pending":
            self.requests.pop(rid)
            self.owners.pop(rid, None)
            return
        adopted, adopted_dependencies = {}, {}
        owners = self.owners.pop(rid, {})
        for dependency, pages in episode.dependency_pages.items():
            selected = {}
            for offset, source in pages.items():
                if offset + episode.page_size > prefix_tokens:
                    continue
                # FLAT_MEMORY: Matching logical offsets do not prove device-value identity.
                snapshot = owners.get((dependency, offset))
                if values_by_page is not None and (
                    snapshot is None
                    or not snapshot.matches(values_by_page.get(offset, ()))
                ):
                    continue
                selected[offset] = source
                previous = adopted.get(offset)
                adopted[offset] = (
                    merge_restore_sources((previous, source))
                    if previous is not None
                    else source
                )
            if selected:
                adopted_dependencies[dependency] = selected
        episode.dependency_pages = adopted_dependencies
        episode.dependencies.intersection_update(episode.dependency_pages)
        episode.update_host_ready()
        tokens = len(adopted) * episode.page_size
        source = merge_restore_sources(adopted.values())
        if any(
            not episode.dependency_source_required[dependency].issubset(adopted)
            for dependency in episode.dependency_pages
        ):
            source = RestoreSource()
        episode.adopt(tokens=tokens, source=source)
        if episode.status != "pending":
            self.requests.pop(rid)

    def cancel(self, *, rid: str, status: str = "cancelled") -> None:
        episode = self.requests.pop(rid, None)
        self.owners.pop(rid, None)
        if episode is not None:
            episode.close(status)
        # FLAT_MEMORY: Waiters own only telemetry, never Req, nodes, tensors or events.
        for dependency, waiters in list(self.waiters.items()):
            kept = [
                item
                for item in waiters
                if item is not episode and item.status == "pending"
            ]
            if kept:
                self.waiters[dependency] = kept
            else:
                self.waiters.pop(dependency)

    def reset(self) -> None:
        for episode in self.requests.values():
            episode.close("cancelled")
        for waiters in self.waiters.values():
            for episode in waiters:
                episode.close("cancelled")
        self.requests.clear()
        self.waiters.clear()
        self.load_sources.clear()
        self.load_full_source.clear()
        self.generations.clear()
        self.owners.clear()
