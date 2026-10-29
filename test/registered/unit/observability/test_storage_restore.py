"""FLAT_MEMORY: Restore evidence requires adoption and the last all-TP completion."""

import importlib.util
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# FLAT_MEMORY: Function tests and direct loading avoid test_utils/package GPU imports.
ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def modules():
    pytest.importorskip(
        "msgspec", reason="CPU lifecycle requires the real msgspec dependency"
    )
    loaded = []
    with patch.dict(sys.modules):
        for name, path in (
            ("restore_lifecycle", "srt/observability/storage_restore.py"),
            ("restore_contract", "benchmark/storage_restore_report.py"),
        ):
            spec = importlib.util.spec_from_file_location(
                name, ROOT / "python/sglang" / path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
        yield loaded


def register(module, tracker, rid):
    timing = module.StorageRestoreTiming()
    episode = timing.begin(tp_size=4, start_ns=1_000_000)
    tracker.register(rid=rid, episode=episode, page_size=64)
    return timing, episode


def test_first_completed_episode_retry_and_retraction(modules):
    m, contract = modules
    timing = m.StorageRestoreTiming()
    old = timing.begin(tp_size=4, start_ns=1_000_000)
    assert timing.begin(tp_size=4, start_ns=9_000_000) is old
    timing.retract()
    episode = timing.begin(tp_size=4, start_ns=10_000_000)
    old.observe_ready(ready_ns=20_000_000, endpoint="flat_all_tp_ready")
    old.adopt(tokens=64)
    assert old.status == "cancelled" and episode.episode_id == 1
    episode.restored_tokens = 128
    episode.observe_ready(ready_ns=30_000_000, endpoint="flat_all_tp_ready")
    assert episode.status == "pending"
    episode.adopt(tokens=64, source=m.RestoreSource())
    records, outcomes = contract.validate_timing(timing.payload())
    assert outcomes == {"cancelled": 1, "completed": 1}
    assert records[0]["elapsed_ms"] == 20.0
    assert records[0]["adopted_tokens"] == 64
    assert not records[0]["source_complete"]
    assert timing.begin(tp_size=4, start_ns=99_000_000) is episode


def test_host_insertion_first_layer_and_last_rank_are_not_full_ready(modules):
    m, contract = modules
    tracker = m.HostRestoreTracker()
    timing, episode = register(m, tracker, "request")
    episode.storage_pages[0] = m.RestoreSource(2, True)
    assert episode.ready_ns is None
    for dep, offset in ((10, 0), (11, 64)):
        tracker.queued(
            dependency=dep,
            pages={offset: m.RestoreSource(2, True)},
            required_pages=set(),
        )
        tracker.watch(rid="request", dependency=dep, start=offset, end=offset + 64)
    tracker.adopt(rid="request", prefix_tokens=128)
    tracker.ready(dependencies=[10], ready_ns=2_000_000)
    assert episode.status == "pending"
    ranks = [Mock() for _ in range(4)]
    for event in ranks:
        event.query.return_value = True
    ranks[-1].query.return_value = False
    assert not all(event.query() for event in ranks)
    assert timing.payload(final=True)["status"] == "unsupported"
    ranks[-1].query.return_value = True
    if all(event.query() for event in ranks):
        tracker.ready(dependencies=[11], ready_ns=8_000_000)
    records, _ = contract.validate_timing(timing.payload(final=True))
    assert records[0]["elapsed_ms"] == 7.0
    assert records[0]["source_mask"] == 2
    assert not tracker.waiters and not tracker.load_sources


def test_shared_merged_dependency_and_split_prefix_source(modules):
    m, contract = modules
    tracker = m.HostRestoreTracker()
    full_timing, full = register(m, tracker, "full")
    short_timing, short = register(m, tracker, "split")
    tracker.queued(
        dependency=10,
        pages={0: m.RestoreSource(1, True), 64: m.RestoreSource(2, True)},
        required_pages=set(),
    )
    tracker.queued(
        dependency=11, pages={128: m.RestoreSource(1, True)}, required_pages=set()
    )
    tracker.watch(rid="full", dependency=10, start=0, end=128)
    tracker.watch(rid="full", dependency=11, start=128, end=192)
    tracker.watch(rid="split", dependency=10, start=0, end=64)
    tracker.adopt(rid="full", prefix_tokens=192)
    tracker.adopt(rid="split", prefix_tokens=64)
    tracker.ready(dependencies=[10, 11], ready_ns=5_000_000)
    assert contract.validate_timing(full_timing.payload())[0][0]["source_mask"] == 3
    assert contract.validate_timing(short_timing.payload())[0][0]["source_mask"] == 1
    assert full.ready_ns == short.ready_ns == 5_000_000


def test_source_reductions_unknown_and_trailing_pool_partial_adoption(modules):
    m, contract = modules
    rank_sources = [
        [m.RestoreSource(1, True)],
        [m.RestoreSource(2, True)],
        [m.RestoreSource()],
        [m.RestoreSource(3, True)],
    ]
    reduced = [
        min(values)
        for values in zip(
            *(m.pack_restore_sources(sources) for sources in rank_sources)
        )
    ]
    assert m.unpack_restore_sources(reduced) == [m.RestoreSource(3, False)]
    assert m.group_restore_sources([1, 2, 1, 0], 2) == [
        m.RestoreSource(3, True),
        m.RestoreSource(1, False),
    ]
    tracker = m.HostRestoreTracker()
    timing, _ = register(m, tracker, "partial")
    tracker.queued(
        dependency=10,
        pages={0: m.RestoreSource(3, True), 64: m.RestoreSource(3, True)},
        required_pages={0, 64},
    )
    tracker.watch(rid="partial", dependency=10, start=0, end=64)
    tracker.ready(dependencies=[10], ready_ns=4_000_000)
    tracker.adopt(rid="partial", prefix_tokens=64)
    record = contract.validate_timing(timing.payload())[0][0]
    assert record["source_mask"] == 0 and not record["source_complete"]


def test_cancel_flush_late_callback_and_pure_hbm_are_not_samples(modules):
    m, contract = modules
    tracker = m.HostRestoreTracker()
    timing, episode = register(m, tracker, "cancel")
    tracker.queued(
        dependency=10, pages={0: m.RestoreSource(1, True)}, required_pages=set()
    )
    tracker.watch(rid="cancel", dependency=10, start=0, end=64)
    tracker.adopt(rid="cancel", prefix_tokens=64)
    tracker.cancel(rid="cancel")
    tracker.ready(dependencies=[10], ready_ns=7_000_000)
    assert contract.validate_timing(timing.payload())[0] == []
    assert episode.status == "cancelled" and episode.ready_ns is None
    hbm, _ = register(m, tracker, "hbm")
    tracker.adopt(rid="hbm", prefix_tokens=128)
    assert contract.validate_timing(hbm.payload()) == ([], {"not_applicable": 1})
    flushed, _ = register(m, tracker, "flush")
    tracker.reset()
    assert contract.validate_timing(flushed.payload())[0] == []
    assert not tracker.requests and not tracker.waiters


def test_discarded_tail_stale_values_and_later_episode_do_not_poison_sample(modules):
    m, contract = modules
    for tail_ready in (None, 9_000_000):
        tracker = m.HostRestoreTracker()
        timing, episode = register(m, tracker, "trim")
        for dependency, offset in ((10, 0), (11, 64)):
            tracker.queued(
                dependency=dependency,
                pages={offset: m.RestoreSource(1, True)},
                required_pages=set(),
            )
            tracker.watch(
                rid="trim", dependency=dependency, start=offset, end=offset + 64
            )
        tracker.ready(dependencies=[10], ready_ns=3_000_000)
        if tail_ready is not None:
            tracker.ready(dependencies=[11], ready_ns=tail_ready)
        tracker.adopt(rid="trim", prefix_tokens=64)
        assert episode.status == "completed" and episode.ready_ns == 3_000_000
        tracker.ready(dependencies=[11], ready_ns=10_000_000)
        assert episode.ready_ns == 3_000_000
        timing.retract()
        timing.begin(tp_size=4, start_ns=20_000_000)
        payload = timing.payload(final=True)
        assert payload["status"] == "available"
        records, outcomes = contract.validate_timing(payload)
        assert len(records) == 1 and outcomes["pending"] == 1

    tracker = m.HostRestoreTracker()
    timing, episode = register(m, tracker, "stale")
    original, replacement = Mock(), Mock()
    tracker.queued(
        dependency=20, pages={0: m.RestoreSource(2, True)}, required_pages=set()
    )
    tracker.watch(rid="stale", dependency=20, start=0, end=64, values=(original, None))
    tracker.ready(dependencies=[20], ready_ns=3_000_000)
    tracker.adopt(
        rid="stale", prefix_tokens=64, values_by_page={0: (replacement, None)}
    )
    assert episode.status == "discarded"
    assert contract.validate_timing(timing.payload())[0] == []
    assert not tracker.owners

    timing, episode = register(m, tracker, "reload")
    for value, mask, ready in ((original, 2, 3_000_000), (replacement, 1, 6_000_000)):
        tracker.queued(
            dependency=20, pages={0: m.RestoreSource(mask, True)}, required_pages=set()
        )
        tracker.watch(
            rid="reload", dependency=20, start=0, end=64, values=(value, None)
        )
        tracker.ready(dependencies=[20], ready_ns=ready)
    tracker.adopt(
        rid="reload", prefix_tokens=64, values_by_page={0: (replacement, None)}
    )
    record = contract.validate_timing(timing.payload())[0][0]
    assert record["source_mask"] == 1 and record["source_complete"]
    assert episode.ready_ns == 6_000_000


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
