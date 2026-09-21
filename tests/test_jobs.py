"""Job registry: progress, completion, failure, cancellation and SSE."""
from __future__ import annotations

import json
import time

import pytest

from app.jobs import JobRegistry, sse_stream


def _wait_for(job, statuses, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job.status in statuses:
            return True
        time.sleep(0.01)
    return False


def test_job_runs_and_reports_its_result():
    registry = JobRegistry()
    job = registry.create("test")

    registry.run(job, lambda j, progress: {"answer": 42})
    assert _wait_for(job, {"done"})
    assert job.result == {"answer": 42}
    assert job.progress == 1.0


def test_progress_events_are_recorded_in_order():
    registry = JobRegistry()
    job = registry.create("test")

    def target(j, progress):
        for i in range(1, 4):
            progress(f"step {i}", i / 3)
        return "ok"

    registry.run(job, target)
    assert _wait_for(job, {"done"})

    stages = [e.stage for e in job.events]
    assert stages == ["step 1", "step 2", "step 3"]
    assert [e.progress for e in job.events] == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_progress_is_clamped():
    registry = JobRegistry()
    job = registry.create("test")

    def target(j, progress):
        progress("over", 5.0)
        progress("under", -1.0)
        return None

    registry.run(job, target)
    assert _wait_for(job, {"done"})
    assert all(0.0 <= e.progress <= 1.0 for e in job.events)


def test_failure_is_captured_not_raised():
    """A failing job must record its error, not take down the server."""
    registry = JobRegistry()
    job = registry.create("test")

    def target(j, progress):
        raise ValueError("no trend providers returned anything")

    registry.run(job, target)
    assert _wait_for(job, {"failed"})
    assert job.status == "failed"
    assert "no trend providers" in job.error
    assert job.result is None


def test_cancellation_stops_the_job():
    registry = JobRegistry()
    job = registry.create("test")
    started = []

    def target(j, progress):
        for i in range(500):
            started.append(i)
            progress(f"step {i}", i / 500)
            time.sleep(0.005)
        return "finished"

    registry.run(job, target)
    while not started:
        time.sleep(0.01)

    assert registry.cancel(job.id)
    assert _wait_for(job, {"cancelled"})
    assert job.result is None


def test_cancelling_a_finished_job_is_rejected():
    registry = JobRegistry()
    job = registry.create("test")
    registry.run(job, lambda j, p: "done")
    assert _wait_for(job, {"done"})
    assert registry.cancel(job.id) is False


def test_unknown_job_is_none():
    assert JobRegistry().get("nope") is None


def test_jobs_are_listed_newest_first():
    registry = JobRegistry()
    first = registry.create("a")
    time.sleep(0.01)
    second = registry.create("b")
    assert [j.id for j in registry.list()][:2] == [second.id, first.id]


def test_sse_replays_history_for_a_late_subscriber():
    """A browser connecting mid-render must still see what already happened."""
    registry = JobRegistry()
    job = registry.create("test")

    def target(j, progress):
        progress("one", 0.3)
        progress("two", 0.6)
        return "ok"

    registry.run(job, target)
    assert _wait_for(job, {"done"})

    frames = [f for f in sse_stream(job, registry) if f.startswith("data:")]
    payloads = [json.loads(f[len("data: "):]) for f in frames]
    stages = [p.get("stage") for p in payloads]

    assert "one" in stages and "two" in stages
    assert payloads[-1]["status"] == "done"


def test_sse_frames_are_well_formed():
    registry = JobRegistry()
    job = registry.create("test")
    registry.run(job, lambda j, p: "ok")
    assert _wait_for(job, {"done"})

    for frame in sse_stream(job, registry):
        assert frame.endswith("\n\n"), "every SSE frame must end with a blank line"
        assert frame.startswith("data:") or frame.startswith(":")


def test_sse_survives_hebrew_in_progress_text():
    """Stage names carry track titles, which are routinely Hebrew."""
    registry = JobRegistry()
    job = registry.create("test")

    def target(j, progress):
        progress("analysing", 0.5, track="אושר כהן - כאפה")
        return "ok"

    registry.run(job, target)
    assert _wait_for(job, {"done"})

    frames = [f for f in sse_stream(job, registry) if f.startswith("data:")]
    decoded = [json.loads(f[len("data: "):]) for f in frames]
    assert any(p.get("track") == "אושר כהן - כאפה" for p in decoded)
