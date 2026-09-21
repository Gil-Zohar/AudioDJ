"""Background job registry with live progress over SSE.

Rendering a mix takes minutes, mostly in stem separation, so the HTTP request
that starts it must return immediately. Jobs run on a worker thread and publish
progress events that the browser follows via EventSource.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Optional

log = logging.getLogger("autodj.jobs")

JobStatus = Literal["pending", "running", "done", "failed", "cancelled"]

# Keep finished jobs around briefly so a page refresh can still read the result.
RETENTION_SECONDS = 3600


@dataclass
class JobEvent:
    stage: str
    progress: float
    at: float = field(default_factory=time.time)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "at": self.at,
            **self.detail,
        }


@dataclass
class Job:
    id: str
    kind: str
    status: JobStatus = "pending"
    progress: float = 0.0
    stage: str = "queued"
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    events: list[JobEvent] = field(default_factory=list)
    _subscribers: list[queue.Queue] = field(default_factory=list, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "status": self.status,
            "progress": round(self.progress, 4), "stage": self.stage,
            "error": self.error, "created_at": self.created_at,
            "finished_at": self.finished_at,
            "result": self.result if self.status == "done" else None,
        }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, kind: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._prune()
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def _prune(self) -> None:
        cutoff = time.time() - RETENTION_SECONDS
        stale = [
            job_id for job_id, job in self._jobs.items()
            if job.finished_at and job.finished_at < cutoff
        ]
        for job_id in stale:
            self._jobs.pop(job_id, None)

    # -- running ----------------------------------------------------------
    def run(self, job: Job, target: Callable[[Job, Callable], Any]) -> Job:
        """Run `target(job, progress)` on a worker thread."""

        def publish(stage: str, progress: float, **detail) -> None:
            if job.cancelled:
                raise JobCancelled(job.id)
            job.stage = stage
            job.progress = max(0.0, min(1.0, progress))
            event = JobEvent(stage=stage, progress=job.progress, detail=detail)
            job.events.append(event)
            self._broadcast(job, event.to_dict())

        def worker() -> None:
            job.status = "running"
            self._broadcast(job, {"stage": "started", "progress": 0.0})
            try:
                job.result = target(job, publish)
                job.status = "done"
                job.progress = 1.0
                job.stage = "done"
            except JobCancelled:
                job.status = "cancelled"
                job.stage = "cancelled"
                log.info("job %s cancelled", job.id)
            except Exception as exc:  # noqa: BLE001
                job.status = "failed"
                job.error = str(exc)
                job.stage = "failed"
                log.exception("job %s failed", job.id)
                log.debug(traceback.format_exc())
            finally:
                job.finished_at = time.time()
                self._broadcast(job, {
                    "stage": job.stage, "progress": job.progress,
                    "status": job.status, "error": job.error, "final": True,
                })

        threading.Thread(target=worker, name=f"job-{job.id}", daemon=True).start()
        return job

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in ("done", "failed", "cancelled"):
            return False
        job._cancel.set()
        return True

    # -- subscriptions ----------------------------------------------------
    def subscribe(self, job: Job) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=256)
        job._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, job: Job, subscriber: queue.Queue) -> None:
        if subscriber in job._subscribers:
            job._subscribers.remove(subscriber)

    def _broadcast(self, job: Job, payload: dict) -> None:
        for subscriber in list(job._subscribers):
            try:
                subscriber.put_nowait(payload)
            except queue.Full:
                # A stalled browser tab must not block the render.
                pass


class JobCancelled(Exception):
    pass


_registry: Optional[JobRegistry] = None


def get_registry() -> JobRegistry:
    global _registry
    if _registry is None:
        _registry = JobRegistry()
    return _registry


def sse_stream(job: Job, registry: JobRegistry) -> Iterator[str]:
    """Yield Server-Sent Events for a job until it finishes.

    Replays what already happened before subscribing, so a browser that connects
    late still sees the full picture rather than starting mid-render.
    """
    import json

    subscriber = registry.subscribe(job)
    try:
        for event in list(job.events):
            yield f"data: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"

        if job.status in ("done", "failed", "cancelled"):
            yield f"data: {json.dumps(job.to_dict(), ensure_ascii=False)}\n\n"
            return

        while True:
            try:
                payload = subscriber.get(timeout=15)
            except queue.Empty:
                # Comment frame: keeps proxies from closing an idle connection.
                yield ": keepalive\n\n"
                continue

            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if payload.get("final"):
                yield f"data: {json.dumps(job.to_dict(), ensure_ascii=False)}\n\n"
                return
    finally:
        registry.unsubscribe(job, subscriber)
