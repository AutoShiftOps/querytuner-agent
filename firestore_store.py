"""
Job/progress tracking for the async pipeline.

Backed by Firestore in real deployments (google-cloud-firestore, using
Cloud Run's Application Default Credentials — no service-account key
file needed when running on Cloud Run itself). Falls back to a
process-local in-memory store when DRY_RUN=true or when no
GOOGLE_CLOUD_PROJECT is configured, so main.py/worker.py/gemini_agent.py
are all fully exercisable on a laptop with zero GCP setup — genuinely
useful under a hackathon deadline, not just a toy: every step except the
two real Gemini/Gemma calls runs identically in dry-run and production,
including the entire real analysis engine (run_analyze_step never has a
"fake" mode — it's plain Python, no external dependency at all).

Job document shape (same fields, either backend):
{
  "job_id": str,
  "status": "queued" | "planning" | "triaging" | "analyzing" | "explaining" | "complete" | "failed",
  "created_at": iso8601 str,
  "updated_at": iso8601 str,
  "input": {...},               # whatever POST /jobs was given, normalized
  "progress": {"step": int, "total_steps": 4, "message": str},
  "plan": {...} | None,          # step 1 output
  "triage": {...} | None,        # step 2 output (Gemma)
  "analysis": {...} | None,      # step 3 output (real engine — run_analyze_step)
  "explanation": {...} | None,   # step 4 output (Gemini)
  "result": {...} | None,        # final combined report, set only on "complete"
  "error": str | None,
}
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from typing import Any

DRY_RUN = os.getenv("DRY_RUN", "").strip().lower() in ("1", "true", "yes")
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "queryagent_jobs")
_GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()

_TOTAL_STEPS = 4


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_job_doc(job_id: str, input_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "input": input_data,
        "progress": {"step": 0, "total_steps": _TOTAL_STEPS, "message": "Queued"},
        "plan": None,
        "triage": None,
        "analysis": None,
        "explanation": None,
        "result": None,
        "error": None,
    }


class _InMemoryStore:
    """Dry-run / local-dev backend — one process's dict, gone on restart.
    Same method surface as _FirestoreBackedStore below so gemini_agent.py
    never needs to know which one it's talking to."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def create_job(self, input_data: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = _new_job_doc(job_id, input_data)
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        return dict(job) if job is not None else None

    def update_job(self, job_id: str, **fields: Any) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"No such job: {job_id}")
        job.update(fields)
        job["updated_at"] = _now()

    def set_progress(self, job_id: str, step: int, message: str, status: str) -> None:
        self.update_job(
            job_id,
            status=status,
            progress={"step": step, "total_steps": _TOTAL_STEPS, "message": message},
        )


class _FirestoreBackedStore:
    """Real backend — google-cloud-firestore. Imported lazily so this
    module (and everything that imports it, including main.py at
    startup) doesn't hard-require the Firestore package or valid GCP
    credentials just to run in dry-run/local mode."""

    def __init__(self) -> None:
        from google.cloud import firestore  # local import — see class docstring

        self._client = firestore.Client(project=_GCP_PROJECT or None)
        self._collection = self._client.collection(FIRESTORE_COLLECTION)

    def create_job(self, input_data: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        doc = _new_job_doc(job_id, input_data)
        self._collection.document(job_id).set(doc)
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        snap = self._collection.document(job_id).get()
        return snap.to_dict() if snap.exists else None

    def update_job(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = _now()
        self._collection.document(job_id).update(fields)

    def set_progress(self, job_id: str, step: int, message: str, status: str) -> None:
        self.update_job(
            job_id,
            status=status,
            progress={"step": step, "total_steps": _TOTAL_STEPS, "message": message},
        )


def get_store() -> _InMemoryStore | _FirestoreBackedStore:
    """Single place that decides which backend to use — call this, don't
    instantiate either class directly, so DRY_RUN/GOOGLE_CLOUD_PROJECT
    only need checking in one place."""
    if DRY_RUN or not _GCP_PROJECT:
        return _in_memory_singleton()
    return _FirestoreBackedStore()


_singleton: _InMemoryStore | None = None


def _in_memory_singleton() -> _InMemoryStore:
    # A real Firestore collection is naturally one shared store across
    # requests; the in-memory fallback needs the same property (a job
    # created by one request must be readable by a later poll) — module-
    # level singleton, not a fresh dict per get_store() call.
    global _singleton
    if _singleton is None:
        _singleton = _InMemoryStore()
    return _singleton
