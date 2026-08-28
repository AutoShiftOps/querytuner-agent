"""
Public API — POST /jobs enqueues a Cloud Tasks task, GET /jobs/{id} polls
Firestore for progress/result. This service never runs the pipeline
itself (worker.py does, invoked by Cloud Tasks) — it only creates the
job record and enqueues the work, so a slow/large batch never ties up
a request here.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

import firestore_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="QueryTuner Async Agent",
    description=(
        "Async background agent: upload a slow-query log or batch export, "
        "a 4-step pipeline (plan -> triage with Gemma -> analyze -> explain "
        "with Gemini) runs it through QueryTuner's real heuristic + "
        "schema-aware analysis engine, poll for the finished report."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DRY_RUN = os.getenv("DRY_RUN", "").strip().lower() in ("1", "true", "yes")


# ── Request/response models ──────────────────────────────────────────────


class SingleQueryInput(BaseModel):
    mode: Literal["single"] = "single"
    query: str = Field(..., min_length=1, description="The SQL query to analyze")
    db_type: str = Field(default="postgresql")
    schema_info: str | None = Field(default=None, description="CREATE TABLE DDL, optional")
    explain_plan: str | None = Field(default=None, description="Pasted EXPLAIN output, optional")


class BatchInput(BaseModel):
    mode: Literal["batch"] = "batch"
    source: Literal["pg_stat_statements", "performance_schema", "query_store"]
    export_text: str = Field(..., min_length=1, description="Pasted production query export")
    schema_info: str | None = Field(default=None)
    top_n: int = Field(default=20, ge=1, le=200)

    @model_validator(mode="after")
    def _check_not_blank(self) -> "BatchInput":
        if not self.export_text.strip():
            raise ValueError("export_text is empty")
        return self


class CreateJobResponse(BaseModel):
    job_id: str
    status: str
    poll_url: str


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True, "dry_run": DRY_RUN}


@app.post("/jobs", response_model=CreateJobResponse)
async def create_job(payload: SingleQueryInput | BatchInput) -> CreateJobResponse:
    """
    Creates the Firestore job record, enqueues a Cloud Tasks task pointing
    at worker.py's /tasks/run-pipeline, and returns immediately — the
    caller polls GET /jobs/{id} for progress, it never blocks here
    waiting on the pipeline (that's the whole point of doing this async).
    """
    store = firestore_store.get_store()
    job_id = store.create_job(payload.model_dump())

    _enqueue_pipeline_task(job_id)

    return CreateJobResponse(job_id=job_id, status="queued", poll_url=f"/jobs/{job_id}")


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    store = firestore_store.get_store()
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")
    return job


# ── Cloud Tasks enqueue ───────────────────────────────────────────────────


def _enqueue_pipeline_task(job_id: str) -> None:
    """
    Real deployments: creates an HTTP push task in Cloud Tasks targeting
    worker.py's /tasks/run-pipeline endpoint, authenticated via an OIDC
    token Cloud Tasks attaches itself (WORKER_SERVICE_ACCOUNT) — that
    token is what worker.py's own IAM invoker policy checks, on top of
    the shared-secret header below as defense in depth. See README's
    "Deploying to Cloud Run + Cloud Tasks" section for the exact
    `gcloud tasks queues create` / IAM bindings this assumes.

    DRY_RUN / local dev: runs the pipeline in-process via FastAPI's
    BackgroundTasks-equivalent (a plain asyncio task) instead of making a
    real Cloud Tasks call, so `uvicorn main:app` alone — no worker
    service, no GCP project — is enough to exercise the whole pipeline
    locally. This is the one piece of orchestration that's genuinely
    different between dry-run and production (everything else in
    firestore_store.py and gemini_agent.py behaves identically either
    way) — real deployments must go through worker.py's own HTTP
    endpoint, not this in-process shortcut, so the split between "API
    service" and "worker service" (the whole point of the Cloud
    Tasks architecture, so worker crashes/OOMs don't take down the public
    API) is real in production.
    """
    if DRY_RUN:
        import asyncio

        import gemini_agent

        asyncio.create_task(gemini_agent.run_pipeline(job_id))
        return

    from google.cloud import tasks_v2

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ["CLOUD_TASKS_LOCATION"]
    queue = os.environ["CLOUD_TASKS_QUEUE"]
    worker_url = os.environ["WORKER_URL"].rstrip("/") + "/tasks/run-pipeline"
    worker_service_account = os.environ["WORKER_SERVICE_ACCOUNT"]
    task_secret = os.environ["TASK_SECRET"]

    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path(project, location, queue)

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": worker_url,
            "headers": {"Content-Type": "application/json", "X-Task-Secret": task_secret},
            "body": json.dumps({"job_id": job_id}).encode(),
            "oidc_token": {"service_account_email": worker_service_account},
        }
    }
    client.create_task(parent=parent, task=task)
    logger.info("Enqueued pipeline task for job %s", job_id)
