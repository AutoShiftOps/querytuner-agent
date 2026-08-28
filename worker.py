"""
Worker service — the only thing that ever calls run_pipeline(). Deployed
as a separate Cloud Run service from main.py, with its Cloud Run IAM
policy set to "no public invoker" so only Cloud Tasks' own service
account (via OIDC token) can reach it. The X-Task-Secret header check
below is defense in depth on top of that IAM restriction, not a
replacement for it — see README's deploy section for the actual IAM
binding command.

Not meant to be hit directly by end users; main.py never talks to this
service's URL except indirectly, by handing it to Cloud Tasks as the
push target.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import gemini_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QueryTuner Async Agent — worker")

TASK_SECRET = os.getenv("TASK_SECRET", "").strip()


class RunPipelineRequest(BaseModel):
    job_id: str


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.post("/tasks/run-pipeline")
async def run_pipeline_task(payload: RunPipelineRequest, x_task_secret: str | None = Header(default=None)) -> dict:
    """
    Cloud Tasks' HTTP push target. Verifies the shared secret (Cloud
    Tasks is configured to send it — see main.py's _enqueue_pipeline_task
    — and it's also set as this service's own env var) before doing any
    work, then runs the full 4-step pipeline synchronously within this
    request. Cloud Tasks itself enforces the request timeout / retry
    policy on its side (configured on the queue, see README) — an
    unhandled exception here surfaces as a non-2xx response, which is
    exactly what tells Cloud Tasks to retry per the queue's retry config.
    """
    if TASK_SECRET and x_task_secret != TASK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Task-Secret")

    logger.info("Running pipeline for job %s", payload.job_id)
    try:
        await gemini_agent.run_pipeline(payload.job_id)
    except KeyError:
        # No such job — don't ask Cloud Tasks to retry a job_id that will
        # never exist; a 4xx here is a permanent-failure signal, not a
        # transient one, so most Cloud Tasks retry configs won't retry it
        # (unlike a 5xx, which is exactly what an unexpected error in
        # run_pipeline's own try/except re-raise below should surface as).
        raise HTTPException(status_code=404, detail=f"No such job: {payload.job_id}") from None

    return {"ok": True, "job_id": payload.job_id}
