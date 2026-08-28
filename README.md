# QueryTuner Async Agent

An async background agent that turns QueryTuner's synchronous "paste one
query, get one answer" tool into a queue-driven pipeline for heavy
workloads: upload a slow-query export (or a single query) → Cloud Tasks
queues the job → a 4-step agent pipeline runs it through the real
analysis engine → Firestore tracks progress → poll for the finished
report.

**Google AI stack only, via the Agent Development Kit.** The only two
functions in this entire project that ever call an AI model are
`run_triage_step` (Gemma) and `run_explain_step` (Gemini) in
`gemini_agent.py` — both routed through `adk_agents.py`, which builds a
real `google.adk.agents.LlmAgent` and drives it with
`google.adk.runners.InMemoryRunner` (see that module's docstring for
exactly what was verified vs. not). No other AI SDK is imported anywhere
in this project — `pip list`-audit-able: `google-adk` and its own
`google-genai` dependency are the only AI packages in `requirements.txt`.
The actual analysis logic (`analysis_engine/`) is deterministic
heuristics + schema/EXPLAIN-plan cross-referencing with zero AI calls of
its own; see "Where the analysis engine came from" below.

Two auth paths, both native GCP: the Gemini Developer API (an AI Studio
key — simplest to hand a judge for a quick spin-up) or Vertex AI (no key
at all — Cloud Run's own attached service account authenticates via
Application Default Credentials, the more "native GCP" of the two). See
`.env.example` and "Deploying" below for both.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
  POST /jobs        │  main.py — public API (Cloud Run, public)    │
  ───────────────►  │  • validates input, creates Firestore job    │
                     │  • enqueues a Cloud Tasks push task          │
  ◄───────────────  │  • returns {job_id} immediately               │
  {job_id}           └───────────────────┬───────────────────────────┘
                                          │ HTTP push (OIDC-authenticated)
                                          ▼
                     ┌─────────────────────────────────────────────┐
                     │  worker.py — pipeline runner (Cloud Run,     │
                     │  no public invoker — Cloud Tasks' service    │
                     │  account only)                               │
                     │  POST /tasks/run-pipeline {job_id}            │
                     └───────────────────┬───────────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────────┐
                     │  gemini_agent.py — run_pipeline(job_id)      │
                     │                                               │
                     │  1. plan     — Gemini (cheap model, best-     │
                     │                effort strategy rationale)     │
                     │  2. triage   — Gemma, priority per query,     │
                     │                BEFORE the real analysis pass  │
                     │  3. analyze  — analysis_engine/ (real         │
                     │                heuristics + schema/EXPLAIN    │
                     │                cross-ref, zero AI calls)      │
                     │  4. explain  — Gemini, executive summary       │
                     │                weighted by triage priority     │
                     │                                               │
                     │  steps 1/2/4 all go through adk_agents.py —   │
                     │  a real ADK LlmAgent + InMemoryRunner per      │
                     │  call, not a raw SDK call (see adk_agents.py)  │
                     │                                               │
                     │  progress written to Firestore after every    │
                     │  step, not just at the end                    │
                     └───────────────────┬───────────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────────┐
                     │  Firestore — job doc: status, progress,      │
                     │  plan/triage/analysis/explanation, result     │
                     └─────────────────────────────────────────────┘
                                          ▲
  GET /jobs/{id}     ┌─────────────────────────────────────────────┐
  ───────────────►   │  main.py polls the same Firestore doc         │
  ◄───────────────   └─────────────────────────────────────────────┘
  {status, progress, result}
```

Two Cloud Run services, one Firestore collection, one Cloud Tasks queue.
`main.py` never runs the pipeline itself — it only enqueues — so a
large batch never ties up a request thread on the public-facing service.

## Where the analysis engine came from

`analysis_engine/` is vendored from [AutoShiftOps/querytuner](https://github.com/AutoShiftOps/querytuner)
— the same product this project builds on top of — specifically:
`sql_analyzer.py`, `optimizer.py`, `explainer.py`, `index_recommender.py`,
`query_parser.py`, `dialect_config.py`, `plan_crossref.py`,
`plan_parsers/`, `batch_parsers.py`, `batch_reconciler.py`, and a
paste-in-only trim of `collectors/`. This is real, tested, already-shipped
analysis code — heuristic SQL pattern detection, schema-aware index
recommendations, EXPLAIN-plan cross-referencing, and cross-query batch
reconciliation — not a reimplementation for this submission.

One deliberate change from the source: **every AI-provider code path was
removed, not just left unused.** QueryTuner's production `sql_analyzer.py`
optionally calls OpenAI or Hugging Face for a supplementary "AI insights"
pass; that entire branch (`_build_llm_prompt`, `_try_llm`, the
`use_llm`/`llm_provider` parameters) was deleted from the vendored copy
here, along with the `LLMProvider` schema and the `llm/router.py`
dependency it needed. `run_analyze_step` in `gemini_agent.py` — the
function this build's actual task was to wire up — now calls this
trimmed engine, so there's no AI-provider code anywhere in the analysis
path to audit away; the Google-AI-stack-only constraint holds by
construction, not by convention. The `collectors/` package was similarly
trimmed to the pasted-EXPLAIN-plan path only (no live-DSN database
connections) — this project's positioning, like QueryTuner's own, is
"paste your export, no live database connection required."

## Running locally with zero GCP setup (`DRY_RUN=true`)

```bash
cp .env.example .env
# edit .env: leave DRY_RUN=true, ignore everything else

pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

In dry-run mode, `main.py` runs the full pipeline in-process (no worker
service, no real Cloud Tasks call) and `gemini_agent.py`'s two AI calls
return clearly-labeled canned text instead of hitting the real API.
**Every other part of the pipeline — Firestore job tracking (via an
in-memory fallback store) and the entire real analysis engine — behaves
identically to production.** This was verified end-to-end before writing
this README: both a single-query job and a batch (`pg_stat_statements`
export, 2 queries) run through `POST /jobs` → poll `GET /jobs/{id}` →
`status: "complete"` with real heuristic findings, real reconciled index
suggestions, and real DDL in the response.

The canned fallback text for the plan/triage/explain steps is worded
accurately for *why* it's canned, not just labeled generically as
"dry-run": `gemini_agent._fallback_reason()` distinguishes three cases —
true `DRY_RUN` mode (no call attempted at all), no credentials configured
(same effect, different reason), and a real, credentialed ADK call that
was genuinely attempted and failed (bad key, rate limit, network/egress
issue) — the last case says so explicitly ("call was attempted but
failed — see server logs") rather than misleadingly implying no attempt
was made. Verified directly: setting a real-looking-but-invalid
`GEMINI_API_KEY` (no `DRY_RUN`) drives the pipeline through an actual ADK
→ `google-genai` → outbound HTTP call that fails, and the job result
correctly reflects "attempted but failed" wording, not "dry-run" wording.

Single query:
```bash
curl -X POST http://localhost:8080/jobs -H "Content-Type: application/json" -d '{
  "mode": "single",
  "query": "SELECT * FROM orders WHERE status = '"'"'pending'"'"' AND YEAR(created_at) = 2026",
  "db_type": "mysql",
  "schema_info": "CREATE TABLE orders (id INT PRIMARY KEY, status VARCHAR(20), created_at DATETIME);"
}'
# -> {"job_id": "...", "status": "queued", "poll_url": "/jobs/..."}

curl http://localhost:8080/jobs/<job_id>
```

Batch (`pg_stat_statements` export):
```bash
curl -X POST http://localhost:8080/jobs -H "Content-Type: application/json" -d '{
  "mode": "batch",
  "source": "pg_stat_statements",
  "export_text": "query|calls|total_time\nSELECT * FROM orders WHERE customer_id = 5|100|5000\nSELECT * FROM orders WHERE customer_id = 5 AND status = '"'"'pending'"'"'|50|3000\n",
  "top_n": 20
}'
```

## Demo UI (`streamlit_app.py`)

The API above is the real product surface, but curl isn't much of a demo.
`streamlit_app.py` is a thin browser UI over the same `POST /jobs` /
`GET /jobs/{id}` calls — it renders whatever the API actually returns
(triage priorities color-coded, reconciled index suggestions as
copy-pasteable `CREATE INDEX` statements, the executive summary,
progress bar while polling), nothing is mocked for it. It's a demo/local
convenience only — kept out of `requirements.txt` and the Docker image on
purpose, in its own `requirements-streamlit.txt`.

```bash
# in one terminal — the real API, same as above
uvicorn main:app --reload --port 8080

# in a second terminal
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. To point it at a deployed API instead
of localhost, set `API_BASE_URL` (or just type the URL into the sidebar
field once the page is open):
```bash
API_BASE_URL=https://queryagent-api-xxxx-uc.a.run.app streamlit run streamlit_app.py
```

Verified headlessly in this environment via Streamlit's own `AppTest`
framework (no browser needed to check for runtime errors): both single-
query and batch mode submit, poll, and render a complete result with
zero exceptions, against a real `DRY_RUN=true` API instance — the same
discipline used throughout this README, not just "should work."

## Deploying to Cloud Run + Cloud Tasks

```bash
PROJECT=your-gcp-project-id
REGION=us-central1

# 1. Build once, push once, deploy from the same image twice.
gcloud builds submit --tag gcr.io/$PROJECT/queryagent

# 2. Firestore (Native mode, if not already provisioned).
gcloud firestore databases create --location=$REGION

# 3. A dedicated service account Cloud Tasks uses to authenticate to
#    the worker service (matches WORKER_SERVICE_ACCOUNT in .env).
gcloud iam service-accounts create queryagent-tasks-invoker

# 4. Worker service — deployed WITHOUT --allow-unauthenticated. This is
#    what actually enforces "only Cloud Tasks can call this," not the
#    X-Task-Secret header alone (that's defense in depth on top of it).
gcloud run deploy queryagent-worker \
  --image gcr.io/$PROJECT/queryagent \
  --region $REGION \
  --no-allow-unauthenticated \
  --command python3 \
  --args="-m,uvicorn,worker:app,--host,0.0.0.0,--port,8080" \
  --set-env-vars TASK_SECRET=<same-value-as-.env>

gcloud run services add-iam-policy-binding queryagent-worker \
  --region $REGION \
  --member "serviceAccount:queryagent-tasks-invoker@$PROJECT.iam.gserviceaccount.com" \
  --role "roles/run.invoker"

# 5. Cloud Tasks queue.
gcloud tasks queues create queryagent-pipeline --location=$REGION

# 6. Public API service — this one IS publicly invokable.
#    Auth option A (Gemini Developer API key) shown below; see the
#    Vertex AI alternative right after this block for the more "native
#    GCP" no-API-key path.
gcloud run deploy queryagent-api \
  --image gcr.io/$PROJECT/queryagent \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT,CLOUD_TASKS_LOCATION=$REGION,CLOUD_TASKS_QUEUE=queryagent-pipeline,WORKER_URL=<queryagent-worker's URL from step 4>,WORKER_SERVICE_ACCOUNT=queryagent-tasks-invoker@$PROJECT.iam.gserviceaccount.com,TASK_SECRET=<same-value-as-worker>,GEMINI_API_KEY=<your key>
```

The service account Cloud Run assigns each service (its default compute
service account, unless overridden) needs `roles/datastore.user` for
Firestore access and `roles/cloudtasks.enqueuer` for the API service to
create tasks — grant both via `gcloud projects add-iam-policy-binding`
if not already present in your project's default permissions.

### Stronger "native GCP" option — Vertex AI instead of an API key

Set `GOOGLE_GENAI_USE_VERTEXAI=true` and `GOOGLE_CLOUD_LOCATION=$REGION`
in place of `GEMINI_API_KEY` on both Cloud Run services — the attached
service account's Application Default Credentials authenticate to Gemini
directly through Vertex AI, no separate secret to manage at all. Needs
`roles/aiplatform.user` on the service account and the Vertex AI API
enabled (`gcloud services enable aiplatform.googleapis.com`). This is the
recommended path for the actual hackathon submission over the API-key
path — it swaps out the last plain secret in the deploy and leans on
IAM/ADC the same way Firestore and Cloud Tasks access already does here.

### Optional hardening — Secret Manager for `TASK_SECRET`

`TASK_SECRET` and (if using the API-key path) `GEMINI_API_KEY` are the
only two secrets in this project passed as plain `--set-env-vars` above.
For a stronger submission, mount them from Secret Manager instead:

```bash
echo -n "<same-value-as-worker>" | gcloud secrets create queryagent-task-secret --data-file=-
gcloud secrets add-iam-policy-binding queryagent-task-secret \
  --member "serviceAccount:<cloud-run-service-account>" --role "roles/secretmanager.secretAccessor"
# then, on each `gcloud run deploy`, replace --set-env-vars TASK_SECRET=... with:
#   --set-secrets TASK_SECRET=queryagent-task-secret:latest
```

**Not build-tested in this environment** — the Dockerfile follows a
standard `python:3.12-slim` + `pip install -r requirements.txt` pattern
and every module it copies has been directly verified importable and
runnable (see above), but no Docker daemon was available here to
actually run `docker build`. Worth a real `docker build && docker run`
pass before the hosted-URL submission deadline, not just trusting the
Dockerfile's shape.

## Publishing as a public repo

```bash
cd querytuner-agent
git init
git add -A
git commit -m "QueryTuner Async Agent — Google ADK hackathon submission"
gh repo create <your-username>/querytuner-agent --public --source=. --push
# or: git remote add origin https://github.com/<your-username>/querytuner-agent.git
#     git branch -M main && git push -u origin main
```

Nothing in this project needs to stay private — `.env.example` has no
real secrets, `.env` itself is `.gitignore`d (add one with `.env`,
`__pycache__/`, `*.pyc` if it isn't present), and every dependency is a
public PyPI package pinned to a real version.

See `SUBMISSION.md` for the written features/tech/learnings summary the
submission form asks for — it's ready to paste in or link to as-is.

## API

- `POST /jobs` — body is either a single-query request (`mode: "single"`)
  or a batch request (`mode: "batch"`, `source` one of
  `pg_stat_statements` / `performance_schema` / `query_store`). Returns
  `{job_id, status, poll_url}` immediately.
- `GET /jobs/{job_id}` — full job document: `status`
  (`queued`→`planning`→`triaging`→`analyzing`→`explaining`→`complete`/`failed`),
  `progress` (`{step, total_steps, message}`), and once complete, `result`
  containing all four steps' output.
- `GET /healthz` — on both `main.py` and `worker.py`.

## What's genuinely real here vs. what's demo scaffolding

Real, tested, verified end-to-end (see "Running locally" above): the
entire analysis engine, batch parsing/ranking/reconciliation, the
Firestore job-tracking contract, the plan/triage/analyze/explain step
sequence and its progress reporting, and the DRY_RUN fallback path.

Also verified directly in this environment, without a live Gemini/Vertex
credential: that `adk_agents.py`'s `LlmAgent` + `InMemoryRunner` +
`InMemorySessionService` construct and wire together correctly against
the installed `google-adk==2.8.0`, and that `run_async()` correctly
drives the full ADK flow up to the literal outbound HTTP call — confirmed
by two distinct expected failure points (no API key at all → `ValueError:
No API key was provided`; a set-but-invalid key → an actual outbound
HTTP attempt that fails only on this sandbox's own network egress
policy, not a code defect). Same technique proved `gemini_agent.py`'s new
`_fallback_reason()` wording is accurate in both the true-`DRY_RUN` case
and the real-call-attempted-and-failed case (see "Running locally"
above).

Not yet exercised against a real GCP project from this environment: the
actual Cloud Tasks enqueue call, real Firestore writes, a real live
Gemini/Gemma model response (only a real *attempt* was verified, not a
successful round trip), and the Cloud Run IAM/OIDC authentication chain
between the two services. The code for all of it is written against the
real `google-cloud-firestore` / `google-cloud-tasks` / `google-adk` /
`google-genai` SDKs (not stubbed interfaces) and every import resolves
cleanly, but a real deploy plus a smoke test against a real
`GEMINI_API_KEY` (or Vertex AI project) is the next step before the
submission's hosted-URL and demo video requirements are met.

## Deploy/test/evaluate checklist

1. `pip install -r requirements.txt`, `cp .env.example .env`, leave
   `DRY_RUN=true` — run the single-query and batch curl examples above,
   confirm both return `status: "complete"` with real findings.
2. Get a key at https://aistudio.google.com/apikey, set
   `GEMINI_API_KEY=<key>` in `.env`, set `DRY_RUN=false`, re-run the same
   two curl examples — confirm `plan.rationale`, `triage[].reason`, and
   `explanation.summary` now contain real model output instead of
   fallback text (fallback wording, if it appears, will say "call was
   attempted but failed" — check server logs for why, most likely an
   invalid/rate-limited key or a network/firewall block on your side).
3. `docker build -t queryagent .` then `docker run` both services
   locally with the same `.env` (was not build-tested in this sandbox —
   see above) — confirm the image builds and both `/healthz` endpoints
   respond.
4. Follow "Deploying to Cloud Run + Cloud Tasks" above for a real GCP
   project; use the Vertex AI variant instead of `GEMINI_API_KEY` for the
   strongest native-GCP story if pursuing that for judging.
5. Confirm `GET /jobs/{job_id}` polling works end-to-end against the
   deployed URL.
6. Point `streamlit_app.py` at that URL (`API_BASE_URL=<deployed url>
   streamlit run streamlit_app.py`) and use the browser UI — not curl —
   for the submission's demo video.
