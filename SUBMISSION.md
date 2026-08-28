# QueryTuner Async Agent — Hackathon Submission Write-Up

## Problem

Query performance review tools are built for one query at a time — paste
a query, get an answer. Real production incidents don't look like that:
an on-call engineer has a slow-query log or a `pg_stat_statements` export
with dozens or hundreds of candidates, and no tool to triage which ones
actually matter before spending real analysis time on each. This project
turns that into a background job: upload the whole batch once, get a
prioritized, cross-referenced report back.

## What it does

1. **Submit** a single query or a batch export (`pg_stat_statements`,
   `performance_schema`, or `query_store` format) to `POST /jobs`. The
   call returns a `job_id` immediately — the caller never waits on the
   pipeline.
2. **Plan** — a lightweight Gemini call writes a one-sentence strategy
   rationale for the job (best-effort; the deterministic plan underneath
   it never depends on this succeeding).
3. **Triage** — Gemma looks at every candidate query's raw text, before
   any real analysis runs, and assigns a fast, cheap priority (high /
   medium / low). This happens first so the expensive step later knows
   where to spend its attention.
4. **Analyze** — the real work: a vendored, deterministic heuristic +
   schema-aware analysis engine (see "Where the analysis engine came
   from" below) parses every query, cross-references it against
   optional schema DDL and EXPLAIN plans, and — in batch mode —
   reconciles per-query index suggestions into one coherent,
   deduplicated recommendation set across the whole batch, instead of N
   independent, possibly-conflicting answers.
5. **Explain** — Gemini turns the structured findings into an
   executive summary, weighted by the triage priorities from step 3: full
   explanation for high-priority findings, a line or a skip for
   low-priority ones — the same way a human reviewer would actually
   spend their attention on a real batch.
6. **Poll** `GET /jobs/{job_id}` for live progress and, once complete,
   the full structured result of all four steps.

Every step's progress is written to Firestore as it happens, not just at
the end, so a caller polling mid-run sees real state
(`queued → planning → triaging → analyzing → explaining → complete`),
not a black box.

## Why this fits the theme

Background agent, not a synchronous request/response tool: the public
API (`main.py`) only ever validates input, writes a job doc, and enqueues
a Cloud Tasks push task — it never runs the pipeline itself, so a large
batch never ties up a request thread on the public-facing service. The
actual pipeline runs on a second, IAM-locked-down worker service
(`worker.py`) that only Cloud Tasks' own service account can invoke.
That queue/worker split is what makes "handle heavy datasets
asynchronously" true at the infrastructure level, not just in how the
code is organized.

## Technical stack

- **Cloud Run** — two services from one image (`main.py` public API,
  `worker.py` IAM-restricted pipeline runner), CMD override at deploy
  time selects which.
- **Cloud Tasks** — the queue between them; OIDC-authenticated push
  tasks, so only the intended worker can be invoked.
- **Firestore** — job/progress document store (native mode).
- **Google Agent Development Kit (`google-adk`)** — every AI call in the
  project goes through a real `LlmAgent` + `InMemoryRunner`
  (`adk_agents.py`), not a raw SDK call. Two of the four pipeline steps
  use it directly for structured JSON output; the third (analyze) is
  deliberately AI-free — see below.
- **Gemini / Gemma** — `gemini-2.0-flash-lite` (plan), `gemma-3-27b-it`
  (triage), `gemini-2.0-flash` (explain), all swappable via env vars,
  including pointing the triage model at a literal Gemma endpoint served
  via Vertex AI Model Garden with no code change.
- **google-genai** — ADK's own underlying Gemini/Vertex client; supports
  both the Gemini Developer API (an AI Studio key) and Vertex AI (ADC,
  no key at all) without any code branching — same `LlmAgent` code either
  way, only environment configuration differs.

**Google AI stack only, provably so:** `google-adk` and its own
`google-genai` dependency are the *only* AI packages in
`requirements.txt` — auditable with a plain `pip list`, not just a
promise in prose.

## Where the analysis engine came from

The step this hackathon build actually had to wire up — `run_analyze_step`
— runs a real, already-shipped analysis engine vendored from
[AutoShiftOps/querytuner](https://github.com/AutoShiftOps/querytuner), the
production SaaS product this project builds on top of: heuristic SQL
pattern detection, schema-aware index recommendations, EXPLAIN-plan
cross-referencing, and cross-query batch reconciliation — not code
written fresh for the submission. One deliberate change from the source:
every existing AI-provider code path (an optional OpenAI/Hugging Face
"AI insights" pass in the original) was *removed*, not just left unused,
so there's no non-Google AI code anywhere in the analysis path to audit
away — the constraint holds by construction.

## What's genuinely verified vs. what still needs a live deploy

Verified end-to-end in a sandbox with no GCP project and no real
Gemini/Vertex credentials, via a `DRY_RUN=true` mode that runs the real
analysis engine and real Firestore-shaped job tracking (in-memory
fallback) while only the two AI calls return labeled fallback text: both
single-query and batch jobs complete with real heuristic findings, real
reconciled index suggestions, and real DDL cross-referencing in the
response.

Also verified, without a live credential: that the ADK integration itself
— `LlmAgent`, `InMemoryRunner`, `InMemorySessionService`, `run_async()` —
constructs correctly and drives a real request up to the literal outbound
HTTP call (confirmed via two distinct, expected failure points: no API
key at all, and a set-but-invalid key that fails only at the network
hop). And that a real, credentialed call which fails for some other
reason (bad key, rate limit) is reported as "attempted but failed," never
confused with true dry-run mode.

Not yet exercised from this environment: an actual successful live
Gemini/Gemma response, a real Cloud Tasks enqueue, real Firestore writes,
and the Cloud Run IAM/OIDC chain between the two services — these need a
real GCP project, which is the next step before this write-up's numbers
become "tested in production" rather than "verified by construction."

## Challenges and learnings

- **Getting the step order right.** The brief specifies
  plan → triage → analyze → explain; an earlier draft ran triage after
  the real analysis pass, which defeats the point of triage as a cheap
  pre-filter for where the expensive step should spend attention.
  Fixed by extracting a cheap, analysis-free candidate-listing step that
  runs before the real engine, so triage genuinely sees only raw query
  text, before any real analysis has happened.
- **"Google-only" is easy to claim, harder to prove.** The initial build
  used a raw `google-generativeai` client call, which is real Gemini
  usage but not literally the Agent Development Kit the brief calls out.
  Migrating to `LlmAgent` + `InMemoryRunner` was a deliberate, later
  change specifically to make "via Agent Builder/ADK" true in the literal
  sense a judge would check for, not just directionally true.
- **A fallback path can accidentally lie.** The dry-run fallback text
  originally said "dry-run — call skipped" unconditionally, including
  when a real, credentialed call had actually been attempted and failed.
  Caught via direct testing rather than code review — worth calling out
  because it's the kind of bug that only shows up when you actually run
  the failure path, not when you read the code.
- **Vendoring vs. calling the live API.** Chose to vendor (copy) the
  analysis engine's source into this project rather than calling the
  live QueryTuner API, specifically because that endpoint is
  subscription-gated in production, an external network dependency is a
  real risk during judging, and a vendored copy is what actually lets a
  judge read the analysis logic instead of trusting a black-box API call.

## What's next

This was deliberately built and scoped as a standalone project, not a
patch to the live QueryTuner product, specifically to avoid coupling an
experimental async/agent architecture to a product with paying users on
a different release cadence. If it proves out, the natural path is an
enterprise-tier feature on the live product: the vendored
`analysis_engine/` here is already the same code path QueryTuner's own
`/analyze/batch` endpoint uses, so re-pointing this pipeline at the live
API (instead of a vendored copy) — or, inversely, upstreaming this
project's queue/worker pattern into the main product behind a feature
flag — are both realistic integration paths once the architecture is
validated independently, rather than something decided under hackathon
time pressure.
