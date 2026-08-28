"""
Streamlit demo UI for the QueryTuner Async Agent — a thin client over the
real API (main.py), built purely to make the async pipeline demoable in a
browser instead of via curl. This file makes zero pipeline decisions of
its own: it POSTs to /jobs, polls GET /jobs/{id}, and renders whatever
comes back. Nothing here talks to Firestore, Gemini, or the analysis
engine directly — that separation is deliberate, so this stays a pure
demo/showcase layer on top of the real service, not a second code path
that could drift from it.

Not part of the deployed Cloud Run image (see Dockerfile / .dockerignore)
— this is a local/demo-only convenience, run with:
    pip install -r requirements-streamlit.txt
    streamlit run streamlit_app.py
against a `uvicorn main:app` you already have running (locally in
DRY_RUN mode, or pointed at a deployed API_BASE_URL).
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080").rstrip("/")

STEP_ORDER = ["queued", "planning", "triaging", "analyzing", "explaining", "complete"]

SAMPLE_SINGLE_QUERY = "SELECT * FROM orders WHERE status = 'pending' AND YEAR(created_at) = 2026"
SAMPLE_SCHEMA = (
    "CREATE TABLE orders (\n"
    "  id INT PRIMARY KEY,\n"
    "  customer_id INT,\n"
    "  status VARCHAR(20),\n"
    "  created_at DATETIME\n"
    ");"
)
SAMPLE_BATCH_EXPORT = (
    "query|calls|total_time\n"
    "SELECT * FROM orders WHERE customer_id = 5|100|5000\n"
    "SELECT * FROM orders WHERE customer_id = 5 AND status = 'pending'|50|3000\n"
    "SELECT o.*, c.name FROM orders o JOIN customers c ON o.customer_id = c.id "
    "WHERE o.status = 'shipped'|30|9000\n"
)

st.set_page_config(page_title="QueryTuner Async Agent", page_icon="⚡", layout="wide")


# ── Sidebar: API connection + at-a-glance architecture note ─────────────────

with st.sidebar:
    st.title("⚡ QueryTuner Agent")
    st.caption("Async, Google-AI-stack-only SQL analysis pipeline")

    api_base = st.text_input("API base URL", value=API_BASE_URL)
    try:
        health = requests.get(f"{api_base}/healthz", timeout=3).json()
        dry_run = health.get("dry_run", True)
        st.success(f"Connected — DRY_RUN={dry_run}")
    except Exception:
        st.error("Can't reach the API. Is `uvicorn main:app` running?")
        dry_run = True

    st.divider()
    st.markdown(
        "**Pipeline:** plan → triage (Gemma) → analyze (real heuristic + "
        "schema-aware engine) → explain (Gemini)\n\n"
        "This page is a thin client over the real `POST /jobs` / "
        "`GET /jobs/{id}` API — every result shown here is the actual "
        "pipeline output, not mocked for the demo."
    )
    if dry_run:
        st.info(
            "DRY_RUN mode: the real analysis engine runs for real. "
            "The plan/triage/explanation text is clearly-labeled fallback "
            "text since no Gemini call is made. Set `GEMINI_API_KEY` and "
            "`DRY_RUN=false` on the API to see live model output here."
        )


# ── Input form ────────────────────────────────────────────────────────────

st.header("Submit a job")

mode = st.radio("Mode", ["single", "batch"], horizontal=True)

if mode == "single":
    col1, col2 = st.columns(2)
    with col1:
        query = st.text_area("SQL query", value=SAMPLE_SINGLE_QUERY, height=100)
        db_type = st.selectbox("Database type", ["mysql", "postgresql", "sqlserver"])
    with col2:
        schema_info = st.text_area("Schema DDL (optional)", value=SAMPLE_SCHEMA, height=100)
        explain_plan = st.text_area("EXPLAIN plan (optional)", value="", height=68)

    payload: dict[str, Any] = {
        "mode": "single",
        "query": query,
        "db_type": db_type,
        "schema_info": schema_info or None,
        "explain_plan": explain_plan or None,
    }
else:
    source = st.selectbox("Export source", ["pg_stat_statements", "performance_schema", "query_store"])
    export_text = st.text_area("Pasted export", value=SAMPLE_BATCH_EXPORT, height=150)
    top_n = st.slider("Top N queries to analyze", min_value=1, max_value=50, value=20)
    schema_info = st.text_area("Schema DDL (optional)", value=SAMPLE_SCHEMA, height=100)

    payload = {
        "mode": "batch",
        "source": source,
        "export_text": export_text,
        "top_n": top_n,
        "schema_info": schema_info or None,
    }

submitted = st.button("Run analysis", type="primary")


# ── Submit + poll ─────────────────────────────────────────────────────────

if submitted:
    try:
        resp = requests.post(f"{api_base}/jobs", json=payload, timeout=10)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
    except Exception as e:
        st.error(f"Failed to create job: {e}")
        st.stop()

    st.success(f"Job created: `{job_id}`")

    progress_bar = st.progress(0.0, text="Queued...")
    status_placeholder = st.empty()

    job: dict[str, Any] = {}
    for _ in range(120):  # ~60s at 0.5s polling — plenty for DRY_RUN or a real call
        try:
            job = requests.get(f"{api_base}/jobs/{job_id}", timeout=10).json()
        except Exception as e:
            status_placeholder.warning(f"Poll failed, retrying: {e}")
            time.sleep(0.5)
            continue

        status = job.get("status", "queued")
        step_idx = STEP_ORDER.index(status) if status in STEP_ORDER else 0
        pct = min(step_idx / (len(STEP_ORDER) - 1), 1.0)
        message = job.get("progress", {}).get("message", status)
        progress_bar.progress(pct, text=f"{status} — {message}")

        if status in ("complete", "failed"):
            break
        time.sleep(0.5)

    if job.get("status") == "failed":
        st.error(f"Job failed: {job.get('error', 'unknown error')}")
        st.stop()
    elif job.get("status") != "complete":
        st.warning("Still running after the demo poll window — check `GET /jobs/{job_id}` directly.")
        st.json(job)
        st.stop()

    result = job.get("result", {})
    plan = result.get("plan", {})
    triage = result.get("triage", [])
    analysis = result.get("analysis", {})
    explanation = result.get("explanation", {})

    st.divider()
    st.header("Result")

    # ── Plan ──
    with st.expander("1. Plan", expanded=False):
        st.write(plan.get("rationale", "—"))
        st.json(plan)

    # ── Triage ──
    with st.expander(f"2. Triage — {len(triage)} candidate(s)", expanded=False):
        for t in triage:
            priority = t.get("priority", "medium")
            color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
            st.write(f"{color} **#{t.get('index')}** — {priority} — {t.get('reason', '')}")

    # ── Explanation (the headline result) ──
    st.subheader("Executive summary")
    st.markdown(explanation.get("summary", "—"))
    top_actions = explanation.get("top_actions", [])
    if top_actions:
        st.markdown("**Top actions:**")
        for a in top_actions:
            st.markdown(f"- {a}")

    # ── Analysis findings ──
    st.subheader("Analysis findings")
    if analysis.get("mode") == "batch":
        st.metric("Queries analyzed", analysis.get("analyzed_count", 0))
        reconciled = analysis.get("reconciled_index_suggestions", [])
        st.markdown(f"**{len(reconciled)} reconciled index suggestion(s)** (deduplicated across the whole batch)")
        for r in reconciled:
            st.code(
                f"CREATE INDEX ON {r.get('table')} ({', '.join(r.get('columns', []))})  "
                f"-- satisfies {len(r.get('satisfies_queries', []))} quer(y/ies)",
                language="sql",
            )
        with st.expander("Per-query breakdown"):
            st.json(analysis.get("queries", []))
        dropped = analysis.get("dropped_suggestions", [])
        if dropped:
            with st.expander(f"Dropped/superseded suggestions ({len(dropped)})"):
                st.json(dropped)
    else:
        suggestions = analysis.get("optimization_suggestions", [])
        col1, col2 = st.columns(2)
        col1.metric("Suggestions", len(suggestions))
        col2.metric("Readability score", analysis.get("readability_score", "—"))

        for s in suggestions:
            severity = s.get("severity", "medium")
            color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
            st.markdown(f"{color} **{s.get('type', 'finding')}** ({severity}) — {s.get('suggestion', '')}")
            if s.get("reason"):
                st.caption(s["reason"])

        if analysis.get("security_issues"):
            st.markdown("**Security issues:**")
            for issue in analysis["security_issues"]:
                st.markdown(f"- ⚠️ {issue}")

        if analysis.get("optimized_query"):
            st.markdown("**Optimized query:**")
            st.code(analysis["optimized_query"], language="sql")

        with st.expander("Full structured analysis"):
            st.json(analysis)

    with st.expander("Raw job document"):
        st.json(job)
