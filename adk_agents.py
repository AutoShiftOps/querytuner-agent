"""
Google Agent Development Kit (google-adk) integration — the actual
"Agent Builder/ADK" route the brief calls out, not just a raw
google-generativeai call. This module owns the two real AI calls in the
whole project (Gemma triage, Gemini explain); gemini_agent.py's pipeline
orchestration calls into it exactly where it previously called a
hand-rolled google-generativeai wrapper, with the same call contract
(prompt in, parsed JSON or a dry-run fallback out) — so this file is a
drop-in upgrade of the model-calling layer, not a rewrite of the
pipeline shape plan -> triage -> analyze -> explain.

Why LlmAgent + Runner rather than a plain client call: this is ADK's
actual entry point for "an agent that talks to Gemini/Gemma," and using
it (session-scoped runs via InMemoryRunner, structured
GenerateContentConfig, ADK's own event stream) is what makes the "Google
AI stack only, via Agent Builder/ADK" claim literally true rather than
just directionally true. A fresh InMemoryRunner + InMemorySessionService
per call is deliberate, not an oversight — each pipeline step is a
single, independent, stateless request (no multi-turn conversation
between triage and explain), so there's no reason to keep a session
alive across them; ADK's session/runner machinery still does real work
per call (assembling the request, dispatching through its own
llm_flows, streaming events back), it's just not asked to remember
anything between calls.

Verified in this environment (no live Gemini/Vertex credentials
available here — see gemini_agent.py's own DRY_RUN path for the fully
offline-testable alternative):
  - LlmAgent + InMemoryRunner + InMemorySessionService construct and
    wire together correctly against the installed google-adk==2.8.0.
  - runner.run_async() correctly drives the full ADK flow up to the
    literal outbound HTTP call — confirmed by two different, expected
    failure points: "No API key was provided" with no key set at all,
    and an actual outbound HTTP attempt (blocked only by this sandbox's
    own network egress policy, not a code defect) once a key was set.
  - Everything up to and including request construction is real,
    demonstrated ADK usage — the one thing that could not be verified
    from this sandbox is a live model response, which needs a real
    GEMINI_API_KEY (or Vertex AI credentials) and a network path Google
    actually allows. Smoke-test this against your own key before
    recording the demo video.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

logger = logging.getLogger(__name__)

_APP_NAME = "queryagent"


def _build_agent(name: str, model: str, instruction: str) -> LlmAgent:
    return LlmAgent(
        name=name,
        model=model,
        instruction=instruction,
        generate_content_config=types.GenerateContentConfig(response_mime_type="application/json"),
    )


async def run_agent_json(agent: LlmAgent, prompt: str, *, dry_run_result: Any) -> Any:
    """
    Runs a single-turn ADK agent invocation and returns the parsed JSON
    response — a fresh InMemoryRunner/session per call (see module
    docstring for why that's deliberate, not sloppy). Any failure
    anywhere in the chain (missing credentials, network error, malformed
    JSON in the response) logs a warning and returns `dry_run_result`
    rather than raising — same fallback contract gemini_agent.py's
    pipeline already expects from its model-calling layer, so a real
    credentials/network problem degrades the same way DRY_RUN mode does,
    instead of taking the whole pipeline step down.
    """
    runner = InMemoryRunner(agent=agent, app_name=_APP_NAME)
    user_id = f"pipeline-{uuid.uuid4().hex[:8]}"

    try:
        session = await runner.session_service.create_session(app_name=_APP_NAME, user_id=user_id)
        message = types.Content(role="user", parts=[types.Part(text=prompt)])

        final_text: str | None = None
        async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=message):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(p.text or "" for p in event.content.parts)

        if not final_text:
            logger.warning("ADK agent %s returned no final response text", agent.name)
            return dry_run_result

        return json.loads(final_text)

    except Exception as e:
        logger.warning("ADK agent %s call failed, falling back to dry-run-style output: %s", agent.name, e)
        return dry_run_result


def build_triage_agent(model: str) -> LlmAgent:
    return _build_agent(
        name="gemma_triage_agent",
        model=model,
        instruction=(
            "You are the triage step of a SQL analysis agent — a fast, lightweight "
            "model. For each query you're given, judge whether it looks likely to "
            "have a real performance issue worth a full explanation (priority: "
            "'high'), a minor one ('medium'), or looks fine ('low'). Be quick and "
            "approximate — a slower, more careful analysis pass runs after you, "
            "independent of your judgment. Always respond with JSON only, no "
            'markdown fences, shaped exactly as: {"triage": [{"index": <int>, '
            '"priority": "high"|"medium"|"low", "reason": "<short phrase>"}]}'
        ),
    )


def build_explain_agent(model: str) -> LlmAgent:
    return _build_agent(
        name="gemini_explain_agent",
        model=model,
        instruction=(
            "You are the final step of a SQL analysis agent — write a concise, "
            "consultant-grade executive summary of the findings you're given, for "
            "an engineer who will act on them. You'll also be given a triage "
            "priority per query — weight your attention accordingly: the most "
            "explanation on 'high' priority findings, a sentence or two on "
            "'medium', one line or a skip on 'low'. Be specific and reference real "
            "table/column names and DDL where present in the findings — no generic "
            "advice. Always respond with JSON only, no markdown fences, shaped "
            'exactly as: {"summary": "<markdown-formatted executive summary>", '
            '"top_actions": ["<action 1>", "<action 2>", ...]}'
        ),
    )


def build_plan_agent(model: str) -> LlmAgent:
    return _build_agent(
        name="gemini_plan_agent",
        model=model,
        instruction=(
            "You are the planning step of a SQL analysis agent. Given a short "
            "description of the job about to run, respond with a single sentence "
            "describing what it's about to do, for a progress indicator a user is "
            "watching. Always respond with JSON only, no markdown fences, shaped "
            'exactly as: {"rationale": "<one sentence>"}'
        ),
    )
