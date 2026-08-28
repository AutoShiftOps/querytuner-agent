"""
Issue #63: cross-references a parsed EXPLAIN plan (Issue #61/#62) against
the index_recommender suggestions already computed for the same query —
the part of this chain that actually makes QueryInput.jsx's UI promise
true. #61/#62 alone just produce another unused `facts` blob; nothing
before this read `facts.plan` for anything.

Two directions, per the design doc:
  - Confirmation: a suggestion whose table shows a full scan (Seq Scan /
    MySQL access_type ALL) in the real plan is upgraded to
    evidence_level="schema-verified" — the literal word the UI already
    promises.
  - Contradiction: if the plan shows an index already being used on the
    EXACT column a heuristic flagged as unindexed, the suggestion is
    likely wrong (stale heuristic, or schema drift) — flagged via
    plan_contradicts rather than silently left as-is or wrongly upgraded.
    Scoped to single-column suggestion types only (see
    _CONTRADICTION_ELIGIBLE_TYPES) — a composite suggestion isn't
    necessarily wrong just because one of its columns already has its own
    index; that's a separate judgment call this v1 doesn't attempt.

The design doc calls out table/column name matching as "the likely
fragile point" — see _matching_nodes' docstring for exactly how aliases
are resolved, and test_plan_crossref.py for the cases this was actually
verified against.

Gap-followup (docs/querytuner-explain-parser-gap-followup.md): v1 only
ever touched the `index_review_*` family from index_recommender.py.
Three more of #63's own six-heuristic acceptance list are wired in here
now — full_scan_risk, order_by_no_limit, function_in_where (all from
sql_analyzer.py, not index_recommender.py, so none of them carry a
"columns" list to resolve via _matching_nodes; each is confirmed against
the whole plan instead, per-type, below). filesort_detected/
temp_table_detected remain out of scope for this module — per the
follow-up doc, those are new MySQL-only detections with no pre-existing
heuristic to upgrade, so they're generated directly from the plan in
sql_analyzer.py rather than cross-referenced here.
"""

from __future__ import annotations

import re
from typing import Any

# _resolve_real_table is underscore-prefixed (index_recommender.py-internal
# by convention) but reused here rather than duplicated — same alias ->
# real-table-name resolution _detect_composite_opportunity already uses,
# and getting that logic out of sync between the two call sites would be
# a worse outcome than the cross-module reach.
from analysis_engine.index_recommender import _resolve_real_table
from analysis_engine.plan_parsers.models import PlanNode

_INDEX_REVIEW_PREFIX = "index_review_"

_CONTRADICTION_ELIGIBLE_TYPES = {
    "index_review_join_key",
    "index_review_where_filter",
    "index_review_order_by_index",
    "index_review_group_by_index",
    "index_review_partial_index_candidate",
}

# Gap-followup: the three non-index_review_* heuristics #63's issue also
# lists, none of which carry a "columns" list (they're sql_analyzer.py's
# own regex-based heuristics, not index_recommender.py's column-level
# ones) — so each gets confirmed against the whole plan rather than a
# specific aliased table.
_FULL_SCAN_RISK_TYPE = "full_scan_risk"
_ORDER_BY_NO_LIMIT_TYPE = "order_by_no_limit"
_FUNCTION_IN_WHERE_TYPE = "function_in_where"

# Same threshold _findings_from_nodes' own "sort_high_cost" Finding already
# uses for a Sort node (both Postgres parsers) — reusing it here keeps
# order_by_no_limit's "elevated cost" confirmation consistent with the
# signal that's already surfaced elsewhere, instead of inventing a second,
# arbitrarily different cutoff for the same underlying idea.
_SORT_ELEVATED_ROWS = 1000

# Mirrors sql_analyzer.py's own _fn_pattern function list (the one
# function_in_where's *creation* uses) — reused here for its
# *confirmation*, so a plan proving one of these functions is actually
# in a node's filter/condition text counts as evidence for the exact same
# pattern the heuristic already looks for, not a second definition of
# "function call" that could drift out of sync.
_FUNCTION_CALL_RE = re.compile(
    r"\b(lower|upper|trim|ltrim|rtrim|substr|substring|"
    r"cast|convert|"
    r"date|year|month|day|datepart|datename|extract|"
    r"to_date|to_char|to_number|"
    r"isnull|ifnull|nvl|coalesce|nullif|"
    r"abs|round|floor|ceil|ceiling|length|len|"
    r"md5|sha|sha1|sha2)\s*\(",
    re.IGNORECASE,
)


def cross_reference_plan(
    suggestions: list[dict[str, Any]],
    plan_nodes: list[PlanNode],
    schema: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Mutates and returns `suggestions` in place. Touches entries whose type
    starts with "index_review_" (the ones index_recommender.py produces
    about specific columns) plus the three whole-plan heuristic types
    listed above (full_scan_risk, order_by_no_limit, function_in_where);
    every other heuristic type (SELECT *, cartesian join, ...) has nothing
    an EXPLAIN plan could confirm or contradict and is left untouched.
    """
    if not plan_nodes:
        return suggestions
    schema = schema or {}

    for suggestion in suggestions:
        s_type = suggestion.get("type", "")

        if s_type == _FULL_SCAN_RISK_TYPE:
            # Issue's own validation trigger: "Seq Scan node or type=ALL
            # present" — no column/alias to resolve, so any full scan
            # anywhere in the plan confirms it.
            if any(node.is_full_scan for node in plan_nodes):
                suggestion["plan_verified"] = True
                suggestion["evidence_level"] = "schema-verified"
            continue

        if s_type == _ORDER_BY_NO_LIMIT_TYPE:
            # Issue's own validation trigger: "Sort node with elevated
            # cost" — see _SORT_ELEVATED_ROWS for why "elevated" reuses
            # the existing sort_high_cost Finding's threshold.
            if any(node.node_type == "Sort" and (node.rows or 0) > _SORT_ELEVATED_ROWS for node in plan_nodes):
                suggestion["plan_verified"] = True
                suggestion["evidence_level"] = "schema-verified"
            continue

        if s_type == _FUNCTION_IN_WHERE_TYPE:
            # Issue's own validation trigger: "Filter contains function
            # name" — needs the raw filter/condition TEXT, not just
            # condition_column (which only extracts the column side of a
            # comparison and would never contain a function call).
            if any(node.condition_text and _FUNCTION_CALL_RE.search(node.condition_text) for node in plan_nodes):
                suggestion["plan_verified"] = True
                suggestion["evidence_level"] = "schema-verified"
            continue

        if not s_type.startswith(_INDEX_REVIEW_PREFIX):
            continue

        confirmed = False
        contradicted = False

        for qualified_col in suggestion.get("columns") or []:
            alias, sep, col = qualified_col.partition(".")
            if not sep:
                # No alias in the original column reference (e.g. a bare
                # `WHERE status = 'pending'` with no table prefix at all)
                # — _matching_nodes' single-relation fallback handles this.
                alias, col = None, alias

            for node in _matching_nodes(alias, schema, plan_nodes):
                if node.is_full_scan:
                    confirmed = True
                elif (
                    s_type in _CONTRADICTION_ELIGIBLE_TYPES
                    and node.is_index_access
                    and node.condition_column
                    and node.condition_column == col
                ):
                    contradicted = True

        if confirmed:
            suggestion["plan_verified"] = True
            suggestion["evidence_level"] = "schema-verified"
        if contradicted:
            suggestion["plan_contradicts"] = True

    return suggestions


def _matching_nodes(
    alias: str | None,
    schema: dict[str, dict[str, str]],
    plan_nodes: list[PlanNode],
) -> list[PlanNode]:
    """
    Resolves a suggestion's table alias to the plan node(s) that actually
    touch that table. Three ways a match can happen, tried in order:
      1. The plan node's own alias matches exactly (the common, reliable
         case — the EXPLAIN plan is generated from the same query the
         suggestion's alias came from, so an aliased query's plan carries
         the identical alias).
      2. No alias was used in the query at all, so the suggestion's
         "alias" is actually the bare table name — matches the plan
         node's relation name directly.
      3. (Only if schema_info was also pasted) the alias resolves to a
         real table name via schema DDL, matched against the plan node's
         relation name — covers a query that used an alias the EXPLAIN
         plan doesn't happen to echo (rare, but schema-verified table
         resolution is already how index_recommender.py itself handles
         this ambiguity elsewhere).

    If none of those match anything (including when the suggestion's
    column had no alias/table info at all), and the plan only touches ONE
    distinct relation, that relation is assumed to be the one in question
    — the common case for a simple single-table query, which is exactly
    what QueryInput.jsx's own EXPLAIN placeholder example shows
    (`Seq Scan on orders ...`, no alias). A multi-table plan with no
    direct match returns no matches rather than guessing.
    """
    real_table = _resolve_real_table(alias, schema) if alias and schema else None
    matches = [
        node
        for node in plan_nodes
        if (alias and node.alias == alias)
        or (alias and node.relation == alias)
        or (real_table and node.relation == real_table)
    ]
    if matches:
        return matches

    relations = {node.relation for node in plan_nodes if node.relation}
    if len(relations) == 1:
        (only_relation,) = relations
        return [node for node in plan_nodes if node.relation == only_relation]

    return []
