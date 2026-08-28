"""
Issue #62: parses a pasted MySQL EXPLAIN plan — JSON (EXPLAIN FORMAT=JSON)
and, per the gap-followup doc, plain tabular EXPLAIN SELECT ... output too
(both the classic pipe-table shape and the `EXPLAIN ... \\G` vertical
shape a mysql-cli user might paste instead).

Not a copy-paste of the Postgres parser (postgres.py) — MySQL's JSON shape
is a distinct format per the UI's own placeholder example
(QueryInput.jsx's explainPlaceholders.mysql: `{"query_block": {"table":
{"table_name": "orders", "access_type": "ALL", ...}}}`), keyed by
`query_block`/`table`/`nested_loop` rather than Postgres's `Node
Type`/`Plan`/`Plans`.

Walks the JSON tree generically (recurse into every dict/list, collect
every "table" object found) rather than modeling MySQL's full JSON EXPLAIN
grammar exactly (nested_loop arrays, materialized subqueries, hash join
representation in MySQL 8+, ...) — robust to the different nesting shapes
a real query can produce without needing to enumerate all of them, which
is more scope than #62 needs (see design doc: "at minimum parses
request.explain_plan the same way the Postgres path does").

Gap-followup (docs/querytuner-explain-parser-gap-followup.md) closed here:
  1. Plain tabular EXPLAIN SELECT ... output — see _parse_tabular_nodes.
     QueryInput.jsx's MySQL placeholder still only shows JSON; tabular
     support is backend-only for now (a user who pastes it despite the
     placeholder's hint still gets it parsed) rather than also changing
     the placeholder copy, which is a product/UI call outside this scope.
  2. `key=NULL` flagged explicitly, distinct from `type=ALL` — see
     "no_index_used" in _findings_from_nodes.
  3. `Using filesort` / `Using temporary` (MySQL's `Extra` field, both
     tabular and JSON) — see _extra_findings/_json_has_flag. These map
     to new `filesort_detected`/`temp_table_detected` Finding types that
     didn't exist anywhere in the codebase before this; they're
     plan-only signals (no static heuristic can guess them from SQL text
     alone), so unlike this module's other findings they get promoted
     straight into optimization_suggestions by sql_analyzer.py rather
     than needing plan_crossref.py to "confirm" a pre-existing guess.
"""

from __future__ import annotations

import json
import re

from analysis_engine.schemas import Finding, PlanArtifact
from analysis_engine.tabular_parse import parse_pipe_table_rows, parse_vertical_rows

from .models import ParsedPlan, PlanNode

# Reused from postgres.py rather than re-implemented — same "identifier
# immediately followed by a comparison operator" rule, so the two parsers'
# condition_column extraction can't silently drift apart.
from .postgres import _extract_condition_column

# ── JSON tree parsing (EXPLAIN FORMAT=JSON, pasted) ──────────────────────


def _walk_mysql_json(node, nodes: list[PlanNode]) -> None:
    if isinstance(node, dict):
        table = node.get("table")
        if isinstance(table, dict):
            table_name = table.get("table_name")
            access_type = table.get("access_type") or ""
            rows = table.get("rows_examined_per_scan")
            if rows is None:
                rows = table.get("rows_produced_per_join")
            key = table.get("key")  # the index actually used, if any — None for a full scan

            cost_info = table.get("cost_info") or {}
            cost = None
            raw_cost = cost_info.get("read_cost") or cost_info.get("prefix_cost") or cost_info.get("query_cost")
            if raw_cost is not None:
                try:
                    cost = float(raw_cost)
                except (TypeError, ValueError):
                    cost = None

            # Gap-followup: MySQL's JSON EXPLAIN exposes the filter MySQL
            # actually applied via "attached_condition" — the JSON-side
            # equivalent of Postgres's "Filter". Not populated pre-followup
            # (v1 didn't extract per-table condition text for MySQL at
            # all); needed for function_in_where cross-referencing.
            condition_text = table.get("attached_condition")

            nodes.append(
                PlanNode(
                    node_type=access_type or "unknown",
                    relation=table_name,
                    rows=int(rows) if rows is not None else None,
                    cost=cost,
                    index_name=key,
                    condition_text=condition_text,
                    condition_column=_extract_condition_column(condition_text),
                    # Issue #62's explicit mapping: access_type == "ALL" is
                    # MySQL's equivalent of Postgres's Seq Scan. Any other
                    # access_type ("ref", "eq_ref", "range", "index",
                    # "const", ...) means the planner is using an index to
                    # at least some degree.
                    is_full_scan=(access_type == "ALL"),
                    is_index_access=bool(access_type) and access_type != "ALL",
                )
            )
        for value in node.values():
            _walk_mysql_json(value, nodes)
    elif isinstance(node, list):
        for item in node:
            _walk_mysql_json(item, nodes)


def _json_has_flag(node, flag_key: str) -> bool:
    """Gap-followup: recursively checks whether `flag_key` (e.g.
    "using_filesort") is true anywhere in the JSON tree. MySQL 8's JSON
    EXPLAIN nests these under "ordering_operation"/"grouping_operation"/
    "duplicates_removal" objects that aren't "table" objects themselves
    and can appear at any depth — a targeted lookup would need to model
    all three shapes; a tree-wide scan doesn't need to."""
    if isinstance(node, dict):
        if node.get(flag_key) is True:
            return True
        return any(_json_has_flag(v, flag_key) for v in node.values())
    if isinstance(node, list):
        return any(_json_has_flag(item, flag_key) for item in node)
    return False


# ── Plain tabular parsing (EXPLAIN SELECT ..., pasted) ───────────────────
#
# Traditional mysql-cli pipe-table output, e.g.:
#   +----+-------------+--------+------+---------------+------+---------+------+-------+-------------+
#   | id | select_type | table  | type | possible_keys | key  | key_len | ref  | rows  | Extra       |
#   +----+-------------+--------+------+---------------+------+---------+------+-------+-------------+
#   |  1 | SIMPLE      | orders | ALL  | NULL          | NULL | NULL    | NULL | 10000 | Using where |
#   +----+-------------+--------+------+---------------+------+---------+------+-------+-------------+
# `partitions` and `filtered` columns (shown by EXPLAIN ... FORMAT=TRADITIONAL
# with EXTENDED/newer defaults) are optional — matched generically by column
# name, not position, so their presence/absence doesn't matter. Both this
# shape and `EXPLAIN ... \G` vertical output are now parsed by
# app.tools.tabular_parse (imported above) — not MySQL-specific, shared
# with batch_parsers.py (Phase 5 #115/#120's pasted DB-export formats).


def _row_int(row: dict[str, str], key: str) -> int | None:
    val = row.get(key)
    if val is None or val.upper() == "NULL":
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _row_to_node(row: dict[str, str]) -> PlanNode | None:
    table_name = row.get("table")
    if not table_name or table_name.upper() == "NULL":
        return None
    # Case preserved as-authored (matches real mysql-cli output and the
    # JSON parser's own access_type handling below) rather than forced to
    # a single case — only the ALL comparison is case-normalized.
    access_type = row.get("type") or ""
    key = row.get("key")
    if key is not None and key.upper() == "NULL":
        key = None
    return PlanNode(
        node_type=access_type or "unknown",
        relation=table_name,
        rows=_row_int(row, "rows"),
        index_name=key,
        # Deliberately NOT populated from the "Extra" column: unlike
        # Postgres's Filter/Index Cond, MySQL's Extra is a fixed set of
        # planner annotations ("Using where", "Using filesort", ...), not
        # the actual condition expression — stuffing it into
        # condition_text would pair it with a condition_column that was
        # never actually extracted from it. Extra's own signals (filesort/
        # temp table) are handled separately — see _tabular_extra_texts.
        condition_text=None,
        is_full_scan=(access_type.upper() == "ALL"),
        is_index_access=bool(access_type) and access_type.upper() != "ALL",
    )


def _tabular_rows(raw: str) -> list[dict[str, str]]:
    rows = parse_pipe_table_rows(raw)
    if not rows:
        rows = parse_vertical_rows(raw)
    return rows


def _parse_tabular_nodes(raw: str) -> list[PlanNode]:
    rows = _tabular_rows(raw)
    return [n for n in (_row_to_node(r) for r in rows) if n is not None]


def _tabular_extra_texts(raw: str) -> list[str]:
    return [row.get("extra", "") for row in _tabular_rows(raw)]


# ── Shared finding generation ────────────────────────────────────────────


def _findings_from_nodes(nodes: list[PlanNode]) -> list[Finding]:
    findings: list[Finding] = []
    for node in nodes:
        if node.is_full_scan and (node.rows or 0) > 1000:
            findings.append(
                Finding(
                    type="full_table_scan",
                    severity="high",
                    title=f"Full table scan on '{node.relation or 'unknown'}'",
                    evidence=f"access_type=ALL, estimated {node.rows} rows",
                    recommendation="Consider adding an index on the filter column(s)",
                )
            )
        elif node.is_index_access:
            findings.append(
                Finding(
                    type="index_scan_confirmed",
                    severity="low",
                    title=f"Index usage confirmed on '{node.relation or 'unknown'}' (access_type={node.node_type})",
                    evidence=(f"Using key `{node.index_name}`" if node.index_name else f"access_type={node.node_type}"),
                    recommendation="No action needed — the planner is already using an index here.",
                )
            )

        # Gap-followup: key=NULL flagged explicitly as its own signal, per
        # the issue text calling it out as "distinct from type=ALL" — a
        # deliberate second Finding even on a row that already produced
        # full_table_scan above (type=ALL rows are usually key=NULL too),
        # since the issue wants key=NULL itself surfaced, not just folded
        # silently into the type=ALL case. Also covers the type!=ALL, no
        # usable key case (e.g. type=index scanning an index without
        # actually filtering by it) that type=ALL alone would miss
        # entirely. Skipped only when access_type is itself unknown
        # (tabular parsing failed to find a `type` column at all) to
        # avoid a false positive on incomplete data.
        if node.index_name is None and node.node_type != "unknown":
            findings.append(
                Finding(
                    type="no_index_used",
                    severity="medium",
                    title=f"No index used on '{node.relation or 'unknown'}' (key=NULL)",
                    evidence=f"access_type={node.node_type}, key=NULL",
                    recommendation="Review possible_keys and query filters — no index was selected for this access",
                )
            )
    return findings


def _extra_findings(raw_extra_texts: list[str]) -> list[Finding]:
    """Gap-followup: 'Using filesort'/'Using temporary' live in MySQL's
    Extra field (tabular) or the using_filesort/using_temporary_table JSON
    keys — surfaced here as two new, query-wide Finding types (not
    per-table, since filesort/temp-table application happens once for the
    query's sort/group operation, not per scanned table)."""
    findings: list[Finding] = []
    combined = " ".join(raw_extra_texts)
    if re.search(r"using filesort", combined, re.IGNORECASE):
        findings.append(
            Finding(
                type="filesort_detected",
                severity="medium",
                title="Filesort detected",
                evidence="EXPLAIN output shows 'Using filesort'",
                recommendation="Add an index matching the ORDER BY column(s) to avoid the sort",
            )
        )
    if re.search(r"using temporary", combined, re.IGNORECASE):
        findings.append(
            Finding(
                type="temp_table_detected",
                severity="medium",
                title="Temporary table detected",
                evidence="EXPLAIN output shows 'Using temporary'",
                recommendation="Review GROUP BY/DISTINCT/ORDER BY combinations — MySQL had to materialize a temp table",
            )
        )
    return findings


# ── Public entry points ──────────────────────────────────────────────────


def parse_mysql_explain(raw: str) -> ParsedPlan | None:
    raw = (raw or "").strip()
    if not raw:
        return None

    try:
        plan_json = json.loads(raw)
    except (ValueError, TypeError):
        plan_json = None

    if plan_json is not None:
        nodes: list[PlanNode] = []
        _walk_mysql_json(plan_json, nodes)
        if not nodes:
            return None
        findings = _findings_from_nodes(nodes)
        if _json_has_flag(plan_json, "using_filesort"):
            findings.extend(_extra_findings(["Using filesort"]))
        if _json_has_flag(plan_json, "using_temporary_table"):
            findings.extend(_extra_findings(["Using temporary"]))
        return ParsedPlan(
            artifact=PlanArtifact(format="json", raw=plan_json),
            nodes=nodes,
            findings=findings,
        )

    # Gap-followup: plain tabular EXPLAIN SELECT ... output — the format
    # the issue lists that JSON-only v1 didn't attempt.
    nodes = _parse_tabular_nodes(raw)
    if not nodes:
        return None
    findings = _findings_from_nodes(nodes)
    findings.extend(_extra_findings(_tabular_extra_texts(raw)))
    return ParsedPlan(
        artifact=PlanArtifact(format="text", raw=raw),
        nodes=nodes,
        findings=findings,
    )


def nodes_from_artifact(artifact: PlanArtifact) -> list[PlanNode]:
    """Mirrors postgres.py's nodes_from_artifact — re-derives structured
    nodes from an already-built PlanArtifact for execution_planner.collect_facts()."""
    if artifact.format == "json":
        nodes: list[PlanNode] = []
        _walk_mysql_json(artifact.raw, nodes)
        return nodes
    if artifact.format == "text":
        return _parse_tabular_nodes(str(artifact.raw))
    return []
