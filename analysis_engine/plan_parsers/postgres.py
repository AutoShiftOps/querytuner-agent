"""
Issue #61: parses a PASTED Postgres EXPLAIN plan — either JSON
(EXPLAIN (FORMAT JSON) ...) or plain-text tabular output
(EXPLAIN (ANALYZE, BUFFERS) ..., the format QueryInput.jsx's own
placeholder actually shows: "Seq Scan on orders  (cost=0.00..431.00
rows=10000 width=244)"). Before this, the only Postgres EXPLAIN parsing
in this codebase (_extract_findings/_walk_node, now folded into
_findings_from_nodes/_walk_json_node below) only ever ran against a
*live* asyncpg connection's already-parsed JSON — it had never been
pointed at request.explain_plan (the raw string a user pastes), so
pasting a real plan did nothing.

Both the pasted-JSON path here and the live-DSN path in
collectors/postgres.py share the same node-walking and finding-generation
logic (json_nodes_for_live_plan) — one implementation, not a duplicate
kept in sync by hand.
"""

from __future__ import annotations

import json
import re

from analysis_engine.schemas import Finding, PlanArtifact

from .models import ParsedPlan, PlanNode

# Node types that represent the database actually using an index to
# narrow rows, as opposed to reading the whole table.
_INDEX_ACCESS_NODE_TYPES = {"Index Scan", "Index Only Scan", "Bitmap Heap Scan", "Bitmap Index Scan"}

# Gap-followup (#61): join/aggregate strategy node types the issue lists
# alongside Hash Join/Nested Loop but that shipped without any
# classification or Finding of their own — grep confirmed none of these
# three literal strings appeared anywhere in this module before this.
# Neither full-scan nor index-access (a Merge Join can run over an index
# OR a sorted seq scan; an aggregate node says nothing about how its
# child scanned), so they get their own bucket rather than being folded
# into either existing set — kept separate in case #63's cross-referencing
# is ever extended to reason about join/aggregate strategy specifically.
_JOIN_OR_AGGREGATE_NODE_TYPES = {"Merge Join", "Hash Aggregate", "Group Aggregate"}

# Matches one node line of Postgres's plain-text EXPLAIN output, e.g.:
#   Seq Scan on orders  (cost=0.00..431.00 rows=10000 width=244)
#   ->  Index Scan using idx_orders_status on orders o  (cost=0.42..8.44 rows=1 width=244)
#   Hash Join  (cost=1.05..431.00 rows=10000 width=244)
# The "using INDEX" and "on RELATION [ALIAS]" groups are both optional —
# join/sort/aggregate node types (Hash Join, Sort, ...) have neither.
# Gap-followup: added the trailing, still-optional "(actual time=X..Y
# rows=N loops=N)" group EXPLAIN ANALYZE appends after the cost group —
# previously unmatched (no $ anchor, so it was just silently ignored
# rather than failing to match), meaning ANALYZE's actual-row/timing data
# was thrown away even when present in the pasted plan.
_NODE_LINE_RE = re.compile(
    r"""^\s*(?:->\s*)?
    (?P<node_type>[A-Za-z][A-Za-z ]*?)
    (?:\s+using\s+(?P<index_name>[\w.]+))?
    (?:\s+on\s+(?P<relation>[\w.]+)(?:\s+(?P<alias>[a-zA-Z_]\w*))?)?
    \s+\(cost=(?P<cost_lo>[\d.]+)\.\.(?P<cost_hi>[\d.]+)\s+rows=(?P<rows>\d+)\s+width=(?P<width>\d+)\)
    (?:\s+\(actual\s+time=(?P<actual_time_lo>[\d.]+)\.\.(?P<actual_time_hi>[\d.]+)
      \s+rows=(?P<actual_rows>\d+)\s+loops=(?P<loops>\d+)\))?""",
    re.VERBOSE,
)

# Matches the column referenced by a condition line ("Index Cond:",
# "Recheck Cond:", "Filter:" in text output; the same three keys in JSON
# output, just without the leading label). Requires the identifier be
# immediately followed by a comparison operator so this doesn't grab an
# unrelated word — e.g. "Filter: (status = 'pending'::text)" -> "status",
# not "pending".
_CONDITION_COLUMN_RE = re.compile(
    r"\(*(?:[A-Za-z_]\w*\.)?([A-Za-z_]\w*)\s*(?:=|<=|>=|<>|!=|<|>|\bIS\b)",
    re.IGNORECASE,
)
_CONDITION_LABEL_RE = re.compile(r"^\s*(?:Index Cond|Recheck Cond|Filter):\s*(.*)$")


def _extract_condition_column(text: str | None) -> str | None:
    if not text:
        return None
    m = _CONDITION_COLUMN_RE.search(text)
    return m.group(1) if m else None


# ── Plain-text tabular parsing ───────────────────────────────────────────


def _parse_text_nodes(raw: str) -> list[PlanNode]:
    nodes: list[PlanNode] = []
    current: PlanNode | None = None

    for line in raw.splitlines():
        if not line.strip():
            continue

        m = _NODE_LINE_RE.match(line)
        if m:
            node_type = m.group("node_type").strip()
            relation = m.group("relation")
            index_name = m.group("index_name")
            # Postgres-specific quirk: "Bitmap Index Scan on X" — X is the
            # INDEX name, not a table, unlike every other "on X" node type.
            if node_type == "Bitmap Index Scan":
                index_name = index_name or relation
                relation = None
            actual_rows = m.group("actual_rows")
            actual_time_hi = m.group("actual_time_hi")
            current = PlanNode(
                node_type=node_type,
                relation=relation,
                alias=m.group("alias"),
                rows=int(m.group("rows")),
                cost=float(m.group("cost_hi")),
                index_name=index_name,
                is_full_scan=(node_type == "Seq Scan"),
                is_index_access=node_type in _INDEX_ACCESS_NODE_TYPES,
                actual_rows=int(actual_rows) if actual_rows is not None else None,
                actual_time_ms=float(actual_time_hi) if actual_time_hi is not None else None,
            )
            nodes.append(current)
            continue

        # Not a new node line — check whether it's a condition detail line
        # (more indented, attached to the most recently seen node).
        if current is not None and current.condition_column is None:
            label_m = _CONDITION_LABEL_RE.match(line)
            if label_m:
                current.condition_text = label_m.group(1)
                current.condition_column = _extract_condition_column(label_m.group(1))

    return nodes


# ── JSON tree parsing (EXPLAIN (FORMAT JSON), pasted or from a live conn) ──


def _walk_json_node(node: dict, nodes: list[PlanNode]) -> None:
    if not isinstance(node, dict):
        return

    node_type = node.get("Node Type", "")
    relation = node.get("Relation Name")
    index_name = node.get("Index Name")
    if node_type == "Bitmap Index Scan":
        index_name = index_name or relation
        relation = None

    condition_column = None
    condition_text = None
    for key in ("Index Cond", "Recheck Cond", "Filter"):
        condition_text = node.get(key)
        if condition_text:
            condition_column = _extract_condition_column(condition_text)
            break

    # Gap-followup: EXPLAIN (FORMAT JSON, ANALYZE) exposes these directly —
    # no regex needed the way the text format requires.
    actual_rows = node.get("Actual Rows")
    actual_time_ms = node.get("Actual Total Time")

    nodes.append(
        PlanNode(
            node_type=node_type,
            relation=relation,
            alias=node.get("Alias"),
            rows=node.get("Plan Rows"),
            cost=node.get("Total Cost"),
            index_name=index_name,
            condition_column=condition_column,
            condition_text=condition_text,
            is_full_scan=(node_type == "Seq Scan"),
            is_index_access=node_type in _INDEX_ACCESS_NODE_TYPES,
            actual_rows=actual_rows,
            actual_time_ms=actual_time_ms,
        )
    )
    for child in node.get("Plans", []) or []:
        _walk_json_node(child, nodes)


def _json_nodes(plan) -> list[PlanNode]:
    if isinstance(plan, list) and plan:
        root = plan[0].get("Plan", {}) if isinstance(plan[0], dict) else {}
    elif isinstance(plan, dict):
        root = plan.get("Plan", plan)
    else:
        root = {}
    nodes: list[PlanNode] = []
    _walk_json_node(root, nodes)
    return nodes


# ── Shared finding generation ────────────────────────────────────────────


def _findings_from_nodes(nodes: list[PlanNode]) -> list[Finding]:
    findings: list[Finding] = []
    for node in nodes:
        if node.node_type == "Seq Scan" and (node.rows or 0) > 1000:
            findings.append(
                Finding(
                    type="seq_scan",
                    severity="high",
                    title=f"Sequential scan on '{node.relation or 'unknown'}'",
                    evidence=f"Estimated {node.rows} rows, cost {node.cost}",
                    recommendation="Consider adding an index on the filter column(s)",
                )
            )
        elif node.node_type in _INDEX_ACCESS_NODE_TYPES:
            # Issue #61: positive signal, new in this pass — confirms an
            # index IS being used, important for #63's cross-referencing
            # (a heuristic suggestion for this table might be contradicted
            # by this, not just left unconfirmed).
            findings.append(
                Finding(
                    type="index_scan_confirmed",
                    severity="low",
                    title=f"{node.node_type} confirms index usage on '{node.relation or node.index_name or 'unknown'}'",
                    evidence=(
                        f"Using `{node.index_name}`"
                        + (f", estimated {node.rows} rows" if node.rows is not None else "")
                        if node.index_name
                        else f"Estimated {node.rows} rows"
                    ),
                    recommendation="No action needed — the planner is already using an index here.",
                )
            )

        if node.node_type == "Nested Loop" and (node.cost or 0) > 5000:
            findings.append(
                Finding(
                    type="nested_loop",
                    severity="medium",
                    title="Expensive Nested Loop join detected",
                    evidence=f"Total cost: {node.cost}",
                    recommendation="Check join conditions and ensure join columns are indexed",
                )
            )

        if node.node_type == "Hash Join":
            findings.append(
                Finding(
                    type="hash_join",
                    severity="low",
                    title="Hash Join detected",
                    evidence=f"Rows: {node.rows}, Cost: {node.cost}",
                    recommendation="Hash joins are generally efficient; ensure adequate work_mem",
                )
            )

        # Gap-followup: the three node types #61's issue lists that shipped
        # with no classification or Finding at all — mirrors Hash Join's
        # "informational, low severity" treatment above (a Merge/Hash/Group
        # strategy choice isn't inherently a problem the way a Seq Scan or
        # expensive Nested Loop is).
        if node.node_type in _JOIN_OR_AGGREGATE_NODE_TYPES:
            findings.append(
                Finding(
                    type="join_or_aggregate_strategy",
                    severity="low",
                    title=f"{node.node_type} detected",
                    evidence=f"Rows: {node.rows}, Cost: {node.cost}",
                    recommendation=(
                        "No action needed by default; review if this appears alongside other high-cost nodes"
                    ),
                )
            )

        # Issue #61: new node type, relevant to the existing
        # order_by_index heuristic — a Sort over a large estimated row
        # count is exactly the filesort that heuristic warns about.
        if node.node_type == "Sort" and (node.rows or 0) > 1000:
            findings.append(
                Finding(
                    type="sort_high_cost",
                    severity="medium",
                    title="Sort operation on large estimated row count",
                    evidence=f"Estimated {node.rows} rows sorted",
                    recommendation="Consider an index matching the ORDER BY column(s) to avoid this sort (filesort)",
                )
            )
    return findings


# ── Public entry points ──────────────────────────────────────────────────


def parse_postgres_explain(raw: str) -> ParsedPlan | None:
    """
    Parses a pasted Postgres EXPLAIN plan — JSON or plain-text tabular,
    matching both formats QueryInput.jsx's own placeholder/hint invite a
    user to paste. Returns None if raw is empty or neither shape yields
    any recognizable node (callers should treat that as "couldn't parse
    this," not silently succeed with an empty plan).
    """
    raw = (raw or "").strip()
    if not raw:
        return None

    try:
        plan_json = json.loads(raw)
    except (ValueError, TypeError):
        plan_json = None

    if plan_json is not None:
        nodes = _json_nodes(plan_json)
        if not nodes:
            return None
        return ParsedPlan(
            artifact=PlanArtifact(format="json", raw=plan_json),
            nodes=nodes,
            findings=_findings_from_nodes(nodes),
        )

    nodes = _parse_text_nodes(raw)
    if not nodes:
        return None
    return ParsedPlan(
        artifact=PlanArtifact(format="text", raw=raw),
        nodes=nodes,
        findings=_findings_from_nodes(nodes),
    )


def nodes_from_artifact(artifact: PlanArtifact) -> list[PlanNode]:
    """Re-derives structured nodes from an already-built PlanArtifact —
    used by execution_planner.collect_facts() to get #63's cross-referencing
    input without changing BaseCollector's return contract for every
    dialect's collector."""
    if artifact.format == "json":
        return _json_nodes(artifact.raw)
    if artifact.format == "text":
        return _parse_text_nodes(str(artifact.raw))
    return []


def json_nodes_for_live_plan(plan_json: list) -> tuple[list[PlanNode], list[Finding]]:
    """Used by PostgresCollector's live-DSN path — asyncpg already returns
    the plan as parsed JSON, so this shares the exact same walking/finding
    logic the pasted-JSON path uses instead of keeping a second copy."""
    nodes = _json_nodes(plan_json)
    return nodes, _findings_from_nodes(nodes)
