"""
Issue #115 — reconciles `index_review_*` suggestions collected across a
BATCH of queries (Issue #120's input) into one set instead of just
concatenating N independent single-query results. Per the issue's own
framing: "queries individually can produce several nearly identical
indexes that look reasonable alone but conflict at the table level
(over-indexing, write performance degradation)."

Deliberately does NOT touch anything but `index_review_*` types — see
the design doc's explicit non-goal: the other heuristic types
(cartesian_join, function_in_where, ...) are inherently per-query
findings with nothing meaningful to reconcile against each other.

Reuses `IndexRecommender.recommend()` and `_resolve_real_table` as-is
(the latter imported from index_recommender.py, same cross-module reuse
plan_crossref.py already established) — this module's only new logic is
grouping and comparing suggestions that already exist, not re-deriving
them.

## Table identity: the one place this needs schema_info to be trustworthy

Two suggestions can only be safely compared as "about the same table"
when the real table name is actually known:
  - Alias resolves via `_resolve_real_table()` against pasted schema DDL
    -> grouped by that real table name. This is the case cross-query
    reconciliation is actually designed for.
  - Alias present but schema doesn't resolve it -> grouped by the raw
    alias string itself. Only correct if the SAME alias happens to mean
    the same table in every query in the batch — a real assumption, not
    a guarantee, for a batch pulled from independently-written production
    queries. Flagged in BatchReconciliationResult.warnings when this path
    is taken for any group.
  - No alias at all (unaliased single-table query, e.g. bare
    `WHERE status = ...`) -> given its own query-scoped key, never
    merged with anything from another query. Merging these across
    queries with no table signal at all would risk treating two
    genuinely different tables as one, which is a worse outcome than
    under-reconciling — same reasoning `_matching_nodes()` in
    plan_crossref.py already applies ("a multi-table plan with no direct
    match returns no matches rather than guessing").

## Two reconciliation moves, per the design doc

1. Subset/superset collapse: a suggestion's column set that's a strict
   subset of another suggestion's set on the same table is dropped as
   redundant — the superset suggestion is kept and its `satisfies_queries`
   absorbs the dropped suggestion's queries too (a composite index also
   serves the narrower need the subset suggestion was about).
2. Column-order conflict: the same column SET suggested in a different
   ORDER by different queries is flagged, not auto-merged — #117's
   column-ordering logic means "right order for query A" can be "wrong
   order for query B" on the exact same columns, and that tension should
   be visible, not silently resolved one way.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from analysis_engine.index_recommender import _resolve_real_table

_INDEX_REVIEW_PREFIX = "index_review_"


def _split_qualified(qualified_col: str) -> tuple[str | None, str]:
    alias, sep, col = qualified_col.partition(".")
    if sep:
        return alias, col
    return None, alias


def _suggestion_alias_and_columns(suggestion: dict[str, Any]) -> tuple[str | None, tuple[str, ...]]:
    """A suggestion's `columns` list is always qualified the same way for
    every entry (index_recommender.py builds one suggestion per single
    alias) — takes the first alias seen as the suggestion's table alias,
    and the bare (unqualified) column names as its column set."""
    cols = suggestion.get("columns") or []
    alias: str | None = None
    bare: list[str] = []
    for qc in cols:
        a, c = _split_qualified(qc)
        if alias is None and a is not None:
            alias = a
        bare.append(c)
    return alias, tuple(bare)


@dataclass
class _Group:
    """One (table, exact column set + order) cluster — the unit a
    reconciled entry or a dropped entry is ultimately built from."""

    alias: str | None
    columns: tuple[str, ...]  # order as originally suggested
    suggestion: dict[str, Any]  # representative — first one seen
    query_indices: list[int] = field(default_factory=list)


@dataclass
class ReconciledSuggestion:
    table: str | None
    suggestion: dict[str, Any]
    satisfies_queries: list[int]


@dataclass
class DroppedSuggestion:
    table: str | None
    columns: list[str]
    suggestion_text: str
    source_query_indices: list[int]
    reason: str
    superseded_by_columns: list[str]


@dataclass
class ColumnOrderConflict:
    table: str | None
    columns: list[str]  # canonical (sorted) column set
    variants: list[dict[str, Any]]  # [{"order": [...], "queries": [...]}, ...]


@dataclass
class BatchReconciliationResult:
    reconciled_suggestions: list[ReconciledSuggestion] = field(default_factory=list)
    dropped_suggestions: list[DroppedSuggestion] = field(default_factory=list)
    column_order_conflicts: list[ColumnOrderConflict] = field(default_factory=list)
    # Unresolved-alias groups were reconciled anyway (see module docstring)
    # — surfaced so a caller/UI can show a "verify" caveat rather than
    # presenting cross-query merges done without real table confirmation
    # with the same confidence as schema-verified ones.
    warnings: list[str] = field(default_factory=list)


def reconcile_index_suggestions(
    query_suggestions: list[tuple[int, list[dict[str, Any]]]],
    schema: dict[str, dict[str, str]] | None = None,
) -> BatchReconciliationResult:
    """
    `query_suggestions`: [(query_index, suggestions_for_that_query), ...]
    — suggestions_for_that_query is whatever IndexRecommender.recommend()
    returned for that one query. Non-index_review_* entries are ignored
    here, untouched — callers that want those at all should keep the
    original per-query list around separately.
    """
    schema = schema or {}
    warnings: list[str] = []
    unresolved_alias_used = False

    # ── Step 1: bucket every index_review_* suggestion by (table, exact
    # column set + order). ──────────────────────────────────────────────
    groups: dict[tuple[str, tuple[str, ...]], _Group] = {}
    table_columns: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    table_display: dict[str, str | None] = {}

    for query_index, suggestions in query_suggestions:
        for suggestion in suggestions or []:
            if not suggestion.get("type", "").startswith(_INDEX_REVIEW_PREFIX):
                continue
            alias, columns = _suggestion_alias_and_columns(suggestion)
            if not columns:
                continue

            real_table = _resolve_real_table(alias, schema) if alias else None
            if real_table:
                table_key, display = real_table, real_table
            elif alias:
                table_key, display = f"alias:{alias}", alias
                unresolved_alias_used = True
            else:
                # No alias at all — never merged across queries (see
                # module docstring).
                table_key, display = f"query{query_index}:unaliased", None

            table_display[table_key] = display
            table_columns[table_key].add(columns)

            key = (table_key, columns)
            grp = groups.get(key)
            if grp is None:
                grp = _Group(alias=alias, columns=columns, suggestion=suggestion)
                groups[key] = grp
            if query_index not in grp.query_indices:
                grp.query_indices.append(query_index)

    if unresolved_alias_used:
        warnings.append(
            "Some suggestions were grouped by alias rather than a schema-resolved real table "
            "name (no schema_info matched). Cross-query reconciliation for those assumes the "
            "same alias means the same table in every query, which may not hold — paste schema "
            "DDL for reliable cross-query reconciliation."
        )

    column_order_conflicts: list[ColumnOrderConflict] = []
    dropped: list[DroppedSuggestion] = []
    dropped_group_keys: set[tuple[str, tuple[str, ...]]] = set()

    for table_key, col_tuples in table_columns.items():
        display = table_display[table_key]
        by_set: dict[frozenset, list[tuple[str, ...]]] = defaultdict(list)
        for cols in col_tuples:
            by_set[frozenset(cols)].append(cols)

        # ── Step 2: same column SET, different ORDER -> flagged, not
        # merged (kept as separate reconciled entries below). ───────────
        for col_set, orders in by_set.items():
            if len(col_set) >= 2 and len(orders) > 1:
                column_order_conflicts.append(
                    ColumnOrderConflict(
                        table=display,
                        columns=sorted(col_set),
                        variants=[
                            {"order": list(o), "queries": sorted(groups[(table_key, o)].query_indices)}
                            for o in sorted(orders)
                        ],
                    )
                )

        # ── Step 3: subset/superset collapse, largest column-set first so
        # a chain (A subset of B subset of C) collapses onto the largest
        # survivor, not an intermediate one. ─────────────────────────────
        distinct_sets = sorted(by_set.keys(), key=lambda s: (-len(s), sorted(s)))
        kept: list[frozenset] = []
        superseded_by: dict[frozenset, frozenset] = {}
        for col_set in distinct_sets:
            superset = next((k for k in kept if col_set < k), None)
            if superset is not None:
                superseded_by[col_set] = superset
            else:
                kept.append(col_set)

        for col_set, superset in superseded_by.items():
            # Fold every order-variant group of the dropped set's queries
            # into the FIRST order-variant group of the surviving
            # superset — when the superset itself has multiple
            # order-variants (its own column_order_conflict, handled
            # above), this is a deliberate simplification: absorb into
            # one representative rather than picking ambiguously per
            # dropped query.
            primary_superset_group = groups[(table_key, by_set[superset][0])]
            for cols in by_set[col_set]:
                grp = groups[(table_key, cols)]
                dropped_group_keys.add((table_key, cols))
                dropped.append(
                    DroppedSuggestion(
                        table=display,
                        columns=list(cols),
                        suggestion_text=grp.suggestion.get("suggestion", ""),
                        source_query_indices=sorted(grp.query_indices),
                        reason=(
                            f"Subsumed by a composite index covering ({', '.join(sorted(superset))}) "
                            "on the same table — a single index satisfying the larger column set "
                            "makes this narrower one redundant."
                        ),
                        superseded_by_columns=sorted(superset),
                    )
                )
                primary_superset_group.query_indices = sorted(
                    set(primary_superset_group.query_indices) | set(grp.query_indices)
                )

    # ── Step 4: emit reconciled entries for every group that survived —
    # done last, after Step 3's query_indices absorption above, so a
    # surviving superset's satisfies_queries includes the queries that
    # only asked for the (now-dropped) subset. ──────────────────────────
    reconciled: list[ReconciledSuggestion] = []
    for table_key, col_tuples in table_columns.items():
        display = table_display[table_key]
        for cols in col_tuples:
            if (table_key, cols) in dropped_group_keys:
                continue
            grp = groups[(table_key, cols)]
            reconciled.append(
                ReconciledSuggestion(
                    table=display,
                    suggestion=grp.suggestion,
                    satisfies_queries=sorted(grp.query_indices),
                )
            )

    return BatchReconciliationResult(
        reconciled_suggestions=reconciled,
        dropped_suggestions=dropped,
        column_order_conflicts=column_order_conflicts,
        warnings=warnings,
    )
