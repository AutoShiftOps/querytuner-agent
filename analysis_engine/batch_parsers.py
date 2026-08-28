"""
Issue #120 — parses a pasted batch-workload export from one of three
named production sources into a normalized `list[BatchQueryEntry]`:
SQL Server Query Store, PostgreSQL `pg_stat_statements`, MySQL
`performance_schema`. Per the design doc
(docs/querytuner-batch-analysis-issue.md): "user runs a standard export
query in their DB client, pastes the results into QueryTuner."

Explicit format (source) selector, not auto-detected from content — same
reasoning #61/#62's gap-followup gives for not auto-detecting EXPLAIN
dialect: these three sources use overlapping column-naming conventions
(e.g. all three have SOME notion of "average time"), and guessing wrong
would silently misattribute which time unit a number is in.

What IS auto-detected, and deliberately so, is the *wire shape* the user
pasted for their chosen source — JSON array, CSV, TSV (SQL Server
Management Studio's grid "Copy" produces tab-separated text), or a
DB client's own pipe-table printout (psql / mysql-cli) — matching the
same "try the plausible parseable shapes" precedent
plan_parsers/postgres.py's parse_postgres_explain and
plan_parsers/mysql.py's parse_mysql_explain already set for a single,
already-known dialect. This is a narrower kind of detection than
guessing the SOURCE itself.

Normalized-text caveat (design doc item 3): pg_stat_statements and
performance_schema give parameterized query text ("WHERE id = $1" /
"WHERE id = ?"), not the literal SQL a real request used. That's a
correctness risk for the heuristic engine's regex-based column/predicate
extraction, which was built and tested against literal SQL — see
test_batch_parsers.py's placeholder-syntax cases and
test_batch_reconciler.py's coverage of what actually happens downstream.
This module only parses the export; it does not attempt to rewrite
placeholders back into literal-shaped SQL (there is nothing to rewrite
them TO — the literal values were never in the export).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from analysis_engine.tabular_parse import parse_csv_rows, parse_pipe_table_rows, parse_tsv_rows, parse_vertical_rows


@dataclass
class BatchQueryEntry:
    """One row of a parsed batch export, normalized across all three
    source formats so batch_reconciler.py and the /analyze/batch endpoint
    don't need source-specific branches."""

    query_text: str
    calls: int | None
    # Normalized to milliseconds regardless of source (pg_stat_statements:
    # already ms; performance_schema: picoseconds / 1e9;
    # Query Store: microseconds / 1000) — see each parser's docstring for
    # exactly which source column(s) fed this and how "total" was derived
    # when the export only gave an average.
    total_time_ms: float | None
    source: str


def _rows_from_export(raw: str) -> list[dict[str, Any]]:
    """Tries, in order: JSON array of objects, mysql-cli/psql pipe-table,
    MySQL `\\G` vertical (harmless to also try for Postgres/SQL Server —
    it simply won't match anything and returns []), TSV (SSMS grid copy),
    then CSV (the universal "export to CSV" fallback). First shape that
    yields at least one row wins."""
    raw = (raw or "").strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, list) and data and all(isinstance(r, dict) for r in data):
            return data
    except (ValueError, TypeError):
        pass

    for parser in (parse_pipe_table_rows, parse_vertical_rows, parse_tsv_rows, parse_csv_rows):
        rows = parser(raw)
        # A "successful" parse with only one column is a strong signal
        # the wrong delimiter was guessed (e.g. TSV parsing genuinely
        # comma-separated text just produces one giant column per row,
        # never raising) rather than a real single-column export — try
        # the next shape instead of returning a bad guess.
        if rows and len(rows[0]) > 1:
            return rows

    return []


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    """Case-insensitive lookup across several acceptable column-name
    aliases for the same logical field — real exports vary this (e.g.
    Postgres renamed total_time -> total_exec_time in PG13), and a JSON
    export's keys may not be lowercased the way the tabular parsers
    normalize theirs. Returns the first key that's present AND not a
    blank/NULL-ish string; None if none match."""
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key in lowered:
            val = lowered[key]
            if val is None:
                continue
            if isinstance(val, str) and (not val.strip() or val.strip().upper() == "NULL"):
                continue
            return val
    return None


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _to_int(val: Any) -> int | None:
    f = _to_float(val)
    return int(f) if f is not None else None


# ── pg_stat_statements ───────────────────────────────────────────────────


def parse_pg_stat_statements(raw: str) -> list[BatchQueryEntry]:
    """
    Expected columns (any subset; both PG <13 and 13+ naming accepted):
    `query`, `calls`, `total_time`/`total_exec_time`,
    `mean_time`/`mean_exec_time` — all already in **milliseconds**, per
    pg_stat_statements' own documented units. A standard export query:

        SELECT query, calls, total_exec_time, mean_exec_time
        FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 50;
    """
    entries: list[BatchQueryEntry] = []
    for row in _rows_from_export(raw):
        query_text = _first_present(row, "query", "query_text", "sql", "sql_text")
        if not query_text or not str(query_text).strip():
            continue
        calls = _to_int(_first_present(row, "calls", "execution_count", "call_count"))
        total_ms = _to_float(_first_present(row, "total_time", "total_exec_time"))
        if total_ms is None:
            mean_ms = _to_float(_first_present(row, "mean_time", "mean_exec_time"))
            if mean_ms is not None:
                total_ms = mean_ms * calls if calls else mean_ms
        entries.append(
            BatchQueryEntry(
                query_text=str(query_text).strip(),
                calls=calls,
                total_time_ms=total_ms,
                source="pg_stat_statements",
            )
        )
    return entries


# ── MySQL performance_schema ─────────────────────────────────────────────


def parse_performance_schema(raw: str) -> list[BatchQueryEntry]:
    """
    Targets `performance_schema.events_statements_summary_by_digest`'s
    own raw columns — `digest_text`, `count_star`, `sum_timer_wait` /
    `avg_timer_wait` — NOT the friendlier `sys.statement_analysis` view,
    whose time columns are pre-formatted display strings ("4.52 s",
    "372.68 ms") rather than a consistent numeric unit; parsing those
    reliably is out of scope for v1. `sum_timer_wait`/`avg_timer_wait`
    are documented as **picoseconds** — divided by 1e9 for milliseconds.
    A standard export query:

        SELECT digest_text, count_star, sum_timer_wait
        FROM performance_schema.events_statements_summary_by_digest
        ORDER BY sum_timer_wait DESC LIMIT 50;
    """
    _PICOSECONDS_PER_MS = 1_000_000_000
    entries: list[BatchQueryEntry] = []
    for row in _rows_from_export(raw):
        query_text = _first_present(row, "digest_text", "sql_text", "query", "query_text")
        if not query_text or not str(query_text).strip():
            continue
        calls = _to_int(_first_present(row, "count_star", "calls", "execution_count"))
        total_ps = _to_float(_first_present(row, "sum_timer_wait", "total_timer_wait"))
        total_ms = total_ps / _PICOSECONDS_PER_MS if total_ps is not None else None
        if total_ms is None:
            avg_ps = _to_float(_first_present(row, "avg_timer_wait"))
            if avg_ps is not None:
                avg_ms = avg_ps / _PICOSECONDS_PER_MS
                total_ms = avg_ms * calls if calls else avg_ms
        entries.append(
            BatchQueryEntry(
                query_text=str(query_text).strip(),
                calls=calls,
                total_time_ms=total_ms,
                source="performance_schema",
            )
        )
    return entries


# ── SQL Server Query Store ───────────────────────────────────────────────


def parse_query_store(raw: str) -> list[BatchQueryEntry]:
    """
    Targets the columns a `sys.query_store_query_text` /
    `sys.query_store_runtime_stats` join naturally produces —
    `query_sql_text`, `count_executions`, `avg_duration` (and friends,
    all **microseconds** per Query Store's documented units — divided by
    1000 for milliseconds). No native "total_duration" column exists;
    total is derived as avg_duration * count_executions when both are
    present. A standard export query:

        SELECT qt.query_sql_text, rs.count_executions, rs.avg_duration
        FROM sys.query_store_query_text qt
        JOIN sys.query_store_query q ON qt.query_text_id = q.query_text_id
        JOIN sys.query_store_plan p ON q.query_id = p.query_id
        JOIN sys.query_store_runtime_stats rs ON p.plan_id = rs.plan_id
        ORDER BY rs.avg_duration DESC;
    """
    _MICROSECONDS_PER_MS = 1_000
    entries: list[BatchQueryEntry] = []
    for row in _rows_from_export(raw):
        query_text = _first_present(row, "query_sql_text", "query_text", "sql_text", "query")
        if not query_text or not str(query_text).strip():
            continue
        calls = _to_int(_first_present(row, "count_executions", "execution_count", "calls"))
        avg_us = _to_float(_first_present(row, "avg_duration", "avg_duration_us"))
        total_us = _to_float(_first_present(row, "total_duration"))
        if total_us is not None:
            total_ms = total_us / _MICROSECONDS_PER_MS
        elif avg_us is not None:
            avg_ms = avg_us / _MICROSECONDS_PER_MS
            total_ms = avg_ms * calls if calls else avg_ms
        else:
            total_ms = None
        entries.append(
            BatchQueryEntry(
                query_text=str(query_text).strip(),
                calls=calls,
                total_time_ms=total_ms,
                source="query_store",
            )
        )
    return entries


_PARSERS = {
    "pg_stat_statements": parse_pg_stat_statements,
    "performance_schema": parse_performance_schema,
    "query_store": parse_query_store,
}


def parse_batch_export(source: str, raw: str) -> list[BatchQueryEntry]:
    """Dispatches to the parser for an already-user-selected `source`.
    Raises ValueError for an unrecognized source — the API layer is
    responsible for restricting `source` to a known Literal before this
    is ever called, so this is a defensive check, not the primary
    validation."""
    parser = _PARSERS.get(source)
    if parser is None:
        raise ValueError(f"Unknown batch export source: {source!r}")
    return parser(raw)


def rank_top_n(entries: list[BatchQueryEntry], n: int) -> list[BatchQueryEntry]:
    """Sorts by total_time_ms descending ("ranked by actual production
    impact", per #120's own wording) and takes the top `n`. Entries with
    no time signal at all (total_time_ms is None — an export that only
    gave query text) sort last, in original order, rather than being
    dropped — still worth analyzing, just not prioritized over entries
    with a real cost signal."""
    ranked = sorted(
        enumerate(entries),
        key=lambda pair: (pair[1].total_time_ms is None, -(pair[1].total_time_ms or 0), pair[0]),
    )
    return [entry for _, entry in ranked[:n]]
