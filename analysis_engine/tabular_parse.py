"""
Generic delimited-tabular text parsing — turning what a DB client prints
or a spreadsheet/grid exports into `list[dict[str, str]]`, keyed by a
lowercased header row. Not SQL-specific and not tied to any one feature:

  - plan_parsers/mysql.py's MySQL EXPLAIN pipe-table / `\\G` vertical
    parsing (Issue #62's gap-followup) used to define these two shapes
    locally; moved here so batch_parsers.py (Phase 5 #115/#120 — pasted
    pg_stat_statements / performance_schema / Query Store exports) can
    reuse the exact same row-extraction instead of a second copy. Same
    reasoning plan_crossref.py already applies to reusing
    index_recommender.py's _resolve_real_table: one implementation, not
    two that can drift.
  - TSV and CSV are new here (batch_parsers.py's need) — TSV because SQL
    Server Management Studio's grid "Copy" is tab-separated with a header
    row; CSV because it's the generic "export to CSV" escape hatch every
    DB client offers.
"""

from __future__ import annotations

import csv
import io
import re

# Two pipe-table shapes, unified into one parser:
#   mysql-cli (boxed, leading/trailing "|" on every row):
#     +----+-------------+--------+------+-------+
#     | id | select_type | table  | type | rows  |
#     +----+-------------+--------+------+-------+
#     |  1 | SIMPLE      | orders | ALL  | 10000 |
#     +----+-------------+--------+------+-------+
#   psql (infix only — no leading/trailing "|"):
#                 query                 | calls | total_time
#     ---------------------------------+-------+------------
#      SELECT * FROM orders WHERE ...  |   120 |    4521.33
# Both separator-line styles ("+----+----+" / "-----+-----+") contain no
# "|" character at all, so requiring "|" in the line already excludes
# them — no separate separator-line pattern needed.
#
# `EXPLAIN ... \G` (MySQL) vertical output, e.g.:
#   *************************** 1. row ***************************
#              id: 1
#           table: orders
_VERTICAL_ROW_HEADER_RE = re.compile(r"^\*+\s*\d+\.\s*row\s*\*+\s*$")
_VERTICAL_FIELD_RE = re.compile(r"^\s*([A-Za-z_][\w ]*)\s*:\s*(.*)$")


def parse_pipe_table_rows(raw: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in raw.splitlines():
        if "|" not in line:
            continue
        stripped = line.strip()
        # Strip one optional leading/trailing "|" (mysql-cli's boxed
        # style) — psql's infix style has none to strip; splitting on "|"
        # afterward works the same either way.
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        cells = [c.strip() for c in stripped.split("|")]
        if header is None:
            header = [c.lower() for c in cells]
            continue
        rows.append(dict(zip(header, cells, strict=False)))
    return rows


def parse_vertical_rows(raw: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in raw.splitlines():
        if _VERTICAL_ROW_HEADER_RE.match(line.strip()):
            current = {}
            rows.append(current)
            continue
        if current is None:
            continue
        m = _VERTICAL_FIELD_RE.match(line)
        if m:
            current[m.group(1).strip().lower()] = m.group(2).strip()
    return rows


def parse_delimited_rows(raw: str, delimiter: str) -> list[dict[str, str]]:
    """Header row + N data rows, split on `delimiter` — TSV (SSMS grid
    copy, delimiter='\\t') and CSV (delimiter=',') are the same shape,
    just a different separator. Uses csv.reader (not str.split) so a
    quoted field containing the delimiter itself doesn't break the row."""
    reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
    rows_raw = [row for row in reader if any(cell.strip() for cell in row)]
    if len(rows_raw) < 2:
        return []
    header = [c.strip().lower() for c in rows_raw[0]]
    return [dict(zip(header, [c.strip() for c in row], strict=False)) for row in rows_raw[1:]]


def parse_tsv_rows(raw: str) -> list[dict[str, str]]:
    return parse_delimited_rows(raw, delimiter="\t")


def parse_csv_rows(raw: str) -> list[dict[str, str]]:
    return parse_delimited_rows(raw, delimiter=",")
