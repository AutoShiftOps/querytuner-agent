from __future__ import annotations

import re
from typing import Any

# Issue #8: schema-aware confirmation — cross-reference recommendations against
# a real CREATE TABLE/CREATE INDEX DDL instead of guessing from the query alone.
from analysis_engine.query_parser import get_indexed_columns, parse_schema_ddl

# Issue #72: dialect-aware index DDL
from analysis_engine.dialect_config import get_dialect, get_index_ddl

_PRIMARY_KEY_NAMES = frozenset({"id", "pk", "oid", "rowid", "uuid", "rownum", "level", "sysdate", "systimestamp"})


def _is_primary_key(col: str) -> bool:
    return col.lower() in _PRIMARY_KEY_NAMES


def _extract_col_name(expr: str) -> str | None:
    e = expr.strip()
    e = re.sub(r"\s+(ASC|DESC)\s*$", "", e, flags=re.IGNORECASE).strip()

    # Quoted identifier, optionally alias-qualified: "Col Name" or alias."Col Name"
    # (standard SQL / Postgres / SQL Server double-quote/bracket style) —
    # or `Col Name` / alias.`Col Name` (MySQL backtick style). Issue #120:
    # MySQL's performance_schema.digest_text normalizes every identifier
    # to backtick-quoted form (`` SELECT * FROM `orders` WHERE `status` =
    # ? ``) — without this, no column from a pasted performance_schema
    # batch export was ever recognized, silently producing zero
    # suggestions for every MySQL batch entry. Found via
    # test_batch_parsers.py's normalized-text-caveat tests, which the
    # design doc explicitly asked to verify rather than assume.
    quoted_m = re.match(r'^(?:[A-Za-z_][A-Za-z0-9_]*\.)?(?:"([^"]+)"|`([^`]+)`)', e)
    if quoted_m:
        return quoted_m.group(1) or quoted_m.group(2)

    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(", e):
        return None
    e = re.split(r"[=<>!]", e)[0].strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", e):
        return e
    return None


def _qualified_col(expr: str) -> tuple[str | None, str] | None:
    col = _extract_col_name(expr)
    if not col:
        return None
    parts = col.split(".")
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, parts[0]


# Issue #117: which operators count as "equality" vs "range/inequality" for
# composite-index column ordering — standard guidance is that equality
# predicates should lead a composite index (they narrow to an exact seek
# point), while range/inequality predicates and sort-only columns trail,
# since the index can only maintain a useful scan order for columns coming
# after the first range predicate. IN and IS NULL are treated as equality
# (a set of discrete lookups / a single-value match, not a range scan).
_EQUALITY_OPERATORS = {"=", "IN"}


def _extract_where_columns(where_clause: str) -> list[tuple[str | None, str, str]]:
    """
    Returns (alias, column, predicate_type) triples — predicate_type is
    "equality" or "range". Used by _detect_composite_opportunity to order
    composite columns; callers that only need (alias, column) can ignore
    the third element.
    """
    if not where_clause:
        return []
    cols: list[tuple[str | None, str, str]] = []
    normalised = re.sub(r"\s+", " ", where_clause).strip()
    conditions = re.split(r"\bAND\b|\bOR\b", normalised, flags=re.IGNORECASE)
    for cond in conditions:
        cond = cond.strip()
        if not cond:
            continue
        # Issue #120: `[A-Za-z_][A-Za-z0-9_.]+|"[^"]+"` alone never
        # matched a MySQL-backtick-quoted condition start (`` `status` =
        # ? ``, exactly the form performance_schema.digest_text
        # normalizes every identifier to) — every batch entry from a
        # pasted performance_schema export silently produced zero WHERE
        # suggestions before this. Same fix as _extract_col_name's own
        # backtick handling above; found by the same test.
        is_null_m = re.match(r"^([A-Za-z_][A-Za-z0-9_.]+|`[^`]+`)\s+IS\s+(NOT\s+)?NULL$", cond, re.IGNORECASE)
        if is_null_m:
            qc = _qualified_col(is_null_m.group(1))
            if qc:
                cols.append((*qc, "equality"))
            continue
        m = re.match(
            r'^([A-Za-z_][A-Za-z0-9_.]+|"[^"]+"|`[^`]+`)\s*'
            r"(=|!=|<>|<=|>=|<|>|\bI?LIKE\b|\bNOT\s+IN\b|\bIN\b|\bBETWEEN\b)",
            cond,
            re.IGNORECASE,
        )
        if m:
            qc = _qualified_col(m.group(1))
            if qc:
                operator = m.group(2).upper()
                predicate_type = "equality" if operator in _EQUALITY_OPERATORS else "range"
                cols.append((*qc, predicate_type))
    return cols


_ON_CLAUSE_END = r"(?:\bJOIN\b|\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\bHAVING\b|;|$)"


def _extract_join_columns(joins_raw: list[dict[str, Any]], query: str) -> list[tuple[str | None, str]]:
    cols: list[tuple[str | None, str]] = []

    # ON clause — loop over every AND-joined col=col pair, not just the first
    for on_m in re.finditer(r"\bON\s+(.*?)(?=" + _ON_CLAUSE_END + r")", query, re.IGNORECASE | re.DOTALL):
        for cond in re.split(r"\bAND\b", on_m.group(1), flags=re.IGNORECASE):
            cond_m = re.match(
                r"^\s*([A-Za-z_][A-Za-z0-9_.]+)\s*=\s*([A-Za-z_][A-Za-z0-9_.]+)\s*$",
                cond.strip(),
            )
            if cond_m:
                for grp in (cond_m.group(1), cond_m.group(2)):
                    qc = _qualified_col(grp)
                    if qc:
                        cols.append(qc)

    # USING (col[, col...]) — shared column name, no left/right alias
    for using_m in re.finditer(r"\bUSING\s*\(\s*([^)]+?)\s*\)", query, re.IGNORECASE):
        for col_expr in using_m.group(1).split(","):
            qc = _qualified_col(col_expr.strip())
            if qc:
                cols.append(qc)

    return cols


def _extract_order_by_columns(order_by: list[str]) -> list[tuple[str | None, str]]:
    cols: list[tuple[str | None, str]] = []
    for expr in order_by:
        qc = _qualified_col(expr)
        if qc:
            cols.append(qc)
    return cols


def _extract_group_by_columns(group_by: list[str]) -> list[tuple[str | None, str]]:
    cols: list[tuple[str | None, str]] = []
    for expr in group_by:
        qc = _qualified_col(expr)
        if qc:
            cols.append(qc)
    return cols


def _resolve_real_table(alias: str | None, schema: dict[str, dict[str, str]]) -> str | None:
    """
    Best-effort alias -> real table name resolution against a parsed schema.

    Resolution order:
      a. Exact match: alias in schema
      b. Prefix match: any table name starts with alias
      c. First-letter match: alias == table_name[0]
    """
    if not alias or not schema:
        return None
    if alias in schema:
        return alias
    for table_name in schema:
        if table_name.startswith(alias):
            return table_name
    for table_name in schema:
        if table_name and alias == table_name[0]:
            return table_name
    return None


def _describe_column_role(
    col: str,
    equality_cols: list[str],
    join_cols: list[str],
    range_cols: list[str],
    order_cols: list[str],
) -> str:
    """Short, human-readable reason a column landed where it did in the
    composite's column order — checked in the same priority order the
    composite itself is built in, so a column that's e.g. both an
    equality filter AND a JOIN key is described by its higher-priority role."""
    if col in equality_cols:
        return "equality filter"
    if col in join_cols:
        return "JOIN key"
    if col in range_cols:
        return "range filter"
    if col in order_cols:
        return "sort column"
    return "query column"


def _composite_ordering_note(
    cols: list[str],
    equality_cols: list[str],
    join_cols: list[str],
    range_cols: list[str],
    order_cols: list[str],
) -> str:
    """Issue #117: makes the column-ordering reasoning visible in the
    suggestion text itself, not just applied silently — e.g. "Column
    order: `status` (equality filter) -> `customer_id` (JOIN key) ->
    `created_at` (sort column)"."""
    if len(cols) < 2:
        return ""
    parts = [f"`{c}` ({_describe_column_role(c, equality_cols, join_cols, range_cols, order_cols)})" for c in cols]
    return " Column order: " + " -> ".join(parts) + "."


def _detect_composite_opportunity(
    where_cols: list[tuple[str | None, str, str]],
    join_cols: list[tuple[str | None, str]],
    order_by_cols: list[tuple[str | None, str]],
    db_type: str = "postgresql",  # Issue #72: added db_type param
    schema: dict[str, dict[str, str]] | None = None,  # Issue #8: schema-aware table resolution
) -> list[dict[str, Any]]:
    from collections import defaultdict

    schema = schema or {}

    # Issue #117: bucket columns by role (not just by alias) so the
    # composite's column list can be built in standard composite-index
    # order — equality WHERE predicates first, then JOIN keys, then
    # range/inequality WHERE predicates, then columns that only appear in
    # ORDER BY — instead of raw WHERE-then-JOIN-then-ORDER-BY extraction
    # order. A composite index that leads with a range or sort-only column
    # performs close to (or worse than) a single-column index: the index
    # can only preserve a useful scan order for columns coming after the
    # first range predicate. Order *within* each bucket stays extraction
    # order (Issue #117's explicit non-goal — no cardinality/selectivity
    # reasoning within a category).
    equality_by_alias: dict[str, list[str]] = defaultdict(list)
    range_by_alias: dict[str, list[str]] = defaultdict(list)
    join_by_alias: dict[str, list[str]] = defaultdict(list)
    order_by_alias: dict[str, list[str]] = defaultdict(list)

    def _add(bucket: dict[str, list[str]], alias: str | None, col: str) -> None:
        if alias and not _is_primary_key(col) and col not in bucket[alias]:
            bucket[alias].append(col)

    for alias, col, predicate_type in where_cols:
        _add(equality_by_alias if predicate_type == "equality" else range_by_alias, alias, col)
    for alias, col in join_cols:
        _add(join_by_alias, alias, col)
    for alias, col in order_by_cols:
        _add(order_by_alias, alias, col)

    # Aliases in first-seen order across the priority buckets — a plain
    # `set()` union would work too but doesn't guarantee stable iteration
    # order, and this keeps composite output order deterministic.
    aliases: list[str] = []
    for bucket in (equality_by_alias, join_by_alias, range_by_alias, order_by_alias):
        for alias in bucket:
            if alias not in aliases:
                aliases.append(alias)

    composites = []
    for alias in aliases:
        equality_cols = equality_by_alias.get(alias, [])
        join_cols_ = join_by_alias.get(alias, [])
        range_cols = range_by_alias.get(alias, [])
        order_cols = order_by_alias.get(alias, [])

        cols: list[str] = []
        for bucket_cols in (equality_cols, join_cols_, range_cols, order_cols):
            for col in bucket_cols:
                if col not in cols:
                    cols.append(col)

        if len(cols) < 2:
            continue

        # Issue #72: use dialect-correct DDL for composite index
        # Issue #8: prefer the real table name over the alias placeholder when known
        real_table = _resolve_real_table(alias, schema)
        idx_name = f"idx_{real_table or alias}_{'_'.join(cols)}"
        table_ph = real_table if real_table else f"<{alias}_table>"
        col_list = ", ".join(cols)
        ddl = get_index_ddl(db_type, table_ph, col_list, idx_name)
        note = get_dialect(db_type).index_ddl_note()
        ordering_note = _composite_ordering_note(cols, equality_cols, join_cols_, range_cols, order_cols)

        composites.append(
            {
                "table_alias": alias,
                "columns": cols,
                "suggestion": (
                    f"Consider a composite index on `{alias}` table columns "
                    f"({', '.join(f'`{c}`' for c in cols)}) — "
                    f"all appear in JOIN/WHERE/ORDER BY together."
                    f"{ordering_note}"
                ),
                "ddl_hint": ddl,
                "ddl_note": note,
            }
        )
    return composites


_LOW_CARDINALITY_PATTERNS = re.compile(
    r"\b(status|type|flag|is_[a-z_]+|active|enabled|deleted|state|role|" r"gender|priority|category|kind|mode|tier)\b",
    re.IGNORECASE,
)


def _is_low_cardinality(col_name: str) -> bool:
    return bool(_LOW_CARDINALITY_PATTERNS.search(col_name))


# Issue #118: write/storage cost estimate — the counterpart to
# estimated_improvement, which only ever states the read-side benefit.
# Heuristic-only, v1: no live DB access means no real row counts or
# EXPLAIN output to back a byte-precise estimate with, so this is
# deliberately coarse, based only on what's actually knowable at
# suggestion time — column count and (when schema_info was pasted)
# column data type.
# Matched against schema[table][col] AS STORED — i.e. already run through
# query_parser.py's _normalize_type(), not the raw DDL text. That matters:
# _normalize_type() (a) strips length specs entirely, so VARCHAR(20) and
# VARCHAR(2000) are indistinguishable by the time anything sees them, and
# (b) _TYPE_ALIAS_GROUPS collapses varchar/varchar2/nvarchar/char/nchar/
# text/clob/nclob ALL into one canonical bucket, "text". That means "text"
# can't be trusted as a large-column signal here — it's exactly as likely
# to be a 1-character status flag as an unbounded free-text column.
# Confirmed empirically during manual verification: matching "text" as a
# substring flagged a VARCHAR(20) status column as "large-text... adds
# storage overhead per row", which is simply wrong. Only types that stay
# genuinely distinguishable after normalization are treated as large:
# canonical "json" (json/jsonb get their own bucket, not collapsed into
# "text"), and blob/xml/image variants, which _TYPE_ALIAS_GROUPS doesn't
# alias at all and so pass through _normalize_type() unchanged. Extending
# query_parser.py to preserve length or split varchar from text is
# explicitly out of scope for #118 ("no new parsing pass needed") —
# this list is deliberately what's honestly knowable from the existing
# normalized schema, not a wishlist of "large" SQL types in general.
_LARGE_NORMALIZED_TYPES = frozenset({"json", "blob", "longblob", "mediumblob", "tinyblob", "xml", "image"})


def _is_large_column_type(col_type: str | None) -> bool:
    if not col_type:
        return False
    return col_type.strip().lower() in _LARGE_NORMALIZED_TYPES


def _estimate_index_cost(
    columns: list[str],
    table_ph: str,
    schema: dict[str, dict[str, str]],
) -> str:
    """
    A simple three-tier write/storage cost label:
      - 1 column, no large-text type -> "Low write cost"
      - 2-3 columns, or a large-text column involved -> "Moderate write cost"
      - 4+ columns -> "Higher write cost — N-column composite"
    More columns means more write amplification (every INSERT/UPDATE/DELETE
    touches every column in the index) and a larger index; a large-text
    column (TEXT/VARCHAR/JSON/...) costs more storage per row than a plain
    int/date, so it bumps an otherwise-low tier up one level. table_ph is
    reused as the schema lookup key as-is — when it's a real resolved table
    name this finds real column types; when it's an unresolved
    "<alias_table>" placeholder, the lookup just finds nothing and this
    falls back to the count-only estimate, which is exactly the intended
    fallback (no separate "is this schema-verified" branch needed).
    """
    n = len(columns)
    table_schema = schema.get(table_ph, {}) if schema else {}
    large_type_cols = [c for c in columns if _is_large_column_type(table_schema.get(c))]

    if n >= 4:
        label = f"Higher write cost — {n}-column composite"
    elif n >= 2 or large_type_cols:
        label = f"Moderate write cost — {n}-column composite" if n >= 2 else "Moderate write cost"
    else:
        label = "Low write cost"

    if large_type_cols:
        cols_label = ", ".join(f"`{c}`" for c in large_type_cols)
        label += f" ({cols_label} is a large-text column — adds storage overhead per row)"

    return label


class IndexRecommender:
    def recommend(
        self,
        query: str,
        parsed: dict[str, Any],
        db_type: str = "postgresql",
        schema_info: str | None = None,
    ) -> list[dict[str, Any]]:
        suggestions: list[dict[str, Any]] = []

        # Issue #8: parse schema DDL (if provided) to confirm/suppress recommendations
        schema = parse_schema_ddl(schema_info) if schema_info else {}
        already_indexed = get_indexed_columns(schema_info) if schema_info else {}

        where_clause = parsed.get("where_clause") or ""
        joins = parsed.get("joins") or []
        order_by = parsed.get("order_by") or []
        group_by = parsed.get("group_by") or []

        where_cols = _extract_where_columns(where_clause)
        join_cols = _extract_join_columns(joins, query)
        order_cols = _extract_order_by_columns(order_by)
        group_cols = _extract_group_by_columns(group_by)

        seen_join_cols: set[str] = set()
        for alias, col in join_cols:
            if _is_primary_key(col):
                continue
            key = f"{alias}.{col}" if alias else col
            if key in seen_join_cols:
                continue
            seen_join_cols.add(key)
            real = self._real_table(alias, schema)
            if real and col in already_indexed.get(real, set()):
                continue
            label = f"`{alias}.{col}`" if alias else f"`{col}`"
            ddl, schema_verified, table_ph = self._ddl_hint(alias, col, db_type, schema)
            suggestions.append(
                self._make(
                    index_type="join_key",
                    severity="high",
                    columns=[key],
                    suggestion=f"JOIN key {label} may lack an index — each matched row triggers a lookup on the joined table",
                    reason="Unindexed JOIN keys cause nested-loop full scans. An index on the foreign key column is one of the highest-ROI changes.",
                    estimated="50-90% faster JOINs on large tables",
                    ddl_hint=ddl,
                    ddl_note=get_dialect(db_type).index_ddl_note(),
                    schema_verified=schema_verified,
                    db_type=db_type,
                    alias=alias,
                    col=col,
                    table_ph=table_ph,
                    schema=schema,
                )
            )

        seen_where_cols: set[str] = set()
        for alias, col, _predicate_type in where_cols:
            if _is_primary_key(col):
                continue
            key = f"{alias}.{col}" if alias else col
            if key in seen_join_cols or key in seen_where_cols:
                continue
            seen_where_cols.add(key)
            real = self._real_table(alias, schema)
            if real and col in already_indexed.get(real, set()):
                continue
            label = f"`{alias}.{col}`" if alias else f"`{col}`"
            if _is_low_cardinality(col):
                ddl, schema_verified, table_ph = self._ddl_partial_hint(alias, col, db_type, schema)
                suggestions.append(
                    self._make(
                        index_type="partial_index_candidate",
                        severity="medium",
                        columns=[key],
                        suggestion=f"WHERE column {label} looks low-cardinality (status/flag/type). A partial index may be more efficient than a full index",
                        reason="Low-cardinality columns have poor selectivity for full indexes. A partial index (WHERE status = 'active') is smaller and faster.",
                        estimated="Significant if active rows are a small subset of the table",
                        ddl_hint=ddl,
                        ddl_note=get_dialect(db_type).index_ddl_note(),
                        schema_verified=schema_verified,
                        db_type=db_type,
                        alias=alias,
                        col=col,
                        table_ph=table_ph,
                        schema=schema,
                    )
                )
            else:
                ddl, schema_verified, table_ph = self._ddl_hint(alias, col, db_type, schema)
                suggestions.append(
                    self._make(
                        index_type="where_filter",
                        severity="high",
                        columns=[key],
                        suggestion=f"WHERE column {label} may lack an index — used as a filter condition",
                        reason="Unindexed WHERE columns force full table or full index scans. Adding a B-tree index enables seek access.",
                        estimated="Often large — enables index seek vs full scan",
                        ddl_hint=ddl,
                        ddl_note=get_dialect(db_type).index_ddl_note(),
                        schema_verified=schema_verified,
                        db_type=db_type,
                        alias=alias,
                        col=col,
                        table_ph=table_ph,
                        schema=schema,
                    )
                )

        seen_order_cols: set[str] = set()
        for alias, col in order_cols:
            if _is_primary_key(col):
                continue
            key = f"{alias}.{col}" if alias else col
            if key in seen_join_cols or key in seen_where_cols or key in seen_order_cols:
                continue
            seen_order_cols.add(key)
            real = self._real_table(alias, schema)
            if real and col in already_indexed.get(real, set()):
                continue
            label = f"`{alias}.{col}`" if alias else f"`{col}`"
            ddl, schema_verified, table_ph = self._ddl_hint(alias, col, db_type, schema)
            suggestions.append(
                self._make(
                    index_type="order_by_index",
                    severity="medium",
                    columns=[key],
                    suggestion=f"ORDER BY column {label} may benefit from an index — avoids filesort on large result sets",
                    reason="Without an index matching the ORDER BY, the DB sorts all matching rows in memory or on disk before returning results.",
                    estimated="Eliminates filesort — often 30-70% faster",
                    ddl_hint=ddl,
                    ddl_note=get_dialect(db_type).index_ddl_note(),
                    schema_verified=schema_verified,
                    db_type=db_type,
                    alias=alias,
                    col=col,
                    table_ph=table_ph,
                    schema=schema,
                )
            )

        for alias, col in group_cols:
            if _is_primary_key(col):
                continue
            key = f"{alias}.{col}" if alias else col
            if key in seen_join_cols or key in seen_where_cols or key in seen_order_cols:
                continue
            real = self._real_table(alias, schema)
            if real and col in already_indexed.get(real, set()):
                continue
            label = f"`{alias}.{col}`" if alias else f"`{col}`"
            ddl, schema_verified, table_ph = self._ddl_hint(alias, col, db_type, schema)
            suggestions.append(
                self._make(
                    index_type="group_by_index",
                    severity="medium",
                    columns=[key],
                    suggestion=f"GROUP BY column {label} may benefit from an index — avoids temporary table for aggregation",
                    reason="An index on GROUP BY columns lets the planner use index scan for grouping instead of a hash aggregate or temp table.",
                    estimated="15-50% faster GROUP BY on large tables",
                    ddl_hint=ddl,
                    ddl_note=get_dialect(db_type).index_ddl_note(),
                    schema_verified=schema_verified,
                    db_type=db_type,
                    alias=alias,
                    col=col,
                    table_ph=table_ph,
                    schema=schema,
                )
            )

        # Issue #72: pass db_type to composite detector; Issue #8: pass schema too
        composites = _detect_composite_opportunity(where_cols, join_cols, order_cols, db_type, schema)
        for comp in composites:
            alias = comp["table_alias"]
            cols = comp["columns"]
            real_table = self._real_table(alias, schema)
            table_ph = real_table if real_table else f"<{alias}_table>"
            suggestions.append(
                self._make(
                    index_type="composite_index",
                    severity="high",
                    columns=[f"{alias}.{c}" for c in cols],
                    suggestion=comp["suggestion"],
                    reason="A composite index covering multiple query columns is more efficient than separate single-column indexes — one index scan satisfies JOIN, filter, and sort in a single pass.",
                    estimated="Often the highest-ROI index change for multi-column queries",
                    ddl_hint=comp["ddl_hint"],
                    ddl_note=comp.get("ddl_note", ""),
                    table_ph=table_ph,
                    schema=schema,
                    composite_bare_columns=cols,
                )
            )

        return self._dedupe(suggestions)

    # Issue #8: alias -> real table name resolution against a parsed schema
    def _real_table(self, alias: str | None, schema: dict[str, dict[str, str]]) -> str | None:
        return _resolve_real_table(alias, schema)

    # Issue #72: replaced hardcoded DDL with dialect_config lookup
    # Issue #8: schema-aware — returns (ddl, schema_verified, table_ph) instead of just ddl
    def _ddl_hint(self, alias: str | None, col: str, db_type: str, schema=None) -> tuple[str, bool, str]:
        schema = schema or {}
        real_table = self._real_table(alias, schema)
        schema_verified = real_table is not None and col in schema.get(real_table, {})
        table_ph = real_table if real_table else (f"<{alias}_table>" if alias else "<table_name>")
        idx_name = f"idx_{real_table or alias or 'tbl'}_{col}"
        ddl = get_index_ddl(db_type, table_ph, col, idx_name)
        return ddl, schema_verified, table_ph

    def _ddl_partial_hint(self, alias: str | None, col: str, db_type: str, schema=None) -> tuple[str, bool, str]:
        """Dialect-correct partial/filtered index DDL. Returns (ddl, schema_verified, table_ph)."""
        schema = schema or {}
        real_table = self._real_table(alias, schema)
        schema_verified = real_table is not None and col in schema.get(real_table, {})
        table_ph = real_table if real_table else (f"<{alias}_table>" if alias else "<table_name>")
        idx = f"idx_{real_table or alias or 'tbl'}_{col}_partial"
        cfg = get_dialect(db_type)

        if db_type == "postgresql":
            ddl = (
                f"CREATE INDEX CONCURRENTLY {idx} ON {table_ph}(id) "
                f"WHERE {col} = '<active_value>';  -- {cfg.index_ddl_note()}"
            )
        elif db_type == "mysql":
            ddl = (
                f"-- MySQL has no native partial indexes.\n"
                f"-- Option A: ALTER TABLE {table_ph} ADD INDEX {idx} ({col}, id);\n"
                f"-- Option B: partition the table by {col} value."
            )
        elif db_type == "sqlserver":
            ddl = f"CREATE NONCLUSTERED INDEX {idx} ON {table_ph}(id) WHERE {col} = '<active_value>' WITH (ONLINE=ON);"
        elif db_type == "oracle":
            ddl = (
                f"-- Oracle: use a function-based index or partition:\n"
                f"CREATE INDEX {idx} ON {table_ph}(id) NOLOGGING;\n"
                f"-- Or: use list partitioning on {col}."
            )
        else:
            # SQLite and fallback
            ddl = f"CREATE INDEX IF NOT EXISTS {idx} ON {table_ph}({col});"

        return ddl, schema_verified, table_ph

    def _make(
        self,
        index_type,
        severity,
        columns,
        suggestion,
        reason,
        estimated,
        ddl_hint,
        ddl_note: str = "",
        schema_verified: bool = False,
        db_type: str = "postgresql",
        alias: str | None = None,
        col: str = "",
        table_ph: str = "",
        evidence_level: str | None = None,
        schema: dict[str, dict[str, str]] | None = None,
        composite_bare_columns: list[str] | None = None,
    ):
        if evidence_level is None:
            evidence_level = "schema-verified" if schema_verified else "needs-runtime-evidence"
        result = {
            "type": f"index_review_{index_type}",
            "severity": severity,
            "schema_verified": schema_verified,
            "evidence_level": evidence_level,
            "columns": columns,
            "suggestion": suggestion,
            "reason": reason,
            "estimated_improvement": estimated,
            "ddl_hint": ddl_hint,
        }
        if ddl_note:
            result["ddl_note"] = ddl_note
        # Issue #118: bare (non-alias-qualified) column names for the
        # schema type lookup — composite_bare_columns for composite_index
        # suggestions (already bare), [col] for every single-column type.
        bare_columns = composite_bare_columns if composite_bare_columns is not None else ([col] if col else None)
        if bare_columns:
            result["cost_estimate"] = _estimate_index_cost(bare_columns, table_ph, schema or {})
        if col and table_ph:
            cfg = get_dialect(db_type)
            result["rollback_ddl"] = cfg.rollback_index.format(
                index=f"idx_{alias or 'tbl'}_{col}",
                table=table_ph,
            )
        return result

    def _dedupe(self, suggestions):
        seen: set[str] = set()
        out = []
        for s in suggestions:
            key = s["suggestion"][:80]
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out
