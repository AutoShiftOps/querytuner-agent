from __future__ import annotations

import re
from typing import Any

import sqlparse

# -----------------------------
# Top-level scanning helpers
# -----------------------------


def _strip_trailing_semicolon(s: str) -> str:
    return (s or "").strip().rstrip(";").strip()


def _split_top_level_commas(s: str) -> list[str]:
    """
    Split by commas only when not inside parentheses or quotes.
    """
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_sq = False  # '
    in_dq = False  # "

    for ch in s:
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif not in_sq and not in_dq:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
                continue
        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_alias(expr: str) -> tuple[str, str | None]:
    """
    Returns: (expression_without_alias, alias_or_none)

    Handles:
      - expr AS alias
      - expr alias   (best-effort)
    """
    e = _strip_trailing_semicolon(expr)

    # Prefer: AS alias
    m = re.search(r"\s+AS\s+([A-Za-z_][A-Za-z0-9_]*|\"[^\"]+\")\s*$", e, flags=re.IGNORECASE)
    if m:
        alias = m.group(1).strip().strip('"')
        expr_wo = e[: m.start()].strip()
        return expr_wo, alias

    # Best-effort: trailing alias without AS
    m2 = re.search(r"\s+([A-Za-z_][A-Za-z0-9_]*|\"[^\"]+\")\s*$", e)
    if m2:
        alias = m2.group(1).strip().strip('"')
        if alias.lower() not in {
            "from",
            "where",
            "group",
            "order",
            "limit",
            "over",
            "join",
        }:
            expr_wo = e[: m2.start()].strip()
            # Accept only if it looks like an expression (reduces false positives)
            if expr_wo and (
                ("(" in expr_wo)
                or (" " in expr_wo)
                or ("+" in expr_wo)
                or ("-" in expr_wo)
                or ("/" in expr_wo)
                or ("*" in expr_wo)
            ):
                return expr_wo, alias

    return e, None


def _find_top_level_keyword_pos(sql: str, keyword: str, start_idx: int = 0) -> int:
    """
    Find the first top-level occurrence of `keyword` (case-insensitive),
    where top-level means paren depth == 0 and not inside quotes.

    keyword should be provided in lowercase, e.g. " from " or " order by ".
    """
    s_normalised = re.sub(r"\s+", " ", sql)
    s = s_normalised
    sl = s_normalised.lower()
    key = keyword.lower()

    depth = 0
    in_sq = False
    in_dq = False

    i = start_idx
    while i < len(s):
        ch = s[i]

        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif not in_sq and not in_dq:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0:
                if sl.startswith(key, i):
                    return i

        i += 1

    return -1


def _extract_clause(sql: str, start_kw: str, end_kws: list[str]) -> str:
    """
    Extract clause body appearing after start_kw up to the earliest of end_kws, all at top-level.
    Example: start_kw=" where ", end_kws=[" group by ", " order by ", " limit ", " fetch "]
    """
    sql = re.sub(r"\s+", " ", sql).strip()
    start = _find_top_level_keyword_pos(sql, start_kw, 0)
    if start == -1:
        return ""

    body_start = start + len(start_kw)
    end_positions = []
    for ek in end_kws:
        p = _find_top_level_keyword_pos(sql, ek, body_start)
        if p != -1:
            end_positions.append(p)

    body_end = min(end_positions) if end_positions else len(sql)
    return _strip_trailing_semicolon(sql[body_start:body_end].strip())


# -----------------------------
# QueryParser
# -----------------------------


class QueryParser:
    """
    JSON-safe SQL parser for common single-statement SELECT queries.

    Goals:
    - Never return sqlparse token objects (only JSON-serializable types)
    - Avoid false matches for ORDER BY / GROUP BY inside window functions (OVER ...)
    - Provide both:
        - columns: list[str] of output names
        - select_items: list[{"output","expr","has_alias"}]
    """

    def parse(self, query: str) -> dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return self._empty()

        stmt = sqlparse.parse(q)[0]  # used only to detect query type safely

        select_items = self._extract_select_items(q)
        columns = [it["output"] for it in select_items if it.get("output")]

        parsed = {
            "query_type": self._get_query_type(stmt),
            "tables": self._extract_tables(q),
            "columns": self._dedupe_keep_order(columns),
            "select_items": select_items,
            "joins": self._extract_joins(q),
            "where_clause": self._extract_where(q),
            "group_by": self._extract_group_by(q),
            "having_clause": self._extract_having(q),
            "order_by": self._extract_order_by(q),
            "subqueries": self._count_subqueries(q),
            "complexity_score": self._complexity_score(q),
        }
        return parsed

    def _empty(self) -> dict[str, Any]:
        return {
            "query_type": "UNKNOWN",
            "tables": [],
            "columns": [],
            "select_items": [],
            "joins": [],
            "where_clause": "",
            "group_by": [],
            "having_clause": "",
            "order_by": [],
            "subqueries": 0,
            "complexity_score": 0.0,
        }

    def _get_query_type(self, stmt) -> str:
        tok = stmt.token_first(skip_ws=True, skip_cm=True)
        if not tok:
            return "UNKNOWN"
        val = getattr(tok, "normalized", None) or getattr(tok, "value", None) or "UNKNOWN"
        return str(val).upper()

    # -----------------------------
    # SELECT items (top-level)
    # -----------------------------

    def _extract_select_items(self, query: str) -> list[dict[str, Any]]:
        select_text = self._extract_select_clause_text(query)
        if not select_text:
            return []

        raw_items = _split_top_level_commas(select_text)
        items: list[dict[str, Any]] = []

        for raw in raw_items:
            expr_wo_alias, alias = _parse_alias(raw)
            output = alias if alias else self._best_output_name(expr_wo_alias)
            items.append(
                {
                    "output": output,
                    "expr": expr_wo_alias.strip(),
                    "has_alias": bool(alias),
                }
            )

        # de-dupe by (output, expr)
        seen = set()
        out = []
        for it in items:
            key = (it["output"], it["expr"])
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    def _extract_select_clause_text(self, query: str) -> str:
        q = _strip_trailing_semicolon(query)

        s_idx = _find_top_level_keyword_pos(q, "select", 0)
        if s_idx == -1:
            return ""

        # Find FROM at top level after SELECT
        from_idx = _find_top_level_keyword_pos(q, " from ", s_idx)
        if from_idx == -1:
            # SELECT without FROM; return everything after SELECT
            return q[s_idx + len("select") :].strip()

        return q[s_idx + len("select") : from_idx].strip()

    def _best_output_name(self, expr: str) -> str:
        """
        If no alias, try to return a reasonable output label.
        - For simple identifiers: id, table.col => last part
        - For functions/expressions: return the whole expr (shortened)
        """
        e = expr.strip()
        # simple identifier or qualified name
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", e):
            return e.split(".")[-1]
        # wildcard
        if e == "*":
            return "*"
        # fallback: expression itself (trim a bit)
        return e[:80] + ("…" if len(e) > 80 else "")

    # -----------------------------
    # FROM / tables / joins
    # -----------------------------

    def _extract_tables(self, query: str) -> list[str]:
        """
        Best-effort: grab the first table token after top-level FROM.
        Also includes joined tables from JOIN parsing.
        """
        q = _strip_trailing_semicolon(query)

        from_body = _extract_clause(q, " from ", [" where ", " group by ", " order by ", " limit ", " fetch "])
        if not from_body:
            # Could be SELECT without FROM
            return self._dedupe_keep_order([j["table"] for j in self._extract_joins(q)])

        # If there are joins, they will be captured separately.
        # For base table, take the first token before JOIN/commas
        base = from_body
        # cut at first top-level join keyword
        cut = len(base)
        for kw in [
            " join ",
            " left join ",
            " right join ",
            " inner join ",
            " full join ",
            " cross join ",
        ]:
            p = base.lower().find(kw)
            if p != -1:
                cut = min(cut, p)
        base = base[:cut].strip()

        # if multiple tables in FROM (comma joins), take them top-level split
        base_tables = _split_top_level_commas(base)
        base_tables = [self._clean_table_ref(t) for t in base_tables if t.strip()]
        join_tables = [j["table"] for j in self._extract_joins(q)]

        return self._dedupe_keep_order(base_tables + join_tables)

    def _clean_table_ref(self, t: str) -> str:
        """
        Turn 'orders o' or 'schema.orders AS o' into 'orders' or 'schema.orders'
        without being too smart.
        """
        tt = t.strip()
        # remove trailing alias
        # split on whitespace at top-level (no parens expected here usually)
        parts = tt.split()
        if not parts:
            return tt
        # handle "schema.table"
        return parts[0].strip().strip('"')

    def _extract_joins(self, query: str) -> list[dict[str, str]]:
        q = query
        joins = []
        for m in re.finditer(
            r"\b(left|right|inner|full|cross)?\s*join\s+([a-zA-Z0-9_.\"]+)",
            q,
            flags=re.I,
        ):
            raw_type = (m.group(1) or "").strip()
            join_type = (raw_type if raw_type else "JOIN").upper()
            table = m.group(2).strip().strip('"')
            if table:  # never append empty table name
                joins.append({"type": join_type, "table": table})
        return joins

    # -----------------------------
    # WHERE / GROUP BY / ORDER BY (top-level only)
    # -----------------------------

    def _extract_where(self, query: str) -> str:
        q = _strip_trailing_semicolon(query)
        return _extract_clause(q, " where ", [" group by ", " order by ", " limit ", " fetch "])

    def _extract_group_by(self, query: str) -> list[str]:
        q = _strip_trailing_semicolon(query)
        body = _extract_clause(q, " group by ", [" having ", " order by ", " limit ", " fetch "])
        if not body:
            return []
        return [c.strip() for c in _split_top_level_commas(body) if c.strip()]

    def _extract_having(self, query: str) -> str:
        q = _strip_trailing_semicolon(query)
        return _extract_clause(q, " having ", [" order by ", " limit ", " fetch "])

    def _extract_order_by(self, query: str) -> list[str]:
        q = _strip_trailing_semicolon(query)
        body = _extract_clause(q, " order by ", [" limit ", " fetch "])
        if not body:
            return []
        return [c.strip() for c in _split_top_level_commas(body) if c.strip()]

    # -----------------------------
    # Heuristics helpers
    # -----------------------------

    def _count_subqueries(self, query: str) -> int:
        # counts additional SELECT occurrences beyond the first
        return max(0, len(re.findall(r"\bselect\b", query, flags=re.I)) - 1)

    def _complexity_score(self, query: str) -> float:
        q = query.upper()
        score = 0.0
        score += 5.0 * len(re.findall(r"\bJOIN\b", q))
        score += 15.0 * self._count_subqueries(query)
        score += 10.0 * len(re.findall(r"\bUNION\b", q))
        # count top-level clauses via our extractors
        if self._extract_where(query):
            score += 5.0
        if self._extract_group_by(query):
            score += 5.0
        if self._extract_order_by(query):
            score += 5.0
        return float(min(score, 100.0))

    def _dedupe_keep_order(self, xs: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in xs:
            if not x:
                continue
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out


# -----------------------------------------------------------------------------
# Schema DDL parsing (Issue #8)
#
# Standalone from QueryParser: these two functions read CREATE TABLE / CREATE
# INDEX DDL (not SELECT queries) so index_recommender can cross-reference a
# real schema instead of guessing column existence from the query text alone.
# -----------------------------------------------------------------------------

_SCHEMA_SKIP_COL_NAMES = frozenset({"id", "pk", "uuid", "rowid", "rownum", "level"})

# Keywords that mark a top-level CREATE TABLE body line as a constraint/index
# definition rather than a column definition. KEY/INDEX/CONSTRAINT aren't real
# column names either (MySQL's bare KEY/INDEX table-level syntax, and named
# CONSTRAINT wrappers around PRIMARY KEY/UNIQUE/FOREIGN KEY/CHECK).
_CONSTRAINT_KEYWORDS = frozenset({"PRIMARY", "UNIQUE", "FOREIGN", "CHECK", "CONSTRAINT", "KEY", "INDEX"})

_QUOTED_IDENT = r'"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*'

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    rf"(?P<name>{_QUOTED_IDENT})(?:\s*\.\s*(?P<name2>{_QUOTED_IDENT}))?"
    r"\s*\(",
    re.IGNORECASE,
)

_CREATE_INDEX_RE = re.compile(
    rf"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:{_QUOTED_IDENT})\s+ON\s+"
    rf"(?P<table>{_QUOTED_IDENT})(?:\s*\.\s*(?P<table2>{_QUOTED_IDENT}))?"
    r"\s*\((?P<cols>[^)]*)\)",
    re.IGNORECASE | re.DOTALL,
)

_COLUMN_NAME_RE = re.compile(
    r'^\s*(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([A-Za-z_][A-Za-z0-9_]*))\s+(.*)$',
    re.DOTALL,
)

# Canonical type -> raw type tokens/aliases seen across dialects.
_TYPE_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "integer": (
        "int",
        "int2",
        "int4",
        "int8",
        "integer",
        "bigint",
        "smallint",
        "tinyint",
        "mediumint",
        "serial",
        "bigserial",
        "smallserial",
    ),
    "text": (
        "varchar",
        "varchar2",
        "nvarchar",
        "nvarchar2",
        "char",
        "nchar",
        "text",
        "ntext",
        "clob",
        "nclob",
        "string",
        "longtext",
        "mediumtext",
        "tinytext",
    ),
    "timestamp": ("timestamp", "timestamptz", "datetime", "datetime2", "smalldatetime"),
    "date": ("date",),
    "boolean": ("bool", "boolean", "bit"),
    "numeric": ("numeric", "decimal", "number", "money", "smallmoney", "real", "float", "float4", "float8", "double"),
    "uuid": ("uuid", "uniqueidentifier"),
    "json": ("json", "jsonb"),
}
_TYPE_ALIASES: dict[str, str] = {
    alias: canonical for canonical, aliases in _TYPE_ALIAS_GROUPS.items() for alias in aliases
}


def _find_matching_close_paren(s: str, open_idx: int) -> int:
    """s[open_idx] must be '('. Returns the index of the matching ')', or -1."""
    depth = 0
    in_sq = False
    in_dq = False
    i = open_idx
    while i < len(s):
        ch = s[i]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif not in_sq and not in_dq:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _clean_identifier(raw: str) -> str:
    """Strip quotes/backticks/brackets and a trailing ASC/DESC from one identifier."""
    s = raw.strip()
    s = re.sub(r"\s+(ASC|DESC)\s*$", "", s, flags=re.IGNORECASE).strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    elif len(s) >= 2 and s[0] == "`" and s[-1] == "`":
        s = s[1:-1]
    elif len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        s = s[1:-1]
    return s.strip()


def _clean_table_name(raw: str) -> str:
    """Strip quoting; drop a schema prefix: public.orders -> orders, [dbo].[Orders] -> Orders."""
    return _clean_identifier(raw.strip().split(".")[-1])


def _normalize_type(raw_type: str) -> str:
    base = raw_type.strip().split("(")[0].strip().lower()
    base = base.split()[0] if base else base
    return _TYPE_ALIASES.get(base, base)


def _is_constraint_or_index_line(part: str) -> bool:
    m = re.match(r"^\s*([A-Za-z_]+)", part)
    if not m:
        return False
    return m.group(1).upper() in _CONSTRAINT_KEYWORDS


# Matches a PRIMARY KEY/UNIQUE clause's own explicit column list, e.g.
# "PRIMARY KEY CLUSTERED ([pk] ASC)" or "UNIQUE NONCLUSTERED (a, b)". Used to
# find a constraint's column list wherever it appears in a body line — not
# just when the line starts with PRIMARY/UNIQUE/CONSTRAINT — because real
# SQL Server DDL sometimes omits the comma before a table-level
# "CONSTRAINT ... PRIMARY KEY CLUSTERED (...)" clause, gluing it onto the
# end of the preceding column definition instead.
_INLINE_PK_UNIQUE_RE = re.compile(
    r"\b(?:PRIMARY\s+KEY|UNIQUE)\b(?:\s+CLUSTERED\b|\s+NONCLUSTERED\b)?\s*\(([^)]*)\)",
    re.IGNORECASE,
)


def _constraint_indexed_columns(part: str) -> set[str]:
    """Column names covered by a table-level PRIMARY KEY / UNIQUE / bare KEY|INDEX line.

    Deliberately excludes FOREIGN KEY / CHECK — an FK column is not indexed
    just because it references another table (the whole reason JOIN keys
    need explicit index review).
    """
    upper = part.upper()
    starts_key_or_index = bool(re.match(r"^\s*(KEY|INDEX)\b", part, re.IGNORECASE))
    if "PRIMARY" not in upper and "UNIQUE" not in upper and not starts_key_or_index:
        return set()
    m = re.search(r"\(([^)]*)\)", part)
    if not m:
        return set()
    return {_clean_identifier(c) for c in m.group(1).split(",") if c.strip()}


def _iter_create_table_blocks(ddl: str):
    """Yield (table_name, [top_level_comma_parts]) for every CREATE TABLE block."""
    for m in _CREATE_TABLE_RE.finditer(ddl):
        open_idx = m.end() - 1
        close_idx = _find_matching_close_paren(ddl, open_idx)
        if close_idx == -1:
            continue
        raw_name = m.group("name2") or m.group("name")
        table = _clean_table_name(raw_name)
        body = ddl[open_idx + 1 : close_idx]
        yield table, _split_top_level_commas(body)


def parse_schema_ddl(ddl: str) -> dict[str, dict[str, str]]:
    """
    Best-effort CREATE TABLE parser: {table: {column: normalised_type}}.

    Skips constraint/index lines (PRIMARY KEY, UNIQUE, FOREIGN KEY, CHECK) and
    primary-key-style column names (id, pk, uuid, rowid, rownum, level) since
    those are assumed already indexed and shouldn't be re-suggested downstream.
    """
    schema: dict[str, dict[str, str]] = {}
    for table, parts in _iter_create_table_blocks(ddl or ""):
        columns: dict[str, str] = {}
        for part in parts:
            part = part.strip()
            if not part or _is_constraint_or_index_line(part):
                continue
            cm = _COLUMN_NAME_RE.match(part)
            if not cm:
                continue
            col_name = next(g for g in cm.group(1, 2, 3, 4) if g is not None)
            if col_name.lower() in _SCHEMA_SKIP_COL_NAMES:
                continue
            rest = cm.group(5).strip()
            # Issue #126: SQL Server DDL sometimes bracket-wraps the type
            # itself, not just identifiers — [varchar](50), [uniqueidentifier],
            # [int]. The bracket-less pattern this used to be
            # (`r"[A-Za-z_][A-Za-z0-9_]*"`) doesn't match at all when `rest`
            # starts with "[" (`type_m` was None), so the column was
            # silently dropped from the schema entirely — not just
            # mis-normalized. Confirmed via a real SQL Server CREATE TABLE
            # with every column bracket-typed: parse_schema_ddl() returned
            # {} for the whole table before this fix, silently disabling
            # every schema-verified check for SQL Server DDL wholesale.
            # `\[?...\]?` strips one optional matching pair around the type
            # name itself, same idea _clean_identifier already applies to
            # column/table names — [varchar](50) -> "varchar" (the (50)
            # never being in scope for either regex, bracketed or not).
            type_m = re.match(r"\[?([A-Za-z_][A-Za-z0-9_]*)\]?", rest)
            if not type_m:
                continue
            columns[col_name] = _normalize_type(type_m.group(1))
        if columns:
            schema.setdefault(table, {}).update(columns)
    return schema


def get_indexed_columns(ddl: str) -> dict[str, set[str]]:
    """
    Best-effort scan for columns that already have an index: standalone
    CREATE INDEX / CREATE UNIQUE INDEX statements, PRIMARY KEY and UNIQUE
    constraints (inline on a column or table-level), and MySQL's bare
    KEY/INDEX table-level syntax. Used by index_recommender to suppress
    suggestions for columns that are already indexed.
    """
    ddl = ddl or ""
    indexed: dict[str, set[str]] = {}

    for m in _CREATE_INDEX_RE.finditer(ddl):
        raw_table = m.group("table2") or m.group("table")
        table = _clean_table_name(raw_table)
        cols = {_clean_identifier(c) for c in m.group("cols").split(",") if c.strip()}
        if cols:
            indexed.setdefault(table, set()).update(cols)

    for table, parts in _iter_create_table_blocks(ddl):
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if _is_constraint_or_index_line(part):
                cols = _constraint_indexed_columns(part)
                if cols:
                    indexed.setdefault(table, set()).update(cols)
                continue
            cm = _COLUMN_NAME_RE.match(part)
            if not cm:
                continue
            col_name = next(g for g in cm.group(1, 2, 3, 4) if g is not None)
            rest = cm.group(5)

            inline_m = _INLINE_PK_UNIQUE_RE.search(rest)
            if inline_m:
                # An explicit column list on the constraint clause is
                # authoritative — the indexed column(s) may not be the one
                # this definition line happens to start with (see
                # _INLINE_PK_UNIQUE_RE docstring above).
                cols = {_clean_identifier(c) for c in inline_m.group(1).split(",") if c.strip()}
                if cols:
                    indexed.setdefault(table, set()).update(cols)
            elif re.search(r"\bPRIMARY\s+KEY\b", rest, re.IGNORECASE) or re.search(r"\bUNIQUE\b", rest, re.IGNORECASE):
                indexed.setdefault(table, set()).add(col_name)

    return indexed
