from __future__ import annotations

import re
from typing import Any

# Issue #73: dialect-aware rewrites
from analysis_engine.dialect_config import get_dialect


class QueryOptimizer:
    """
    Rule-based query rewriter.
    Applies targeted, safe SQL rewrites based on heuristic findings.
    Never changes query semantics silently — uses comments to flag intent.
    Called by SQLAnalyzerAgent._compose_optimized_query().
    """

    def rewrite(self, query: str, suggestions: list[dict[str, Any]], db_type: str) -> str:
        q = query.strip()
        cfg = get_dialect(db_type)  # Issue #73: dialect config for this rewrite session

        header_lines: list[str] = [
            "-- QueryTuner: Suggested optimized SQL (review before applying)",
            f"-- Dialect: {cfg.display}",  # Issue #73: show dialect display name not raw string
        ]
        applied: list[str] = []

        # --- Rewrite 1: YEAR(col) = N → dialect-correct range condition ---
        q, changed = self._rewrite_year_function(q, db_type)
        if changed:
            applied.append(f"YEAR() rewritten as range condition ({cfg.display} syntax) to enable index seek")

        # --- Rewrite 2: MONTH(col) = N → range condition ---
        q, changed = self._rewrite_month_function(q)
        if changed:
            applied.append("MONTH() rewritten as range condition to enable index seek")

        # --- Rewrite 3: LOWER(col) = 'val' comment ---
        q, changed = self._rewrite_lower_function(q, db_type)
        if changed:
            applied.append(
                f"LOWER() in WHERE flagged — use {cfg.display} case-insensitive approach: {cfg.like_ci.splitlines()[0]}"
            )

        # --- Rewrite 4: SELECT * placeholder ---
        q, changed = self._rewrite_select_star(q)
        if changed:
            applied.append("SELECT * replaced with column placeholder — list only required columns")

        # --- Rewrite 5: LIKE leading wildcard comment (#28) ---
        # LIKE '%value' prevents index usage — flag it with an inline comment
        q, changed = self._rewrite_like_leading_wildcard(q)
        if changed:
            applied.append(
                "LIKE leading wildcard flagged — index cannot be used; consider full-text search or prefix match"
            )
        ql = q.lower()
        has_order_by = bool(re.search(r"\border\s+by\b", ql, re.IGNORECASE))
        has_limit = (
            bool(re.search(r"\blimit\b", ql))
            or bool(re.search(r"\bfetch\s+first\b", ql))
            or "rownum" in ql
            or bool(re.search(r"\btop\s+\d", ql))
        )
        if has_order_by and not has_limit:
            q, changed = self._suggest_limit(q, db_type)
            if changed:
                applied.append(f"Pagination placeholder added ({cfg.display} syntax)")

        # --- Rewrite 6: Correlated subquery → CTE hint ---
        q, changed = self._suggest_cte_for_subquery(q)
        if changed:
            applied.append("Correlated subquery detected — CTE refactor suggested above query")

        if applied:
            header_lines.append("-- Changes applied:")
            for a in applied:
                header_lines.append(f"--   • {a}")
        else:
            header_lines.append("-- No automatic rewrites applied; see suggestions above")

        return "\n".join(header_lines) + "\n\n" + q

    # ------------------------------------------------------------------
    # Individual rewrite methods — each returns (new_query, was_changed)
    # ------------------------------------------------------------------

    def _rewrite_year_function(self, q: str, db_type: str):
        """
        YEAR(col) = 2025 → col BETWEEN '2025-01-01' AND '2025-12-31'
        Issue #73: Oracle uses DATE literal, others use string literal.
        """
        pattern = r"YEAR\s*\(\s*([\w.]+)\s*\)\s*=\s*(\d{4})"

        def _replace(m):
            col = m.group(1).strip()
            yr = m.group(2).strip()
            if db_type == "oracle":
                # Oracle date literals
                return f"{col} >= DATE '{yr}-01-01' AND {col} < DATE '{int(yr)+1}-01-01'"
            elif db_type == "sqlserver":
                # SQL Server uses CAST or string literals
                return f"{col} >= '{yr}-01-01' AND {col} < '{int(yr)+1}-01-01'"
            else:
                # PostgreSQL, MySQL, SQLite — ISO 8601 strings work
                return f"{col} BETWEEN '{yr}-01-01' AND '{yr}-12-31'"

        new_q, n = re.subn(pattern, _replace, q, flags=re.IGNORECASE)
        return new_q, n > 0

    def _rewrite_month_function(self, q: str):
        """MONTH(col) = 3 → col BETWEEN 'YYYY-03-01' AND 'YYYY-03-31' (approximate)"""
        pattern = r"MONTH\s*\(\s*([\w.]+)\s*\)\s*=\s*(\d{1,2})"

        def _replace(m):
            col = m.group(1).strip()
            mo = int(m.group(2).strip())
            mo_s = f"{mo:02d}"
            last = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}.get(mo, 30)
            return f"{col} BETWEEN 'YYYY-{mo_s}-01' AND 'YYYY-{mo_s}-{last}'" f" /* replace YYYY */"

        new_q, n = re.subn(pattern, _replace, q, flags=re.IGNORECASE)
        return new_q, n > 0

    def _rewrite_lower_function(self, q: str, db_type: str):
        """
        LOWER(col) = 'val' → dialect-specific case-insensitive alternative comment.
        Issue #73: PostgreSQL has ILIKE; others use collation.
        """
        pattern = r"LOWER\s*\(\s*([\w.]+)\s*\)\s*=\s*('[^']*')"

        def _replace(m):
            col = m.group(1).strip()
            val = m.group(2).strip()
            if db_type == "postgresql":
                hint = f"/* TIP (PostgreSQL): use {col} ILIKE {val} for case-insensitive match with index support via pg_trgm */"
            elif db_type == "mysql":
                hint = f"/* TIP (MySQL): set ci collation on {col} — then {col} = {val} is case-insensitive without LOWER() */"
            elif db_type == "oracle":
                hint = (
                    f"/* TIP (Oracle): use NLS_UPPER or a function-based index: CREATE INDEX idx ON t(UPPER({col})) */"
                )
            elif db_type == "sqlserver":
                hint = f"/* TIP (SQL Server): use {col} COLLATE SQL_Latin1_General_CP1_CI_AS = {val} */"
            else:
                hint = f"/* TIP: avoid LOWER() in WHERE — use a case-insensitive collation or index on {col} */"
            return f"{hint} {col} = {val}"

        new_q, n = re.subn(pattern, _replace, q, flags=re.IGNORECASE)
        return new_q, n > 0

    def _rewrite_select_star(self, q: str):
        """SELECT * → SELECT /* TODO: list required columns */"""
        pattern = r"\bSELECT\s+\*"
        new_q, n = re.subn(pattern, "SELECT /* TODO: list required columns */", q, flags=re.IGNORECASE)
        return new_q, n > 0

    def _rewrite_like_leading_wildcard(self, q: str):
        """
        Issue #28: LIKE '%value' or LIKE '%value%' → add inline comment.
        A leading wildcard forces a full index scan or full table scan.
        We don't remove the condition (semantics must be preserved),
        but we flag it prominently so the developer sees it.
        Pattern matches: LIKE '%...  (any quote style, any content starting with %)
        """
        pattern = r"(\bLIKE\s+)(['\"])(%[^'\"]*)\2"

        def _replace(m):
            like_kw = m.group(1)
            quote = m.group(2)
            content = m.group(3)
            return (
                f"/* ⚠ LIKE leading wildcard: index cannot be used on this column "
                f"— consider a full-text index or rewrite as prefix match if possible */ "
                f"{like_kw}{quote}{content}{quote}"
            )

        new_q, n = re.subn(pattern, _replace, q, flags=re.IGNORECASE)
        return new_q, n > 0

    def _suggest_limit(self, q: str, db_type: str):
        """
        Append dialect-correct pagination syntax at end of query.
        Issue #73: dialect-aware pagination comment per DB type.
        """
        cfg = get_dialect(db_type)
        q_stripped = "\n".join(line.rstrip() for line in q.rstrip().rstrip(";").splitlines())
        display = cfg.display  # e.g. "SQL Server", "Oracle", "PostgreSQL"

        if db_type == "sqlserver":
            # TOP N goes after SELECT — too risky to auto-inject; use OFFSET/FETCH comment
            return (
                q_stripped + f"\n-- TODO ({display}): ORDER BY col OFFSET 0 ROWS FETCH NEXT 100 ROWS ONLY",
                True,
            )
        elif db_type == "oracle":
            return q_stripped + f"\nFETCH FIRST 100 ROWS ONLY  -- {display}: adjust N; use ROWNUM for 11g", True
        elif db_type == "sqlite":
            return q_stripped + f"\nLIMIT 100  -- {display}: adjust to your page size", True
        else:
            # PostgreSQL, MySQL — standard LIMIT/OFFSET
            return q_stripped + f"\nLIMIT 100  -- {display}: adjust to your page size", True

    def _suggest_cte_for_subquery(self, q: str):
        """If nested SELECTs exist, prepend a CTE refactor hint."""
        nested = len(re.findall(r"\bSELECT\b", q, re.IGNORECASE)) - 1
        if nested < 1:
            return q, False

        cte_hint = (
            "/* QueryTuner CTE hint: this query has nested subqueries that may be refactored.\n"
            "   Example pattern:\n"
            "   WITH subquery_name AS (\n"
            "       SELECT ... FROM ... WHERE ...\n"
            "   )\n"
            "   SELECT ... FROM main_table JOIN subquery_name ON ...\n"
            "*/\n"
        )
        return cte_hint + q, True
