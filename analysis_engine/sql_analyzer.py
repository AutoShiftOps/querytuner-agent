from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from analysis_engine.explainer import QueryExplainer
from analysis_engine.optimizer import QueryOptimizer
from analysis_engine.schemas import DatabaseType
from analysis_engine.schemas import QueryRequest as QR
from analysis_engine.execution_planner import collect_facts
from analysis_engine.index_recommender import IndexRecommender
from analysis_engine.plan_crossref import cross_reference_plan
from analysis_engine.query_parser import QueryParser, parse_schema_ddl

logger = logging.getLogger(__name__)

# Three-tier evidence labels for heuristic (non-index-recommender) findings.
# Deterministic: the pattern is always correct regardless of data distribution.
# Everything else falls back to "needs-runtime-evidence" — pattern-based,
# cannot be confirmed without live DB stats or an EXPLAIN plan.
_DETERMINISTIC_TYPES = frozenset(
    {
        "cartesian_join",
        "like_wildcard",
        "function_in_where",
        "implicit_cast",
        "subquery_to_join",
        "column_selection",
        "not_in_nullable",
        "case_in_predicate",
    }
)


@dataclass
class AnalyzerConfig:
    max_query_chars: int = 32000

    @staticmethod
    def from_env() -> AnalyzerConfig:
        return AnalyzerConfig(
            max_query_chars=int(os.getenv("MAX_QUERY_CHARS", "32000")),
        )


class SQLAnalyzerAgent:
    """
    SQL Analyzer Agent — deterministic heuristics + schema/EXPLAIN-plan
    cross-referencing, vendored from QueryTuner (AutoShiftOps/querytuner,
    backend/app/agents/sql_analyzer.py) for this hackathon build's
    run_analyze_step.

    Deliberately zero AI-provider calls anywhere in this class or
    anything it imports — no OpenAI, no Hugging Face. QueryTuner's
    production version of this file optionally calls an LLM for a
    supplementary "AI insights" pass; that entire branch has been removed
    here, not just left unused, so this engine has no AI-provider
    awareness of its own to audit away. This project's only AI calls
    (Gemma for triage, Gemini for the final explain step) live entirely
    in gemini_agent.py, downstream of this class's plain, structured
    output — matching the hackathon's Google-AI-stack-only constraint by
    construction rather than by convention.
    """

    def __init__(self, config: AnalyzerConfig | None = None):
        self.config = config or AnalyzerConfig.from_env()
        self.parser = QueryParser()
        self.optimizer = QueryOptimizer()
        self.explainer = QueryExplainer()
        self.index_recommender = IndexRecommender()

    async def analyze(
        self,
        query: str,
        db_type: str,
        schema_info: str | None = None,
        focus: str = "performance",
        explain_plan: str | None = None,  # Issue #60: EXPLAIN plan paste-in
    ) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            raise ValueError("Query is empty")

        if len(query) > self.config.max_query_chars:
            raise ValueError(f"Query too large (>{self.config.max_query_chars} chars)")

        schema_info = (schema_info or "").strip()
        explain_plan = (explain_plan or "").strip()
        focus = (focus or "performance").strip().lower()

        parsed = self._safe_parse(query)
        security_issues = self._security_checks(query)
        readability_score = self._readability_score(query, parsed)
        suggestions = self._heuristic_suggestions(
            query,
            parsed,
            db_type=db_type,
            focus=focus,
            schema_info=schema_info,  # Phase 2: pass schema through
        )

        # Issue #61/#62/#63: parse a pasted EXPLAIN plan (or fall back to a
        # live DSN connection, Postgres only) and cross-reference it
        # against the index suggestions just computed. Deliberately done
        # here — before the optimizer/explainer/LLM below, all of which
        # also read `suggestions` — so everything downstream sees the
        # final, plan-informed evidence_level rather than a stale
        # pre-cross-reference state. Previously this collect_facts() call
        # happened last, after everything else, and its result was never
        # used for anything but the raw `facts` field in the API response.
        facts_result = None
        try:
            _req = QR(
                query=query,
                db_type=db_type if isinstance(db_type, DatabaseType) else DatabaseType(db_type),
                explain_plan=explain_plan or None,
                schema_info=schema_info or None,
            )
            _facts, _plan_nodes = await collect_facts(_req)
            facts_result = _facts.model_dump()
            if _plan_nodes:
                _plan_schema = parse_schema_ddl(schema_info) if schema_info else {}
                suggestions = cross_reference_plan(suggestions, _plan_nodes, _plan_schema)
            # Gap-followup (docs/querytuner-explain-parser-gap-followup.md,
            # #63 item 4): filesort_detected/temp_table_detected are
            # MySQL-Extra-field-based and only ever detectable from a
            # parsed plan — no static SQL-text heuristic produces them the
            # way the rest of _heuristic_suggestions()'s entries do, so
            # there's nothing for cross_reference_plan() to "confirm."
            # Promoted straight from the plan's own findings into
            # optimization_suggestions instead, already schema-verified
            # since they come directly from the plan, not a guess.
            suggestions.extend(self._plan_native_suggestions(_facts.findings))
        except Exception as _e:
            facts_result = {
                "db_type": db_type,
                "warnings": [f"Plan collection skipped: {str(_e)}"],
                "findings": [],
            }

        optimized_query = self.optimizer.rewrite(query, suggestions, db_type=db_type)

        plain_explanation = self.explainer.explain(
            query=query,
            parsed=parsed,
            suggestions=suggestions,
            db_type=db_type,
            security_issues=security_issues,
            schema_info=schema_info,  # Phase 2: pass schema through
        )

        # No AI-provider call anywhere in this module, deliberately — see
        # the module docstring. What used to be an optional OpenAI/
        # Hugging Face "ai_insights" pass here has been removed entirely
        # rather than left dormant; gemini_agent.py's own `explain` step
        # (Gemini) is this project's only AI call over this function's
        # output, downstream of run_analyze_step, not inside it.
        return {
            "parsing_result": parsed,
            "optimization_suggestions": suggestions,
            "optimized_query": optimized_query,
            "plain_explanation": plain_explanation,
            "security_issues": security_issues,
            "readability_score": readability_score,
            "facts": facts_result,
        }

    # -------------------------
    # Parsing / safety
    # -------------------------

    def _safe_parse(self, query: str) -> dict[str, Any]:
        try:
            parsed = self.parser.parse(query)
            if not isinstance(parsed, dict):
                return {}
            return parsed
        except Exception:
            return {}

    # -------------------------
    # Heuristics
    # -------------------------

    def _heuristic_suggestions(
        self,
        query: str,
        parsed: dict[str, Any],
        db_type: str,
        focus: str,
        schema_info: str | None = None,
    ) -> list[dict[str, Any]]:
        q = query.strip()
        ql = q.lower()

        subqueries = parsed.get("subqueries") or 0
        complexity = parsed.get("complexity_score") or 0

        suggestions: list[dict[str, Any]] = []

        # 1) SELECT *
        if re.search(r"\bselect\s+\*", ql):
            suggestions.append(
                self._suggest(
                    type_="column_selection",
                    severity="medium",
                    suggestion="Avoid SELECT *; specify only needed columns",
                    reason="Reduces I/O and memory usage; can improve planning and network transfer",
                    estimated="5-15% faster (varies)",
                )
            )

        # 2) Missing WHERE for SELECT (risky)
        has_select = re.search(r"\bselect\b", ql, re.IGNORECASE) is not None
        has_where = re.search(r"\bwhere\b", ql, re.IGNORECASE) is not None

        if has_select and not has_where:
            suggestions.append(
                self._suggest(
                    type_="full_scan_risk",
                    severity="medium",
                    suggestion="Query has no WHERE clause; ensure this is intentional",
                    reason="May scan entire table(s), especially costly on large datasets",
                    estimated="Varies",
                )
            )

        # 3) LIKE with leading wildcard (ILIKE included — PostgreSQL case-insensitive LIKE)
        if re.search(r"\bi?like\s+'%[^']*'", ql):
            suggestions.append(
                self._suggest(
                    type_="like_wildcard",
                    severity="high",
                    suggestion="Leading-wildcard LIKE (e.g. LIKE '%abc') cannot use a B-tree index",
                    reason="Consider full-text search or a trigram index (pg_trgm for PostgreSQL, FULLTEXT for MySQL)",
                    estimated="Often large — full index scan avoided",
                )
            )

        # 4) Functions on columns in WHERE
        _fn_pattern = (
            r"\bwhere\b.*\b("
            r"lower|upper|trim|ltrim|rtrim|substr|substring|"
            r"cast|convert|"
            r"date|year|month|day|datepart|datename|extract|"
            r"to_date|to_char|to_number|"
            r"isnull|ifnull|nvl|coalesce|nullif|"
            r"abs|round|floor|ceil|ceiling|length|len|"
            r"md5|sha|sha1|sha2"
            r")\s*\("
        )
        if re.search(_fn_pattern, ql, re.DOTALL | re.IGNORECASE):
            suggestions.append(
                self._suggest(
                    type_="function_in_where",
                    severity="high",
                    suggestion="Avoid wrapping filtered columns in functions inside WHERE",
                    reason=(
                        "Functions on indexed columns prevent index seeks. "
                        "Rewrite as a range condition (e.g. YEAR(col)=2025 → col BETWEEN '2025-01-01' AND '2025-12-31')"
                    ),
                    estimated="Often large — enables index seek instead of full scan",
                )
            )

        # 5) ORDER BY without LIMIT
        # Uses the parser's top-level order_by (excludes ORDER BY inside window
        # function OVER(...) clauses) instead of a raw whole-string regex.
        if (
            bool(parsed.get("order_by"))
            and not bool(re.search(r"\blimit\b", ql))
            and not bool(re.search(r"\bfetch\s+first\b", ql))
            # SQL Server pagination: OFFSET @n ROWS FETCH NEXT @n ROWS ONLY.
            # Without this, every paginated SQL Server query using its native
            # syntax (instead of the PostgreSQL/Oracle-style FETCH FIRST)
            # falsely fires "missing pagination".
            and not bool(re.search(r"\boffset\b.*\bfetch\s+next\b", ql, re.DOTALL))
        ):
            suggestions.append(
                self._suggest(
                    type_="order_by_no_limit",
                    severity="medium",
                    suggestion="Consider adding LIMIT/FETCH FIRST for user-facing queries with ORDER BY",
                    reason="Sorting large result sets is expensive; limiting reduces sort work",
                    estimated="Varies",
                )
            )

        # 6) Too many JOINs
        join_count = ql.count(" join ")
        if join_count >= 4:
            suggestions.append(
                self._suggest(
                    type_="join_complexity",
                    severity="high",
                    suggestion=f"Query has {join_count} JOINs; review join order, keys, and filter pushdown",
                    reason="Many joins can amplify row counts and increase planner complexity",
                    estimated="Varies",
                )
            )

        # 6.5) Cartesian JOIN — JOIN without ON or USING
        cartesian_joins = re.findall(
            r"\bJOIN\s+\S+(?:\s+\w+)?\s*(?!ON\b|USING\b)(?=\s+(?:JOIN|WHERE|GROUP|ORDER|LIMIT|FETCH|$)|\s*;|$)",
            q,
            re.IGNORECASE | re.DOTALL,
        )
        if cartesian_joins:
            suggestions.append(
                self._suggest(
                    type_="cartesian_join",
                    severity="critical",
                    suggestion=(
                        f"Cartesian JOIN detected ({len(cartesian_joins)} occurrence(s)) — "
                        f"JOIN used without ON or USING clause"
                    ),
                    reason=(
                        "A JOIN without ON/USING produces a cartesian product: every row in the left "
                        "table is matched with every row in the right table. On tables with 1k rows each "
                        "this returns 1,000,000 rows. Almost always a bug."
                    ),
                    estimated="Query may return exponentially more rows than intended",
                )
            )

        # 7) Subquery count — generic refactor suggestion
        if isinstance(subqueries, int) and subqueries >= 2:
            suggestions.append(
                self._suggest(
                    type_="subquery_refactor",
                    severity="medium",
                    suggestion="Consider refactoring nested subqueries into CTEs (WITH) or JOINs where appropriate",
                    reason="Improves readability; may improve planning depending on the DB and query shape",
                    estimated="Varies",
                )
            )

        # 8) Issue #25: implicit_cast — detect type coercion patterns in WHERE
        # Catches: PostgreSQL cast operator (::), SQL Server CONVERT(type, col),
        # and ID columns compared to string literals (e.g. WHERE user_id = '123')
        _implicit_cast_patterns = [
            # PostgreSQL cast operator in WHERE: col::type
            (r"\bwhere\b.*[\w.]+\s*::\s*\w+", "PostgreSQL :: cast operator in WHERE prevents index use"),
            # SQL Server / MySQL CONVERT(type, col) — already caught by function_in_where,
            # but flag explicitly here for type-coercion context
            (r"\bconvert\s*\(\s*\w+\s*,\s*[\w.]+\s*\)", "CONVERT() performs an implicit type cast"),
            # ID/FK columns compared to string literals — likely implicit int→varchar cast
            (
                r"\b(user_id|customer_id|order_id|product_id|account_id|tenant_id)\s*=\s*'[^']+'",
                "ID column compared to string literal — implicit cast may prevent index use",
            ),
            # Numeric literal compared to column that looks like a string/code column
            (
                r"\b(status|code|flag|type|kind|role|tier)\s*=\s*\d+\b",
                "String-like column compared to numeric literal — implicit cast may prevent index use",
            ),
        ]

        implicit_cast_reasons = []
        for pattern, reason in _implicit_cast_patterns:
            if re.search(pattern, ql, re.IGNORECASE | re.DOTALL):
                implicit_cast_reasons.append(reason)

        if implicit_cast_reasons:
            suggestions.append(
                self._suggest(
                    type_="implicit_cast",
                    severity="high",
                    suggestion=(
                        "Implicit or explicit type cast detected in WHERE — "
                        "may prevent index use and cause full scans"
                    ),
                    reason=(
                        f"{implicit_cast_reasons[0]}. "
                        "Ensure the comparison value matches the column's data type to allow index seeks."
                    ),
                    estimated="Often significant — removes type-coercion full scan",
                )
            )

        # 9) Issue #26: subquery_to_join — flag correlated subqueries in SELECT list
        # Pattern: SELECT (...) , ... where the subquery references outer columns
        # Detect subqueries directly in the SELECT clause (not just in WHERE)
        select_clause = self._extract_select_clause(q)
        select_subquery_count = len(re.findall(r"\bSELECT\b", select_clause, re.IGNORECASE))
        # select_subquery_count > 0 means there is a nested SELECT in the SELECT list
        if select_subquery_count > 0:
            suggestions.append(
                self._suggest(
                    type_="subquery_to_join",
                    severity="high",
                    suggestion=(
                        f"Correlated subquery detected in SELECT clause "
                        f"({select_subquery_count} occurrence(s)) — consider rewriting as a JOIN or CTE"
                    ),
                    reason=(
                        "A subquery in the SELECT list executes once per row of the outer query. "
                        "On a 10,000-row result set this means 10,000 separate lookups. "
                        "A LEFT JOIN or CTE is evaluated once and is far more efficient."
                    ),
                    estimated="Often 10x–100x faster for large result sets",
                )
            )

        # 10) Complexity score
        try:
            c = float(complexity)
            if c >= 70:
                suggestions.append(
                    self._suggest(
                        type_="high_complexity",
                        severity="medium",
                        suggestion=(
                            "High complexity query: consider splitting into steps, "
                            "using temp tables, or pre-aggregation"
                        ),
                        reason="Complex queries are harder to optimize and maintain",
                        estimated="Varies",
                    )
                )
        except Exception:
            pass

        # 11) Column-level index recommendations
        index_suggestions = self.index_recommender.recommend(
            query=q,
            parsed=parsed,
            db_type=db_type,
            schema_info=schema_info,  # Phase 2: pass schema through
        )
        suggestions.extend(index_suggestions)

        # Boost complexity score based on HIGH index findings
        high_index_count = sum(1 for s in index_suggestions if s.get("severity") == "high")
        if high_index_count > 0:
            base_score = float(parsed.get("complexity_score") or 0)
            parsed["complexity_score"] = min(100.0, base_score + (high_index_count * 8.0))

        # 12) Focus-specific
        if focus == "security":
            suggestions.append(
                self._suggest(
                    type_="security_best_practice",
                    severity="medium",
                    suggestion="Ensure application uses parameterized queries (no string concatenation)",
                    reason="Reduces SQL injection risk and improves query plan caching",
                    estimated="Risk reduction",
                )
            )

        # 13) not_in_nullable — NOT IN with a subquery that could contain NULLs.
        # Deterministic: SQL three-valued logic means any NULL in the subquery
        # silently zeroes out every row, regardless of the actual data.
        if re.search(r"\bnot\s+in\s*\(\s*select\b", ql, re.IGNORECASE):
            suggestions.append(
                self._suggest(
                    type_="not_in_nullable",
                    severity="high",
                    suggestion="NOT IN with a subquery can return zero rows if the subquery contains any NULL values",
                    reason=(
                        "SQL three-valued logic: NOT IN propagates NULLs. If any row in the subquery is NULL, "
                        "the entire NOT IN condition evaluates to UNKNOWN, returning no rows. Use NOT EXISTS "
                        "instead: WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.col = outer.col)"
                    ),
                    estimated="Correctness fix — may also improve performance by enabling index use",
                )
            )

        # 14) case_in_predicate — CASE expression in WHERE prevents index use.
        # Anchored after WHERE (mirrors function_in_where) so a CASE in the
        # SELECT list doesn't false-positive — same index-blocking problem class.
        if re.search(r"\bwhere\b.*\bcase\b.*\bwhen\b", ql, re.DOTALL | re.IGNORECASE):
            suggestions.append(
                self._suggest(
                    type_="case_in_predicate",
                    severity="high",
                    suggestion="CASE expression in WHERE clause prevents index usage on the evaluated column",
                    reason=(
                        "The database cannot use a B-tree index when a column is wrapped in a CASE expression "
                        "in the WHERE clause. Refactor to use direct comparisons: instead of WHERE CASE WHEN "
                        "status = 'active' THEN 1 END = 1, use WHERE status = 'active'"
                    ),
                    estimated="Often significant — enables index seek vs full scan",
                )
            )

        # 15) or_expansion — OR in WHERE may prevent efficient index use.
        # Not deterministic: harmless on small tables, so this is an estimate.
        # Uses the parsed where_clause (not the raw query) to avoid matching
        # OR that appears in the SELECT list or HAVING clause.
        where_clause = parsed.get("where_clause") or ""
        if re.search(r"\bor\b", where_clause, re.IGNORECASE):
            suggestions.append(
                self._suggest(
                    type_="or_expansion",
                    severity="medium",
                    suggestion="OR conditions on different columns may prevent efficient index use",
                    reason=(
                        "The database must evaluate each OR branch separately. On large tables this forces "
                        "two index scans or a full table scan with bitmap OR. Consider rewriting as UNION ALL: "
                        "SELECT ... WHERE col1 = 'a' UNION ALL SELECT ... WHERE col2 = 'b' AND col1 != 'a'"
                    ),
                    estimated="Varies — significant on large tables, negligible on small ones. Verify with EXPLAIN.",
                )
            )

        # 16) cte_multiple_references — a CTE referenced 2+ times may re-execute.
        # In MySQL/SQL Server, CTEs re-execute per reference; PostgreSQL 12+
        # inlines them by default. Impact depends on dialect, so this is an
        # estimate rather than a deterministic finding.
        cte_names = re.findall(r"(?:\bwith\b|,)\s*(\w+)\s+as\s*\(", ql, re.IGNORECASE)
        for cte_name in dict.fromkeys(cte_names):
            total_occurrences = len(re.findall(rf"\b{re.escape(cte_name)}\b", ql))
            reference_count = total_occurrences - 1  # subtract the CTE's own definition
            if reference_count >= 2:
                suggestions.append(
                    self._suggest(
                        type_="cte_multiple_references",
                        severity="medium",
                        suggestion=(
                            f"CTE '{cte_name}' is referenced {reference_count} times — "
                            f"may execute multiple times depending on database dialect"
                        ),
                        reason=(
                            "In MySQL and SQL Server, referencing a CTE multiple times causes it to re-execute "
                            "on each reference. In PostgreSQL 12+, CTEs are inlined by default. Consider "
                            "materialising with CREATE TEMP TABLE or using a subquery with a lateral join if "
                            "the CTE is expensive. PostgreSQL: add /*+ MATERIALIZED */ hint."
                        ),
                        estimated="Depends on CTE complexity and row count — verify with EXPLAIN",
                    )
                )

        return self._dedupe_suggestions(suggestions)

    # -------------------------
    # Security / readability
    # -------------------------

    def _security_checks(self, query: str) -> list[str]:
        ql = query.lower()
        issues: list[str] = []

        if ql.count(";") >= 2:
            issues.append("Multiple SQL statements detected; consider restricting to a single statement")

        for op in (" drop ", " truncate ", " alter ", " grant ", " revoke "):
            if op in f" {ql} ":
                issues.append(f"Potentially destructive/admin operation detected: {op.strip().upper()}")

        if "--" in query or "/*" in query:
            issues.append("SQL comments detected; ensure input is trusted/parameterized")

        if " union " in f" {ql} ":
            issues.append("UNION detected; validate inputs and prefer parameterized queries")

        if "||" in query or "concat(" in ql:
            issues.append("String concatenation detected; use parameterized queries to reduce injection risk")

        return issues

    def _readability_score(self, query: str, parsed: dict[str, Any]) -> float:
        score = 100.0

        complexity = parsed.get("complexity_score", 0) or 0
        try:
            score -= min(float(complexity) * 0.3, 30.0)
        except Exception:
            pass

        if re.search(r"\bselect\s+\*\b", query, re.IGNORECASE):
            score -= 10

        if query.count("\n") < 2:
            score -= 15

        if len(query) > 1500:
            score -= 10

        return float(max(0.0, min(100.0, score)))

    # -------------------------
    # Helpers
    # -------------------------

    def _extract_select_clause(self, query: str) -> str:
        """Extract everything between SELECT and FROM at top level."""
        q = re.sub(r"\s+", " ", query).strip()
        ql = q.lower()
        sel = ql.find("select")
        if sel == -1:
            return ""
        frm = ql.find(" from ", sel)
        if frm == -1:
            return q[sel + 6 :]
        return q[sel + 6 : frm]

    # -------------------------
    # Utilities
    # -------------------------

    def _suggest(self, type_: str, severity: str, suggestion: str, reason: str, estimated: str) -> dict[str, Any]:
        return {
            "type": type_,
            "severity": severity,
            "suggestion": suggestion,
            "reason": reason,
            "estimated_improvement": estimated,
            "evidence_level": ("deterministic" if type_ in _DETERMINISTIC_TYPES else "needs-runtime-evidence"),
        }

    # Gap-followup: maps the plan-only Finding types mysql.py's parser can
    # produce (see plan_parsers/mysql.py's _extra_findings) to the
    # user-facing suggestion copy — kept here rather than in mysql.py
    # since it's presentation text for optimization_suggestions, the same
    # layer every other _suggest() call in this file lives at.
    _PLAN_NATIVE_SUGGESTION_COPY = {
        "filesort_detected": (
            "high",
            "MySQL is using a filesort to satisfy this query's ORDER BY",
            (
                "A filesort means no index covers the sort order, so MySQL sorts rows in memory "
                "(or on disk, for large result sets) after fetching them — confirmed directly from "
                "your pasted EXPLAIN output's 'Using filesort'."
            ),
            "Often significant — an index matching ORDER BY can avoid the sort entirely",
        ),
        "temp_table_detected": (
            "high",
            "MySQL is materializing a temporary table to run this query",
            (
                "Common with GROUP BY/DISTINCT/ORDER BY combinations the available indexes can't "
                "satisfy directly — confirmed directly from your pasted EXPLAIN output's "
                "'Using temporary'."
            ),
            "Often significant — a covering index for the GROUP BY/ORDER BY columns can eliminate the temp table",
        ),
    }

    def _plan_native_suggestions(self, plan_findings: list[Any]) -> list[dict[str, Any]]:
        """Converts filesort_detected/temp_table_detected Finding objects
        (plan_parsers/mysql.py) into optimization_suggestions entries.
        Already evidence_level="schema-verified" — unlike every other
        suggestion in this file, these were never a guess plan_crossref.py
        needed to confirm; they came directly from the plan."""
        out: list[dict[str, Any]] = []
        seen_types = set()
        for finding in plan_findings or []:
            f_type = getattr(finding, "type", None) or (finding.get("type") if isinstance(finding, dict) else None)
            copy = self._PLAN_NATIVE_SUGGESTION_COPY.get(f_type)
            if not copy or f_type in seen_types:
                continue
            seen_types.add(f_type)
            severity, suggestion, reason, estimated = copy
            out.append(
                {
                    "type": f_type,
                    "severity": severity,
                    "suggestion": suggestion,
                    "reason": reason,
                    "estimated_improvement": estimated,
                    "evidence_level": "schema-verified",
                    "plan_verified": True,
                }
            )
        return out

    def _dedupe_suggestions(self, suggestions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set = set()
        out: list[dict[str, Any]] = []
        for s in suggestions:
            key = (s.get("type"), s.get("suggestion"))
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out
