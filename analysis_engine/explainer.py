from __future__ import annotations

from typing import Any

# Issue #8: schema DDL parsing — used to build the Schema Context section
from analysis_engine.query_parser import get_indexed_columns, parse_schema_ddl

# Issue #74 + #75: dialect-aware LLM context and maintenance commands
from analysis_engine.dialect_config import get_dialect

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


class QueryExplainer:
    """
    Plain-English explanation layer.

    Takes structured heuristic output and produces a human-readable,
    severity-ranked diagnosis — independently of the LLM.
    Runs always (fast, free) and feeds into the LLM prompt as context.

    Issue #74: db_type is now passed through to LLM context (via sql_analyzer)
    Issue #75: adds dialect-specific maintenance commands section
    """

    def explain(
        self,
        query: str,
        parsed: dict[str, Any],
        suggestions: list[dict[str, Any]],
        db_type: str,
        security_issues: list[str] | None = None,
        schema_info: str | None = None,
    ) -> str:
        sections: list[str] = []

        # 1 — Query summary
        sections.append(self._summarize_query(query, parsed, db_type))

        # 2 — Findings
        if suggestions:
            sections.append(self._format_findings_summary(suggestions))
        else:
            sections.append("✅ **No performance issues detected** by heuristic analysis.")

        # 3 — Security
        if security_issues:
            sections.append(self._format_security(security_issues))

        # 4 — Readability tips
        readability_tip = self._readability_tip(query, parsed)
        if readability_tip:
            sections.append(readability_tip)

        # 5 — Issue #75: dialect-specific maintenance commands
        maintenance = self._format_maintenance(suggestions, db_type, parsed)
        if maintenance:
            sections.append(maintenance)

        # Issue #8: schema context — inserted after the query summary, before findings
        if schema_info:
            schema_summary = self._format_schema_summary(schema_info, suggestions, parsed)
            if schema_summary is not None:
                sections.insert(1, schema_summary)

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _summarize_query(self, query: str, parsed: dict[str, Any], db_type: str) -> str:
        tables = parsed.get("tables") or []
        joins = parsed.get("joins") or []
        subqueries = parsed.get("subqueries") or 0
        complexity = parsed.get("complexity_score") or 0
        query_type = parsed.get("query_type") or "SELECT"
        group_by = parsed.get("group_by") or []
        order_by = parsed.get("order_by") or []

        # Issue #74: show dialect display name (e.g. "PostgreSQL" not "postgresql")
        cfg = get_dialect(db_type)
        lines = [f"## Query Summary ({cfg.display})"]
        lines.append(f"- **Query type:** {query_type}")

        if tables:
            lines.append(f"- **Tables accessed:** {', '.join(tables)}")
        if joins:
            join_strs = [f"{j.get('type', 'JOIN')} {j.get('table', '?')}" for j in joins]
            lines.append(f"- **Joins:** {', '.join(join_strs)}")
        if group_by:
            lines.append(f"- **Grouped by:** {', '.join(str(c) for c in group_by)}")
        if order_by:
            lines.append(f"- **Ordered by:** {', '.join(str(c) for c in order_by)}")
        if subqueries > 0:
            lines.append(f"- **Nested subqueries:** {subqueries}")

        complexity_label = (
            "Low" if complexity < 30 else "Medium" if complexity < 60 else "High" if complexity < 80 else "Very High"
        )
        lines.append(f"- **Complexity score:** {complexity:.0f}/100 ({complexity_label})")

        return "\n".join(lines)

    def _format_findings_summary(self, suggestions: list[dict[str, Any]]) -> str:
        high = sum(1 for s in suggestions if s.get("severity") in ("high", "critical"))
        medium = sum(1 for s in suggestions if s.get("severity") == "medium")
        total = len(suggestions)
        return (
            f"## Performance Findings\n"
            f"**{total} issue(s) detected** — {high} high, {medium} medium severity. "
            f"See the Suggestions panel for full details and DDL hints."
        )

    def _format_security(self, issues: list[str]) -> str:
        lines = ["## Security Observations"]
        for issue in issues:
            lines.append(f"- {issue}")
        return "\n".join(lines)

    def _readability_tip(self, query: str, parsed: dict[str, Any]) -> str | None:
        tips: list[str] = []

        if query.count("\n") < 2:
            tips.append(
                "Format your query across multiple lines — " "improves readability and diff clarity in code review."
            )

        subqueries = parsed.get("subqueries") or 0
        if subqueries >= 2:
            tips.append(
                "This query has multiple nested subqueries. "
                "Consider extracting them into CTEs (`WITH` clauses) "
                "for clarity and potential planner benefits."
            )

        complexity = parsed.get("complexity_score") or 0
        if complexity >= 60:
            tips.append(
                "High complexity queries are harder to debug and maintain. "
                "Consider breaking this into intermediate steps or materialized views."
            )

        if not tips:
            return None

        lines = ["## Readability Tips"]
        for t in tips:
            lines.append(f"- {t}")
        return "\n".join(lines)

    def _format_schema_summary(
        self,
        schema_info: str | None,
        suggestions: list[dict[str, Any]],
        parsed: dict[str, Any],
    ) -> str | None:
        """
        Issue #8: summarise the parsed schema DDL — tables/columns/existing
        indexes — and how many suggestions were schema-verified against it
        rather than estimated from the query text alone.
        """
        try:
            schema = parse_schema_ddl(schema_info)
            indexed = get_indexed_columns(schema_info)
        except Exception:
            return None

        if not schema:
            return None

        lines = ["## Schema Context"]
        for table, columns in schema.items():
            col_items = list(columns.items())
            shown = col_items[:3]
            remaining = len(col_items) - len(shown)
            col_str = ", ".join(f"{col} {ctype}" for col, ctype in shown)
            if remaining > 0:
                col_str += f" + {remaining} more"
            lines.append(f"- **{table}** ({len(col_items)} columns) — {col_str}")

            table_indexed = indexed.get(table)
            if table_indexed:
                lines.append(f"  - Existing indexes: {', '.join(sorted(table_indexed))}")

        if suggestions:
            deterministic_count = sum(1 for s in suggestions if s.get("evidence_level") == "deterministic")
            schema_verified_count = sum(1 for s in suggestions if s.get("evidence_level") == "schema-verified")
            runtime_evidence_count = sum(1 for s in suggestions if s.get("evidence_level") == "needs-runtime-evidence")
            lines.append(
                f"- {deterministic_count} deterministic findings · "
                f"{schema_verified_count} schema-verified · "
                f"{runtime_evidence_count} needs runtime evidence"
            )

        return "\n".join(lines)

    def _format_maintenance(
        self,
        suggestions: list[dict[str, Any]],
        db_type: str,
        parsed: dict[str, Any],
    ) -> str | None:
        """
        Issue #75: Append dialect-specific maintenance commands when
        relevant findings exist (index recommendations, full scans, etc.).
        """
        index_types = {
            "index_review_join_key",
            "index_review_where_filter",
            "index_review_composite_index",
            "index_review_order_by_index",
            "index_review_group_by_index",
            "index_review_partial_index_candidate",
            "sequential_scan",
            "full_table_scan",
            "full_table_access",
        }

        has_index_finding = any(s.get("type", "") in index_types for s in suggestions)
        if not has_index_finding:
            return None

        cfg = get_dialect(db_type)
        tables = parsed.get("tables") or ["<table>"]
        table = tables[0] if tables else "<table>"

        lines = [f"## {cfg.display} Maintenance Commands"]
        lines.append("After adding indexes, update statistics so the planner uses them:")
        lines.append(f"```sql\n{cfg.update_stats.format(table=table)}\n```")
        lines.append("Verify with EXPLAIN:")
        explain_example = cfg.explain_cmd.replace("{query}", "-- your query here")
        lines.append(f"```sql\n{explain_example}\n```")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Issue #74: LLM context string — used by sql_analyzer when building
    # the LLM prompt. Call this to get the dialect-specific system context.
    # ------------------------------------------------------------------

    def get_llm_context(self, db_type: str) -> str:
        """
        Returns the dialect-specific context string to prepend to any
        LLM prompt for this analysis. Call from sql_analyzer.py when
        building the prompt passed to call_llm().

        Usage in sql_analyzer.py:
            explainer = QueryExplainer()
            dialect_ctx = explainer.get_llm_context(db_type)
            prompt = f"{dialect_ctx}\n\n{your_existing_prompt}"
        """
        return get_dialect(db_type).llm_context
