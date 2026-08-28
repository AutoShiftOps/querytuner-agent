"""
Trimmed from QueryTuner's backend/app/tools/collectors/postgres.py —
pasted-EXPLAIN-plan path only. The original also has an optional
live-DSN fallback (asyncpg) for a real Postgres connection; deliberately
dropped here, matching this project's "paste a slow-query export, no
live database connection required" scope (and QueryTuner's own
production deployment, where POSTGRES_DSN has never actually been set —
the pasted-plan path is the one that's ever really been exercised).
"""

from analysis_engine.plan_parsers.postgres import parse_postgres_explain
from analysis_engine.schemas import AnalysisFacts, QueryRequest

from .base import BaseCollector


class PostgresCollector(BaseCollector):
    async def collect(self, request: QueryRequest) -> AnalysisFacts:
        explain_plan = (request.explain_plan or "").strip()
        if explain_plan:
            parsed = parse_postgres_explain(explain_plan)
            facts = AnalysisFacts(db_type="postgresql")
            if parsed:
                facts.plan = parsed.artifact
                facts.findings = parsed.findings
            else:
                facts.warnings.append(
                    "Pasted EXPLAIN plan could not be parsed — expected JSON "
                    "(EXPLAIN (FORMAT JSON) ...) or plain-text tabular output "
                    "(EXPLAIN (ANALYZE, BUFFERS) ...)."
                )
            return facts

        return self.not_configured("postgresql")
