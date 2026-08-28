from analysis_engine.schemas import AnalysisFacts, QueryRequest
from analysis_engine.plan_parsers.mysql import parse_mysql_explain

from .base import BaseCollector


class MySQLCollector(BaseCollector):
    async def collect(self, request: QueryRequest) -> AnalysisFacts:
        # Issue #62: parses a pasted EXPLAIN FORMAT=JSON plan — same
        # paste-in-only scope as Postgres in #61, minus the live-DSN
        # fallback. No MySQL DSN wiring exists anywhere in this app, and
        # the design doc explicitly keeps that out of scope: closing the
        # UI's paste-in claim only needs this path, matching Postgres's
        # existing optional-DSN pattern rather than adding a new live
        # connection story for MySQL too.
        explain_plan = (request.explain_plan or "").strip()
        if explain_plan:
            parsed = parse_mysql_explain(explain_plan)
            facts = AnalysisFacts(db_type="mysql")
            if parsed:
                facts.plan = parsed.artifact
                facts.findings = parsed.findings
            else:
                facts.warnings.append("Pasted EXPLAIN plan could not be parsed — expected EXPLAIN FORMAT=JSON output.")
            return facts

        return self.not_configured("mysql")
