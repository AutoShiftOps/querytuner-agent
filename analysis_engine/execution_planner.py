"""
Trimmed from QueryTuner's backend/app/tools/execution_planner.py — routes
a pasted EXPLAIN plan to the right dialect-specific parser. The original
also routes sqlite/sqlserver/oracle to their own live-DB collectors;
those three are dropped here (this project's batch sources — Postgres
pg_stat_statements, MySQL performance_schema, SQL Server Query Store —
only ever produce postgresql or mysql db_type today per
batch_parsers.py's _BATCH_SOURCE_DB_TYPE-equivalent mapping, and there's
no pasted-plan support for the other three dialects to route to here
anyway) — any other db_type falls through to the same "no collector
available" result the original used as its own catch-all.
"""

from analysis_engine.plan_parsers.models import PlanNode
from analysis_engine.schemas import AnalysisFacts, DatabaseType, QueryRequest


async def collect_facts(request: QueryRequest) -> tuple[AnalysisFacts, list[PlanNode]]:
    db_type = request.db_type

    if db_type == DatabaseType.POSTGRES:
        from analysis_engine.collectors.postgres import PostgresCollector
        from analysis_engine.plan_parsers.postgres import nodes_from_artifact

        facts = await PostgresCollector().collect(request)
        nodes = nodes_from_artifact(facts.plan) if facts.plan else []
        return facts, nodes

    elif db_type == DatabaseType.MYSQL:
        from analysis_engine.collectors.mysql import MySQLCollector
        from analysis_engine.plan_parsers.mysql import nodes_from_artifact as mysql_nodes_from_artifact

        facts = await MySQLCollector().collect(request)
        nodes = mysql_nodes_from_artifact(facts.plan) if facts.plan else []
        return facts, nodes

    else:
        return AnalysisFacts(db_type=db_type.value, warnings=["No collector available for this DB type"]), []
