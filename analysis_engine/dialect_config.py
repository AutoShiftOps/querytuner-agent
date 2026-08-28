"""
dialect_config.py — Single source of truth for dialect-specific output
Place at: backend/app/utils/dialect_config.py

Issue #72: Dialect-aware index DDL
Issue #73: Dialect-aware optimizer syntax
Issue #74: Dialect-aware LLM prompt context
Issue #75: Dialect-aware maintenance commands

Import this in:
  - backend/app/tools/index_recommender.py  (or wherever DDL is built)
  - backend/app/agents/optimizer.py
  - backend/app/agents/explainer.py
  - backend/app/llm/router.py

Usage:
  from analysis_engine.dialect_config import get_dialect, DIALECTS

  cfg = get_dialect("postgresql")
  ddl = cfg.index_ddl("orders", "customer_id")
  ctx = cfg.llm_context
"""

from dataclasses import dataclass, field

# from typing import Callable


# ── Dialect dataclass ─────────────────────────────────────────────────────────


@dataclass
class DialectConfig:
    name: str
    display: str

    # ── Index DDL ──────────────────────────────────────────────────────────
    # Template for generating CREATE INDEX DDL from table + column name
    _index_ddl_template: str = ""
    # Statement to undo a suggested index — {index}/{table} placeholders
    rollback_index: str = ""

    # ── Optimizer syntax ───────────────────────────────────────────────────
    # Correct syntax for common rewrites in this dialect
    pagination: str = ""  # LIMIT/OFFSET or FETCH FIRST
    like_ci: str = ""  # Case-insensitive LIKE
    date_range: str = ""  # Date range pattern (instead of YEAR() in WHERE)
    exists_hint: str = ""  # Preferred EXISTS pattern note
    optimizer_hint: str = ""  # How to specify index hints

    # ── Maintenance commands ───────────────────────────────────────────────
    update_stats: str = ""  # Command to update table statistics
    rebuild_index: str = ""  # Command to rebuild a specific index
    check_bloat: str = ""  # Command to check table/index bloat
    explain_cmd: str = ""  # How to run EXPLAIN in this dialect

    # ── LLM system prompt context ──────────────────────────────────────────
    llm_context: str = ""

    # ── Unsupported features ───────────────────────────────────────────────
    # List of things that do NOT work in this dialect
    unsupported: list = field(default_factory=list)

    def index_ddl(self, table: str, column: str, idx_name: str = "") -> str:
        """Return dialect-correct CREATE INDEX DDL."""
        name = idx_name or f"idx_{table}_{column}"
        return self._index_ddl_template.format(name=name, table=table, column=column)

    def index_ddl_note(self) -> str:
        """Return a one-line production note about index creation for this dialect."""
        return DIALECT_INDEX_NOTES.get(self.name, "")


# ── Index creation notes ──────────────────────────────────────────────────────

DIALECT_INDEX_NOTES = {
    "postgresql": (
        "Use CONCURRENTLY to avoid locking the table in production. "
        "Validate with: EXPLAIN (ANALYZE, BUFFERS) after creation."
    ),
    "mysql": (
        "ALTER TABLE locks the table in MySQL <5.6. "
        "Use pt-online-schema-change for zero-downtime index creation on large tables."
    ),
    "oracle": (
        "NOLOGGING speeds up creation but skips redo log — only use on non-production hours. "
        "Run DBMS_STATS after creation."
    ),
    "sqlserver": (
        "WITH (ONLINE=ON) allows concurrent reads/writes during index creation (Enterprise edition). "
        "Monitor with sys.dm_exec_requests during build."
    ),
    "sqlite": ("SQLite has no CONCURRENTLY — index creation locks the database. " "Schedule during off-peak windows."),
}


# ── Dialect definitions ───────────────────────────────────────────────────────

DIALECTS: dict[str, DialectConfig] = {
    "postgresql": DialectConfig(
        name="postgresql",
        display="PostgreSQL",
        _index_ddl_template=("CREATE INDEX CONCURRENTLY {name} ON {table}({column});"),
        rollback_index="DROP INDEX CONCURRENTLY {index};",
        pagination="LIMIT {n} OFFSET {m}",
        like_ci="col ILIKE '%value%'  -- case-insensitive, uses pg_trgm index",
        date_range=(
            "WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01'"
            "  -- replaces YEAR(created_at) = 2024; allows index scan"
        ),
        exists_hint="Use EXISTS instead of IN for correlated subqueries",
        optimizer_hint=(
            "-- PostgreSQL uses the planner; hints via pg_hint_plan extension only\n"
            "-- SET enable_seqscan = off; -- for debugging only, not production"
        ),
        update_stats="ANALYZE {table};",
        rebuild_index="REINDEX INDEX CONCURRENTLY {index};",
        check_bloat=("SELECT * FROM pgstattuple('{table}');  -- requires pgstattuple extension"),
        explain_cmd="EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query};",
        llm_context=(
            "You are analyzing a PostgreSQL query. "
            "Reference pg_stat_user_tables, pg_stat_user_indexes, VACUUM ANALYZE, "
            "EXPLAIN (ANALYZE, BUFFERS), partial indexes, CONCURRENTLY index creation, "
            "CTEs with MATERIALIZED hint (PG14+), and the pg_trgm extension for LIKE optimisation. "
            "Never suggest MySQL or Oracle syntax. "
            "Index DDL must use CREATE INDEX CONCURRENTLY. "
            "For queries filtering on a PRIMARY KEY or UNIQUE constraint column, do NOT "
            "recommend additional indexes — these queries already use the optimal access "
            "path. Only recommend indexes when there is clear evidence of a missing index "
            "on a non-key column used in a WHERE, JOIN, or ORDER BY clause that is not "
            "already covered by an existing index in the provided schema."
        ),
        unsupported=[
            "FORCE INDEX hints (use pg_hint_plan extension instead)",
            "ROWNUM (use LIMIT/OFFSET or ROW_NUMBER())",
            "DBMS_STATS (use ANALYZE)",
        ],
    ),
    "mysql": DialectConfig(
        name="mysql",
        display="MySQL",
        _index_ddl_template=("ALTER TABLE {table} ADD INDEX {name} ({column});"),
        rollback_index="ALTER TABLE {table} DROP INDEX {index};",
        pagination="LIMIT {n} OFFSET {m}",
        like_ci=(
            "col LIKE '%value%'  -- case-insensitive by default on ci collations\n"
            "-- For explicit: LOWER(col) LIKE LOWER('%value%')"
        ),
        date_range=(
            "WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01'"
            "  -- replaces YEAR(created_at) = 2024; allows index on created_at"
        ),
        exists_hint="Prefer EXISTS over IN — MySQL optimiser handles EXISTS better for large subqueries",
        optimizer_hint=(
            "SELECT /*+ USE_INDEX(t idx_name) */ ...  -- inline hint\n"
            "-- or: SELECT * FROM t FORCE INDEX (idx_name) WHERE ..."
        ),
        update_stats="ANALYZE TABLE {table};",
        rebuild_index="ALTER TABLE {table} ENGINE=InnoDB;  -- forces full index rebuild",
        check_bloat="SHOW TABLE STATUS LIKE '{table}';  -- check Data_free column",
        explain_cmd="EXPLAIN FORMAT=JSON {query};",
        llm_context=(
            "You are analyzing a MySQL/InnoDB query. "
            "Reference EXPLAIN FORMAT=JSON, ALTER TABLE ADD INDEX, ANALYZE TABLE, "
            "USE INDEX and FORCE INDEX hints, storage engine considerations (InnoDB vs MyISAM), "
            "the slow query log, pt-query-digest for query analysis, and "
            "pt-online-schema-change for production index creation. "
            "Never suggest CONCURRENTLY (PostgreSQL only) or DBMS_STATS (Oracle only). "
            "Index DDL must use ALTER TABLE ... ADD INDEX. "
            "For queries filtering on a PRIMARY KEY or UNIQUE constraint column, do NOT "
            "recommend additional indexes — these queries already use the optimal access "
            "path. Only recommend indexes when there is clear evidence of a missing index "
            "on a non-key column used in a WHERE, JOIN, or ORDER BY clause that is not "
            "already covered by an existing index in the provided schema."
        ),
        unsupported=[
            "CREATE INDEX CONCURRENTLY (PostgreSQL only)",
            "FETCH FIRST n ROWS ONLY (use LIMIT)",
            "ILIKE (use LIKE with ci collation)",
            "DBMS_STATS (use ANALYZE TABLE)",
        ],
    ),
    "oracle": DialectConfig(
        name="oracle",
        display="Oracle",
        _index_ddl_template=("CREATE INDEX {name} ON {table}({column}) NOLOGGING;"),
        rollback_index="DROP INDEX {index};",
        pagination=("FETCH FIRST {n} ROWS ONLY  -- Oracle 12c+\n" "-- Oracle 11g: WHERE ROWNUM <= {n}"),
        like_ci=(
            "WHERE UPPER(col) LIKE UPPER('%value%')\n"
            "-- Or use function-based index: CREATE INDEX idx ON t(UPPER(col))"
        ),
        date_range=(
            "WHERE created_at >= DATE '2024-01-01' AND created_at < DATE '2025-01-01'"
            "  -- replaces TRUNC(created_at, 'YYYY') = ...; allows index range scan"
        ),
        exists_hint="Use EXISTS — Oracle optimiser handles it well; avoid NOT IN with NULLs",
        optimizer_hint=(
            "SELECT /*+ INDEX(t idx_name) */ ...  -- force index\n"
            "SELECT /*+ FULL(t) */ ...            -- force full scan\n"
            "SELECT /*+ NO_MERGE(subq) */ ...     -- prevent view merging"
        ),
        update_stats=(
            "EXEC DBMS_STATS.GATHER_TABLE_STATS(" "ownname => 'SCHEMA', tabname => '{table}', cascade => TRUE);"
        ),
        rebuild_index="ALTER INDEX {index} REBUILD NOLOGGING;",
        check_bloat=("SELECT blocks, empty_blocks FROM dba_tables WHERE table_name = UPPER('{table}');"),
        explain_cmd=("EXPLAIN PLAN FOR {query};\n" "SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY);"),
        llm_context=(
            "You are analyzing an Oracle SQL query. "
            "Reference EXPLAIN PLAN, DBMS_XPLAN.DISPLAY, DBMS_STATS.GATHER_TABLE_STATS, "
            "optimizer hints (/*+ INDEX */ /*+ FULL */ /*+ NO_MERGE */), "
            "bind variable peeking, partition pruning, ROWNUM vs FETCH FIRST, "
            "function-based indexes, and NOLOGGING for bulk operations. "
            "Never suggest CONCURRENTLY (PostgreSQL only), LIMIT (use FETCH FIRST or ROWNUM), "
            "or ALTER TABLE ADD INDEX (Oracle uses CREATE INDEX). "
            "Index DDL must use: CREATE INDEX name ON table(col) NOLOGGING; "
            "For queries filtering on a PRIMARY KEY or UNIQUE constraint column, do NOT "
            "recommend additional indexes — these queries already use the optimal access "
            "path. Only recommend indexes when there is clear evidence of a missing index "
            "on a non-key column used in a WHERE, JOIN, or ORDER BY clause that is not "
            "already covered by an existing index in the provided schema."
        ),
        unsupported=[
            "CREATE INDEX CONCURRENTLY (PostgreSQL only)",
            "LIMIT (use FETCH FIRST n ROWS ONLY or ROWNUM)",
            "ILIKE (use UPPER() or NLS settings)",
            "ALTER TABLE ADD INDEX (use CREATE INDEX)",
        ],
    ),
    "sqlserver": DialectConfig(
        name="sqlserver",
        display="SQL Server",
        _index_ddl_template=(
            "CREATE NONCLUSTERED INDEX {name} ON {table}({column}) " "WITH (ONLINE=ON, FILLFACTOR=90);"
        ),
        rollback_index="DROP INDEX {index} ON {table};",
        pagination=("ORDER BY col\n" "OFFSET {m} ROWS FETCH NEXT {n} ROWS ONLY  -- SQL Server 2012+"),
        like_ci=(
            "col LIKE '%value%'  -- case sensitivity depends on collation\n"
            "-- For explicit ci: col COLLATE SQL_Latin1_General_CP1_CI_AS LIKE '%value%'"
        ),
        date_range=(
            "WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01'"
            "  -- replaces YEAR(created_at) = 2024; allows index seek"
        ),
        exists_hint="Use EXISTS — SQL Server handles it efficiently with proper indexing",
        optimizer_hint=(
            "SELECT * FROM t WITH (INDEX(idx_name)) WHERE ...  -- table hint\n"
            "-- or: OPTION (USE HINT('FORCE_LEGACY_CARDINALITY_ESTIMATION'))"
        ),
        update_stats="UPDATE STATISTICS {table};",
        rebuild_index=("ALTER INDEX {index} ON {table} REBUILD WITH (ONLINE=ON);"),
        check_bloat=(
            "SELECT * FROM sys.dm_db_index_physical_stats" "(DB_ID(), OBJECT_ID('{table}'), NULL, NULL, 'DETAILED');"
        ),
        explain_cmd=("SET STATISTICS IO, TIME ON;\n" "{query};\n" "-- Or: use Actual Execution Plan in SSMS"),
        llm_context=(
            "You are analyzing a SQL Server T-SQL query. "
            "Reference SET STATISTICS IO ON, sys.dm_exec_query_stats, "
            "sys.dm_exec_requests, CREATE NONCLUSTERED INDEX WITH (ONLINE=ON), "
            "UPDATE STATISTICS, Query Store (sys.query_store_plan), "
            "clustered vs non-clustered index strategy, WITH (NOLOCK) risks, "
            "and OPTION query hints. "
            "Never suggest CONCURRENTLY (PostgreSQL only), LIMIT (use OFFSET/FETCH), "
            "or DBMS_STATS (Oracle only). "
            "Index DDL must use: CREATE NONCLUSTERED INDEX name ON table(col) WITH (ONLINE=ON); "
            "For queries filtering on a PRIMARY KEY or UNIQUE constraint column, do NOT "
            "recommend additional indexes — these queries already use the optimal access "
            "path. Only recommend indexes when there is clear evidence of a missing index "
            "on a non-key column used in a WHERE, JOIN, or ORDER BY clause that is not "
            "already covered by an existing index in the provided schema."
        ),
        unsupported=[
            "CREATE INDEX CONCURRENTLY (PostgreSQL only)",
            "LIMIT (use OFFSET n ROWS FETCH NEXT n ROWS ONLY)",
            "ILIKE (use LIKE with CI collation)",
            "DBMS_STATS (use UPDATE STATISTICS)",
        ],
    ),
    "sqlite": DialectConfig(
        name="sqlite",
        display="SQLite",
        _index_ddl_template=("CREATE INDEX IF NOT EXISTS {name} ON {table}({column});"),
        rollback_index="DROP INDEX IF EXISTS {index};",
        pagination="LIMIT {n} OFFSET {m}",
        like_ci=(
            "col LIKE '%value%'  -- case-insensitive for ASCII by default\n"
            "-- For unicode: use PRAGMA case_sensitive_like = ON; carefully"
        ),
        date_range=(
            "WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01'"
            "  -- SQLite stores dates as TEXT/REAL/INTEGER; use ISO 8601 strings"
        ),
        exists_hint="EXISTS is supported; prefer over IN for subqueries",
        optimizer_hint=(
            "-- SQLite has no index hints\n" "-- Use INDEXED BY clause: SELECT * FROM t INDEXED BY idx_name WHERE ..."
        ),
        update_stats="ANALYZE;  -- updates sqlite_stat1 table",
        rebuild_index="DROP INDEX IF EXISTS {index}; -- then recreate",
        check_bloat="PRAGMA integrity_check; PRAGMA freelist_count;",
        explain_cmd="EXPLAIN QUERY PLAN {query};",
        llm_context=(
            "You are analyzing a SQLite query. "
            "Reference EXPLAIN QUERY PLAN, PRAGMA statements, "
            "the INDEXED BY clause for index hints, WAL mode (PRAGMA journal_mode=WAL), "
            "SQLite's dynamic typing system, and the absence of stored procedures, "
            "concurrent DDL, and complex data types. "
            "SQLite has no CONCURRENTLY — index creation locks the database file. "
            "Never suggest ALTER TABLE ADD INDEX (use CREATE INDEX), "
            "or stored procedures (not supported). "
            "Index DDL must use: CREATE INDEX IF NOT EXISTS name ON table(col); "
            "For queries filtering on a PRIMARY KEY or UNIQUE constraint column, do NOT "
            "recommend additional indexes — these queries already use the optimal access "
            "path. Only recommend indexes when there is clear evidence of a missing index "
            "on a non-key column used in a WHERE, JOIN, or ORDER BY clause that is not "
            "already covered by an existing index in the provided schema."
        ),
        unsupported=[
            "CREATE INDEX CONCURRENTLY (not supported — locks DB file)",
            "ALTER TABLE ADD INDEX (use CREATE INDEX)",
            "Stored procedures",
            "FETCH FIRST (use LIMIT/OFFSET)",
            "DBMS_STATS (use ANALYZE)",
        ],
    ),
}


# ── Public API ────────────────────────────────────────────────────────────────


def get_dialect(db_type: str) -> DialectConfig:
    """
    Return the DialectConfig for a given db_type string.
    Falls back to PostgreSQL if unrecognised (safe default).
    """
    return DIALECTS.get(db_type.lower().strip(), DIALECTS["postgresql"])


def get_llm_context(db_type: str) -> str:
    """Return the LLM system prompt context for a dialect."""
    return get_dialect(db_type).llm_context


def get_index_ddl(db_type: str, table: str, column: str, idx_name: str = "") -> str:
    """Return dialect-correct CREATE INDEX DDL."""
    return get_dialect(db_type).index_ddl(table, column, idx_name)


def get_unsupported_warnings(db_type: str, detected_patterns: list[str]) -> list[str]:
    """
    Cross-reference detected patterns against known unsupported features
    for this dialect. Returns a list of warning strings.

    Example:
        detected_patterns = ["CONCURRENTLY", "ILIKE"]
        get_unsupported_warnings("mysql", detected_patterns)
        → ["'CONCURRENTLY' is PostgreSQL-only — not valid in MySQL",
           "'ILIKE' is PostgreSQL-only — use LIKE with a CI collation in MySQL"]
    """
    cfg = get_dialect(db_type)
    warnings = []
    for pattern in detected_patterns:
        for unsupported in cfg.unsupported:
            if pattern.upper() in unsupported.upper():
                warnings.append(f"'{pattern}' is not supported in {cfg.display}: {unsupported}")
    return warnings
