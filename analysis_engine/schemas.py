"""
Trimmed schema subset vendored from QueryTuner (AutoShiftOps/querytuner,
backend/app/schemas/models.py) — only the pieces the analysis engine
itself needs. LLMProvider and every AI-provider-related field are
deliberately dropped: this engine has zero AI-provider awareness of its
own, by design (see sql_analyzer.py's module docstring) — the calling
agent (gemini_agent.py) is the only place in this project that ever
talks to an AI model, and that's Gemini/Gemma exclusively.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DatabaseType(StrEnum):
    POSTGRES = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    SQL_SERVER = "sqlserver"
    ORACLE = "oracle"


class QueryRequest(BaseModel):
    query: str = Field(..., description="SQL query to analyze")
    db_type: DatabaseType = Field(default=DatabaseType.POSTGRES)
    schema_info: str | None = Field(None, description="Schema DDL for better context")
    explain_plan: str | None = Field(
        None, description="Raw EXPLAIN plan output pasted by user (dialect-specific format)"
    )
    focus: str = Field(default="performance")


class Finding(BaseModel):
    type: str
    severity: str
    title: str
    evidence: str | None = None
    recommendation: str | None = None


class PlanArtifact(BaseModel):
    format: str  # "json" | "xml" | "text"
    raw: Any


class AnalysisFacts(BaseModel):
    db_type: str
    normalized_query: str | None = None
    redacted_query: str | None = None
    findings: list[Finding] = Field(default_factory=list)
    plan: PlanArtifact | None = None
    warnings: list[str] = Field(default_factory=list)
