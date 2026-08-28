from abc import ABC, abstractmethod

from analysis_engine.schemas import AnalysisFacts, QueryRequest


class BaseCollector(ABC):
    """
    All DB collectors must implement collect().
    Returns normalized AnalysisFacts — never raw DB output.
    """

    @abstractmethod
    async def collect(self, request: QueryRequest) -> AnalysisFacts:
        pass

    def not_configured(self, db_type: str) -> AnalysisFacts:
        return AnalysisFacts(
            db_type=db_type,
            warnings=[
                f"{db_type} collector not configured — no DSN provided. "
                "Set the DSN env variable to enable live EXPLAIN plans."
            ],
        )
