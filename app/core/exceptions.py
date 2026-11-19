class ResearchError(RuntimeError):
    """Base exception for workflow failures."""


class PlanRejectedError(ResearchError):
    """Raised when a human rejects the research plan."""


class NoEvidenceFoundError(ResearchError):
    """Raised when no valid evidence is found for a research task."""


class CriticFailureError(ResearchError):
    """Raised when the report fails the quality gate."""


class ReportWriterTokenLimitError(ResearchError):
    """Raised when no complete or previously reviewed report can be returned."""
