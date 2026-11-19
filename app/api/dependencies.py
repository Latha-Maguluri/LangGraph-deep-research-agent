from __future__ import annotations

from functools import lru_cache

from app.services.research_service import ResearchService


@lru_cache(maxsize=1)
def get_research_service() -> ResearchService:
    return ResearchService()
