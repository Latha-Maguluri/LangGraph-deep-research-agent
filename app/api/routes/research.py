from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_research_service
from app.api.schemas import DecisionRequest, ResearchRequest, ResearchResponse
from app.services.research_service import ResearchService

router = APIRouter()


@router.post("/research", response_model=ResearchResponse, status_code=status.HTTP_202_ACCEPTED)
def start_research(request: ResearchRequest, service: ResearchService = Depends(get_research_service)) -> dict:
    return service.start_research(request.question)


@router.get("/research/{thread_id}", response_model=ResearchResponse)
def get_research(thread_id: str, service: ResearchService = Depends(get_research_service)) -> dict:
    try:
        return service.get_run(thread_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research run not found.") from exc


@router.post("/research/{thread_id}/decision", response_model=ResearchResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_decision(
    thread_id: str,
    request: DecisionRequest,
    service: ResearchService = Depends(get_research_service),
) -> dict:
    try:
        return service.decide(thread_id, request.action, request.notes)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research run not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
