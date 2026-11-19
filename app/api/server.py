from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.dependencies import get_research_service
from app.api.routes.health import router as health_router
from app.api.routes.research import router as research_router

UI_DIRECTORY = Path(__file__).resolve().parents[1] / "ui"


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = get_research_service()
    try:
        yield
    finally:
        get_research_service.cache_clear()
        service.close()


app = FastAPI(title="Research Review", lifespan=lifespan)
app.include_router(research_router, prefix="/api")
app.include_router(health_router, prefix="/api")
app.mount("/static", StaticFiles(directory=UI_DIRECTORY), name="static")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(UI_DIRECTORY / "index.html")