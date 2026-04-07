from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.settings_loader import load_settings
from .schemas import HealthResponse


settings = load_settings()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="FaceGuard MVP backend for local prototype demonstration.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        model_ready=False,
        model_name=settings.model_name,
    )
