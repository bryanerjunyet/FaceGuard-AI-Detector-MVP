from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.settings_loader import load_settings


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
