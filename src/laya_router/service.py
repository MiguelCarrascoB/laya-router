import os

from fastapi import FastAPI

from .laya_backend import build_backend
from .models import RouteRequest, RouteResponse, TriageRequest, TriageResponse
from .router import ModelRouter

backend_kind = os.getenv("LAYA_BACKEND", "mock")
router = ModelRouter(build_backend(backend_kind))
app = FastAPI(title="Laya Model Router", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "laya_backend": router.backend.name}


@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest) -> dict:
    return router.route(request.prompt)


@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest) -> dict:
    return router.triage(request.task, request.context)
