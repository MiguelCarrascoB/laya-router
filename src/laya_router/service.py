import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .laya_backend import build_backend
from .models import RouteRequest, RouteResponse, TriageRequest, TriageResponse
from .router import ModelRouter

backend_kind = os.getenv("LAYA_BACKEND", "mock")
router = ModelRouter(build_backend(backend_kind))
app = FastAPI(title="Laya Model Router", version="0.1.0")

# Documented convention: any failure raised by the laya inference boundary
# (crash, timeout, malformed laya payload) surfaces as 503 Service Unavailable
# with a JSON {"detail": ...} body, never a 500 stack trace.
BACKUP_UNAVAILABLE_STATUS = 503


def _backend_failure(exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=BACKUP_UNAVAILABLE_STATUS,
        content={"detail": f"laya backend unavailable: {type(exc).__name__}: {exc}"},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "laya_backend": router.backend.name}


@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest) -> JSONResponse:
    try:
        return router.route(request.prompt)
    except Exception as exc:  # noqa: BLE001 - deliberate inference-boundary guard
        return _backend_failure(exc)


@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest) -> JSONResponse:
    try:
        return router.triage(request.task, request.context)
    except Exception as exc:  # noqa: BLE001 - deliberate inference-boundary guard
        return _backend_failure(exc)
