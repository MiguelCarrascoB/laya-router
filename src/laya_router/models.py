from typing import Any, Literal

from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    prompt: str = Field(min_length=1)


class RouteResponse(BaseModel):
    model: str
    confidence: float = Field(ge=0, le=1)
    escalate: bool
    laya_raw: dict[str, Any]


class TriageRequest(BaseModel):
    task: str = Field(min_length=1)
    context: str = ""


class TriageResponse(BaseModel):
    injection_risk: float = Field(ge=0, le=1)
    urgency: int = Field(ge=1, le=5)
    block: bool


class Classification(BaseModel):
    complexity: Literal["trivial", "medium", "hard"]
    confidence: float = Field(ge=0, le=1)
    raw: dict[str, Any]
