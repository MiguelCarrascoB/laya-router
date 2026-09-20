import os
from typing import Any

from .laya_backend import ROUTE_QUESTIONS, TRIAGE_QUESTIONS, LayaBackend, build_backend
from .models import Classification

ROUTING_TABLE = {"trivial": "glm-5.3-flash", "medium": "deepseek-v4.1-flash", "hard": "deepseek-v4-pro"}
CONFIDENCE_GATE = 0.85


def confidence_gate() -> float:
    return float(os.getenv("ROUTE_CONFIDENCE_GATE", str(CONFIDENCE_GATE)))


class ModelRouter:
    def __init__(self, backend: LayaBackend | None = None, confidence_threshold: float | None = None) -> None:
        self.backend = backend or build_backend()
        self.confidence_threshold = confidence_gate() if confidence_threshold is None else confidence_threshold

    def classify(self, prompt: str) -> Classification:
        raw = self.backend.predict({"prompt": prompt}, ROUTE_QUESTIONS)
        answer = raw["answers"]["complexity"]
        return Classification(complexity=answer["choice"], confidence=answer["confidence"], raw=raw)

    def route(self, prompt: str) -> dict[str, Any]:
        result = self.classify(prompt)
        escalate = result.confidence < self.confidence_threshold
        return {"model": "deepseek-v4-pro" if escalate else ROUTING_TABLE[result.complexity], "confidence": result.confidence, "escalate": escalate, "laya_raw": result.raw}

    def triage(self, task: str, context: str = "") -> dict[str, Any]:
        raw = self.backend.predict({"task": f"{task}\n{context}"}, TRIAGE_QUESTIONS)
        injection = raw["answers"]["injection"]["noul"]
        return {"injection_risk": injection, "urgency": int(raw["answers"]["urgency"]["score"]), "block": injection >= 0.9}
