"""Guardrail boundary: injection/urgency triage for incoming tasks.

Extracted from the router as its own module so the guardrail can later be
split into a separate package without touching routing: it depends only on
the narrow LayaBackend.predict interface and the triage questions. Until
then, the ModelRouter delegates here and both features share one backend.
"""

from typing import Any

from .laya_backend import TRIAGE_QUESTIONS, LayaBackend

INJECTION_BLOCK_THRESHOLD = 0.9


class Guardrail:
    def __init__(self, backend: LayaBackend) -> None:
        self.backend = backend

    def triage(self, task: str, context: str = "") -> dict[str, Any]:
        raw = self.backend.predict({"task": f"{task}\n{context}"}, TRIAGE_QUESTIONS)
        injection = raw["answers"]["injection"]["noul"]
        return {"injection_risk": injection, "urgency": int(raw["answers"]["urgency"]["score"]), "block": injection >= INJECTION_BLOCK_THRESHOLD}
