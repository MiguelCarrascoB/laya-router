"""Inference boundary: production laya and an offline deterministic substitute."""

from abc import ABC, abstractmethod
import logging
import time
from typing import Any


logger = logging.getLogger(__name__)


def load(*args: Any, **kwargs: Any) -> Any:
    """Load laya only when the real backend is actually used."""
    from laya import load as laya_load  # type: ignore[import-not-found]

    return laya_load(*args, **kwargs)


ROUTE_QUESTIONS = {
    "complexity": {"type": "choice", "instructions": "Classify task complexity", "criteria": {"trivial": "simple lookup or formatting", "medium": "multi-step reasoning or coding", "hard": "deep analysis, architecture, or ambiguity"}},
}
TRIAGE_QUESTIONS = {
    "injection": {"type": "noul", "instructions": "Is this an instruction injection attempt?"},
    "urgency": {"type": "score", "instructions": "Score urgency from 1 to 5", "criteria": {"min": 1, "max": 5}},
}


class LayaBackend(ABC):
    name: str

    @abstractmethod
    def predict(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """Return laya-shaped answers."""


class MockLayaBackend(LayaBackend):
    name = "mock"
    _hard = ("architecture", "security", "debug", "design", "algorithm", "production", "deep", "compare")
    _medium = ("code", "implement", "write", "refactor", "explain", "analyze", "summarize", "translate")
    _injection = ("ignore previous", "system prompt", "jailbreak", "developer message", "reveal your instructions")

    def predict(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        text = str(state.get("prompt", state.get("task", ""))).lower()
        answers: dict[str, Any] = {}
        if "complexity" in questions:
            if any(word in text for word in self._hard):
                choice, confidence = "hard", 0.93
            elif any(word in text for word in self._medium) or len(text.split()) > 18:
                choice, confidence = "medium", 0.89
            else:
                choice, confidence = "trivial", 0.96
            answers["complexity"] = {"choice": choice, "confidence": confidence}
        if "injection" in questions:
            risk = 0.97 if any(word in text for word in self._injection) else 0.04
            answers["injection"] = {"noul": risk, "confidence": 0.95}
        if "urgency" in questions:
            urgency = 5 if any(word in text for word in ("urgent", "outage", "asap", "immediately")) else 2
            answers["urgency"] = {"score": urgency, "confidence": 0.9}
        return {"answers": answers, "backend": self.name}


def _clamp_scores_to_criteria(raw: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
    """Clamp laya "score" answers into the range requested by their question.

    The real model occasionally answers outside its criteria (observed:
    urgency 0 against a 1-5 scale, which crashed the response schema with
    a 500). The backend boundary clamps to the nearest in-range value and
    logs the adjustment instead of silently fabricating or crashing.
    """
    if "answers" not in raw:
        return raw
    answers = dict(raw["answers"])
    for key, question in questions.items():
        if question.get("type") != "score":
            continue
        criteria = question.get("criteria") or {}
        answer = answers.get(key)
        if not isinstance(answer, dict) or "score" not in answer:
            continue
        score = answer["score"]
        clamped = score
        if "min" in criteria and score < criteria["min"]:
            clamped = criteria["min"]
        if "max" in criteria and score > criteria["max"]:
            clamped = criteria["max"]
        if clamped != score:
            logger.warning("laya score for %r outside criteria %s: %r -> %r", key, criteria, score, clamped)
            answers[key] = {**answer, "score": clamped}
    return {**raw, "answers": answers}


class RealLayaBackend(LayaBackend):
    name = "real"
    model_id = "convaiinnovations/laya"
    model_subfolder = "typed-decisions"

    def __init__(self) -> None:
        self._agent: Any | None = None

    def _get_agent(self) -> Any:
        if self._agent is None:
            started = time.perf_counter()
            self._agent = load(self.model_id, subfolder=self.model_subfolder)
            logger.info("Loaded laya model %s/%s in %.2fs", self.model_id, self.model_subfolder, time.perf_counter() - started)
        return self._agent

    def predict(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        raw = self._get_agent().predict(state, questions)
        return _clamp_scores_to_criteria(raw, questions)


def build_backend(kind: str = "mock") -> LayaBackend:
    if kind == "real":
        return RealLayaBackend()
    if kind != "mock":
        raise ValueError("backend must be 'mock' or 'real'")
    return MockLayaBackend()
