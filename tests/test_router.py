import time

import pytest
from fastapi.testclient import TestClient

import laya_router.laya_backend as laya_backend
from laya_router.laya_backend import LayaBackend
from laya_router.router import CONFIDENCE_GATE, ModelRouter
from laya_router.service import app


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class StaticBackend(LayaBackend):
    """Returns one canned complexity decision for every prompt."""

    name = "mock"

    def __init__(self, choice="trivial", confidence=0.96):
        self.choice = choice
        self.confidence = confidence

    def predict(self, state, questions):
        return {
            "answers": {"complexity": {"choice": self.choice, "confidence": self.confidence}},
            "backend": self.name,
        }


class LowConfidenceBackend(StaticBackend):
    def __init__(self):
        super().__init__(choice="trivial", confidence=0.5)


# ---------------------------------------------------------------------------
# Routing table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "expected_model"),
    [
        ("What is 2 + 2?", "glm-5.3-flash"),
        ("Explain how to write code for an API", "deepseek-v4.1-flash"),
        ("Design a secure production architecture", "deepseek-v4-pro"),
    ],
)
def test_routing_table_maps_complexity_to_model(prompt, expected_model):
    router = ModelRouter()
    assert router.route(prompt)["model"] == expected_model


@pytest.mark.parametrize(
    ("choice", "expected_model"),
    [
        ("trivial", "glm-5.3-flash"),
        ("medium", "deepseek-v4.1-flash"),
        ("hard", "deepseek-v4-pro"),
    ],
)
def test_every_routing_table_entry_is_reachable(choice, expected_model):
    router = ModelRouter(StaticBackend(choice=choice, confidence=0.99))
    assert router.route("anything")["model"] == expected_model


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.0, 0.3, 0.7, 0.849, 0.85, 0.851, 0.99])
def test_confidence_gate_boundary(confidence):
    router = ModelRouter(StaticBackend(choice="trivial", confidence=confidence))
    result = router.route("anything")
    assert result["escalate"] is (confidence < CONFIDENCE_GATE)
    assert result["confidence"] == confidence
    if confidence < CONFIDENCE_GATE:
        assert result["model"] == "deepseek-v4-pro"
    else:
        assert result["model"] == "glm-5.3-flash"


def test_confidence_gate_escalates():
    result = ModelRouter(LowConfidenceBackend()).route("anything")
    assert result["model"] == "deepseek-v4-pro"
    assert result["escalate"] is True


def test_confidence_gate_from_environment(monkeypatch):
    monkeypatch.setenv("ROUTE_CONFIDENCE_GATE", "0.95")
    router = ModelRouter(LowConfidenceBackend())
    assert router.confidence_threshold == 0.95


@pytest.mark.parametrize("gate", ["0.5", "0.7", "0.99"])
def test_confidence_gate_env_override_changes_escalation(monkeypatch, gate):
    monkeypatch.setenv("ROUTE_CONFIDENCE_GATE", gate)
    router = ModelRouter(StaticBackend(choice="trivial", confidence=0.8))
    assert router.confidence_threshold == float(gate)
    assert router.route("anything")["escalate"] is (0.8 < float(gate))


def test_confidence_gate_env_beats_explicit_none(monkeypatch):
    monkeypatch.setenv("ROUTE_CONFIDENCE_GATE", "0.3")
    assert ModelRouter().confidence_threshold == 0.3
    assert ModelRouter(confidence_threshold=0.6).confidence_threshold == 0.6


# ---------------------------------------------------------------------------
# Unknown complexity, empty and long prompts
# ---------------------------------------------------------------------------


def test_unknown_complexity_label_is_rejected_by_schema():
    router = ModelRouter(StaticBackend(choice="mystery", confidence=0.99))
    with pytest.raises(Exception) as excinfo:
        router.route("anything")
    assert "mystery" in str(excinfo.value)


def test_case_variant_complexity_label_is_rejected_by_schema():
    router = ModelRouter(StaticBackend(choice="TRIVIAL", confidence=0.99))
    with pytest.raises(Exception) as excinfo:
        router.route("anything")
    assert "TRIVIAL" in str(excinfo.value)


def test_empty_prompt_is_still_classified_by_mock():
    router = ModelRouter()
    result = router.route("")
    assert result["model"] in {"glm-5.3-flash", "deepseek-v4.1-flash", "deepseek-v4-pro"}


def test_very_long_prompt_is_medium_or_hard_not_trivial():
    router = ModelRouter()
    result = router.route("word " * 500)
    assert result["model"] != "glm-5.3-flash"


# ---------------------------------------------------------------------------
# Real backend path with a fake laya agent
# ---------------------------------------------------------------------------


class RecordingAgent:
    def __init__(self, confidence=0.91):
        self.calls = []
        self.confidence = confidence

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {
            "answers": {
                "complexity": {"choice": "medium", "confidence": self.confidence},
                "injection": {"noul": 0.02, "confidence": 0.88},
                "urgency": {"score": 3, "confidence": 0.77},
            }
        }


def _install_fake_agent(monkeypatch, confidence=0.91):
    agent = RecordingAgent(confidence=confidence)
    loads = []

    def fake_load(*args, **kwargs):
        loads.append((args, kwargs))
        return agent

    monkeypatch.setattr(laya_backend, "load", fake_load)
    return agent, loads


def test_real_backend_laya_raw_fields_pass_through(monkeypatch):
    agent, _ = _install_fake_agent(monkeypatch, confidence=0.86)
    backend = laya_backend.RealLayaBackend()
    raw = backend.predict({"prompt": "write code"}, {"complexity": {"type": "choice"}})
    assert raw["answers"]["complexity"]["confidence"] == 0.86
    assert raw["answers"]["injection"]["noul"] == 0.02
    assert raw["answers"]["urgency"]["score"] == 3


def test_real_backend_route_escalate_logic(monkeypatch):
    _, _ = _install_fake_agent(monkeypatch, confidence=0.84)
    router = ModelRouter(laya_backend.RealLayaBackend())
    result = router.route("write code")
    assert result["escalate"] is True
    assert result["model"] == "deepseek-v4-pro"

    _, _ = _install_fake_agent(monkeypatch, confidence=0.95)
    router = ModelRouter(laya_backend.RealLayaBackend())
    result = router.route("write code")
    assert result["escalate"] is False
    assert result["model"] == "deepseek-v4.1-flash"


def test_real_backend_loads_typed_decisions_lazily(monkeypatch):
    agent, loads = _install_fake_agent(monkeypatch)
    backend = laya_backend.RealLayaBackend()
    assert loads == []
    assert agent.calls == []
    questions = {"complexity": {"type": "choice"}}
    backend.predict({"prompt": "write code"}, questions)
    backend.predict({"prompt": "write more code"}, questions)
    assert loads == [(("convaiinnovations/laya",), {"subfolder": "typed-decisions"})]
    assert len(agent.calls) == 2


def test_real_backend_classify_wraps_laya_payload(monkeypatch):
    agent, _ = _install_fake_agent(monkeypatch, confidence=0.91)
    router = ModelRouter(laya_backend.RealLayaBackend())
    classification = router.classify("write code")
    assert classification.complexity == "medium"
    assert classification.confidence == 0.91
    assert classification.raw["answers"]["complexity"]["choice"] == "medium"


class ScoreAgent:
    def __init__(self, score):
        self.score = score

    def predict(self, state, questions):
        return {"answers": {"urgency": {"score": self.score, "confidence": 0.9}}}


@pytest.mark.parametrize(
    ("raw_score", "expected_score"),
    [(0, 1), (-3, 1), (2, 2), (2.5, 2.5), (5, 5), (6, 5), (7.5, 5)],
)
def test_real_backend_clamps_scores_into_criteria_range(monkeypatch, raw_score, expected_score):
    # Regression test for the audited real-laya bug: urgency 0 crashed the
    # response schema with a 500. The backend boundary now clamps into the
    # question's criteria range.
    agent = ScoreAgent(raw_score)

    def fake_load(*args, **kwargs):
        return agent

    monkeypatch.setattr(laya_backend, "load", fake_load)
    backend = laya_backend.RealLayaBackend()
    questions = {"urgency": {"type": "score", "criteria": {"min": 1, "max": 5}}}
    raw = backend.predict({"task": "x"}, questions)
    assert raw["answers"]["urgency"]["score"] == expected_score


# ---------------------------------------------------------------------------
# Service-level: latency and health
# ---------------------------------------------------------------------------


def test_mock_route_latency_is_far_below_model_load_time():
    client = TestClient(app)
    started = time.perf_counter()
    response = client.post("/route", json={"prompt": "What is 2 + 2?"})
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert response.status_code == 200
    assert elapsed_ms < 100.0


def test_real_backend_lazy_load_is_called_once_and_health_stays_fast(monkeypatch):
    agent, loads = _install_fake_agent(monkeypatch)
    backend = laya_backend.RealLayaBackend()
    router = ModelRouter(backend)
    router.route("first prompt")
    router.route("second prompt")
    assert len(loads) == 1
    assert len(agent.calls) == 2

    from laya_router.service import health

    started = time.perf_counter()
    payload = health()
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert payload["status"] == "ok"
    assert elapsed_ms < 100.0
