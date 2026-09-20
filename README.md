# laya-router

A small FastAPI model router that turns a prompt into a typed complexity decision. It keeps the laya inference boundary isolated, uses a deterministic mock by default, and can be switched to pretrained laya models when available.

## Architecture

```mermaid
flowchart LR
    subgraph api["FastAPI service (service.py)"]
        ROUTE["POST /route"]
        TRIAGE["POST /triage"]
        HEALTH["GET /health"]
    end
    subgraph core["Decision core"]
        ROUTER["ModelRouter (router.py)<br/>classify → confidence gate → route/escalate"]
        GUARD["Guardrail (guardrail.py)<br/>injection noul >= 0.9 → block"]
    end
    subgraph boundary["Laya inference boundary (laya_backend.py)"]
        MOCK["MockLayaBackend<br/>deterministic, offline"]
        REAL["RealLayaBackend<br/>lazy laya load, score clamping"]
        FAIL["FailmonkeyLayaBackend<br/>always fails → 503"]
    end
    ROUTE --> ROUTER
    TRIAGE --> GUARD
    ROUTER --> MOCK & REAL & FAIL
    GUARD --> MOCK & REAL & FAIL
```

The service validates every response against its schema inside the guarded boundary, so backend failures — including schema-violating model answers — surface as `503 {"detail": ...}` rather than 500s.

## Quickstart

```bash
uv sync --extra dev
uv run uvicorn laya_router.service:app --reload
curl -X POST localhost:8000/route -H 'content-type: application/json' -d '{"prompt":"Design a secure production architecture"}'
curl -X POST localhost:8000/triage -H 'content-type: application/json' -d '{"task":"Please do this urgently","context":"release outage"}'
curl localhost:8000/health
```

If uv is unavailable, create a virtualenv and install `.[dev]` with pip. The CLI is also available as `uv run python -m laya_router route "What is 2+2?"`.

## Decisions and backend switch

The routing table is trivial → `glm-5.3-flash`, medium → `deepseek-v4.1-flash`, hard → `deepseek-v4-pro`. Classifications below 0.85 confidence escalate to the pro model (strictly below; at the gate the regular table model is used). Triage blocks injection risk at ≥ 0.9 and reports urgency on a 1–5 scale. If the laya inference boundary raises at request time (backend crash, timeout, malformed payload), both endpoints return `503` with a JSON `{"detail": ...}` body.

```mermaid
flowchart TD
    A["POST /route {prompt}"] --> B["laya classify<br/>complexity + confidence"]
    B --> C{"confidence < 0.85?<br/>(strict; ROUTE_CONFIDENCE_GATE)"}
    C -- "yes — escalate" --> D["deepseek-v4-pro"]
    C -- "no" --> E["ROUTING_TABLE[complexity]<br/>glm-5.3-flash / deepseek-v4.1-flash / deepseek-v4-pro"]
    D --> F["200 {model, confidence, escalate, laya_raw}"]
    E --> F
    B -. "crash / timeout / malformed" .-> X["503 {detail}"]
```

```mermaid
flowchart TD
    T["POST /triage {task, context}"] --> P["Guardrail: laya predicts<br/>injection noul + urgency (1-5)"]
    P --> Q{"noul >= 0.9?"}
    Q -- "yes" --> B1["block: true"]
    Q -- "no" --> B2["block: false"]
    B1 --> R["200 {injection_risk, urgency, block}"]
    B2 --> R
    P -. "out-of-range scores clamped + logged;<br/>any other failure" .-> X["503 {detail}"]
```

The default `MockLayaBackend` is deterministic and needs no model download. To use the verified real backend, install the optional dependency and set `LAYA_BACKEND=real`:

```bash
uv sync --extra real
LAYA_BACKEND=real uv run uvicorn laya_router.service:app
```

`RealLayaBackend` lazily calls `laya.load("convaiinnovations/laya", subfolder="typed-decisions")` on its first prediction and then uses laya's `predict(state, questions)` interface. With `laya==0.3.4`, model loading takes roughly 20–30 seconds in this environment (about 22 s in a September 2026 run) and is logged; once loaded, CPU predictions return in tens to low hundreds of milliseconds and `/health` stays fast. Score-type answers that fall outside the question's criteria (observed: urgency 0 against a 1–5 scale) are clamped into range at the backend boundary and the adjustment is logged. The `typed-decisions` checkpoint is the workflow-decision model (the `laya` and `multilingual` subfolders are also available). Real laya confidence values can be low (for example, 0.126), so the default 0.85 escalation gate will commonly route to `deepseek-v4-pro`; do not replace laya's own confidence with a fabricated value. Set `ROUTE_CONFIDENCE_GATE` to change the gate, for example `ROUTE_CONFIDENCE_GATE=0.7`.

`LAYA_BACKEND=failmonkey` selects an always-failing backend (it raises inside `predict` on every call) so the documented 503 convention can be exercised over real HTTP.

## Testing

```bash
uv run pytest -q
```

All tests run offline against the deterministic mock backend — no model is downloaded or loaded. The current suite is 93 tests across three files:

- `tests/test_router.py` (36 tests): every routing-table entry (trivial → `glm-5.3-flash`, medium → `deepseek-v4.1-flash`, hard → `deepseek-v4-pro`) reached through both real prompts and canned backends; confidence-gate semantics parametrized around the 0.85 boundary and the `ROUTE_CONFIDENCE_GATE` env override with explicit expected outcomes (strict `<`: confidence exactly at the gate does not escalate); unknown and case-variant complexity labels rejected by the `Classification` schema; empty and very long prompts; the real-backend path with a fake laya `Agent` verifying lazy single load, raw-field passthrough, escalation, and clamping of out-of-range scores into the question criteria; and a <100 ms bound on the mock `/route` path.
- `tests/test_guardrail.py` (54 tests): the triage guardrail — injection (`noul`) block threshold at 0.9 parametrized over boundary values 0.0–1.0 (inclusive at 0.9); urgency 1–5 mapping and schema range checks; a canonical keyword injection corpus (blocked) plus benign corpus (allowed) and encoded/roleplay variants the mock explicitly does not flag (documented limitation); malformed `/route` and `/triage` requests returning 422 with JSON `detail`; end-to-end FastAPI request/response contracts; and `503 Service Unavailable` with `{"detail": ...}` when the laya inference boundary raises (crash, timeout, malformed payload) — including schema-violating backend answers (regression tests for the audited urgency-0 and out-of-range-confidence cases) and the `LAYA_BACKEND=failmonkey` backend on both endpoints.
- `tests/test_compare_jev.py` (3 tests): Jev comparison parsing, agreement, and table rendering.

## Benchmark

```bash
uv run python scripts/benchmark.py -n 8
```

The benchmark prints decisions and rough relative cost/latency savings against always selecting `deepseek-v4-pro`. Its units are illustrative rather than provider billing or measured wall-clock latency.

## Jev comparison

Jev (`opencode/jev-1.13-free`) is a competing decision-model API available through the OpenCode CLI. Compare it with the deterministic, offline laya backend using:

```bash
uv run python scripts/compare_jev.py
```

The script sends ten fixed prompts to both classifiers and prints each decision, ground truth, model agreement, and mean wall-clock latency. It uses this non-interactive OpenCode command internally:

```bash
opencode run --model opencode/jev-1.13-free --format json '...routing prompt...'
```

> **Run note (2026-09-20):** a comparison run with `--timeout 60` timed out on Jev's first prompt and exited 1. One retry with `--timeout 120` reached Jev but returned HTTP 500 (`provider.internal`) and also exited 1. This is consistent with the earlier server-side outage, not an invocation problem; the script printed laya-only results.
>
> When Jev answers, the verdict line reflects laya's published benchmarks against Jev: laya wins typed decisions (0.766 vs 0.727) and calibration (ECE 0.081 vs 0.246) at ~7× lower latency; Jev is stronger on label spaces with more than 20 options. Straightforward tasks dominate the benchmark set above, and laya-with-mock missed only 1/10 (a linguistically-trivial but multi-step case it judged medium).

Real results from the 2026-09-20 run (the Jev retry remained unavailable):

| Prompt | Expected | Laya (choice, confidence, latency) | Jev (choice, latency) | Laya | Jev |
| --- | --- | --- | --- | --- | --- |
| What is 2 + 2? | trivial | trivial, 0.960, 0.021ms | unavailable (provider 500; retried 2026-09-20) | ✓ | — |
| Format this short list as JSON: apples, pears | trivial | trivial, 0.960, 0.006ms | unavailable (provider 500; retried 2026-09-20) | ✓ | — |
| Translate 'good morning' into Spanish | trivial | medium, 0.890, 0.003ms | unavailable (provider 500; retried 2026-09-20) | ✗ | — |
| Explain how to implement a REST endpoint with validation | medium | medium, 0.890, 0.002ms | unavailable (provider 500; retried 2026-09-20) | ✓ | — |
| Write code to parse CSV rows and report malformed records | medium | medium, 0.890, 0.002ms | unavailable (provider 500; retried 2026-09-20) | ✓ | — |
| Analyze the tradeoffs between caching and consistency | medium | medium, 0.890, 0.002ms | unavailable (provider 500; retried 2026-09-20) | ✓ | — |
| Design a secure production architecture for a payments API | hard | hard, 0.930, 0.001ms | unavailable (provider 500; retried 2026-09-20) | ✓ | — |
| Debug a distributed outage involving retries and duplicate writes | hard | hard, 0.930, 0.001ms | unavailable (provider 500; retried 2026-09-20) | ✓ | — |
| Design an algorithm for scheduling dependent jobs at scale | hard | hard, 0.930, 0.001ms | unavailable (provider 500; retried 2026-09-20) | ✓ | — |
| Compare database sharding strategies under ambiguous requirements | hard | hard, 0.930, 0.002ms | unavailable (provider 500; retried 2026-09-20) | ✓ | — |

**Summary:** laya correct **9/10**; Jev **unavailable (0/10 evaluated)**; laya/Jev agreement **N/A**; mean latency: laya **0.004ms**, Jev **N/A**.

_Run 2026-09-20; deterministic mock laya backend; Jev via the OpenCode CLI._

If OpenCode or Jev is unavailable, laya results are still printed and the command exits non-zero. The published laya benchmark reports stronger typed decisions (0.766 vs Jev's 0.727) and calibration (ECE 0.081 vs 0.246), while Jev performs better on label spaces with more than 20 options.
