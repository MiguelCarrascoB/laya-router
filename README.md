# laya-router

A small FastAPI model router that turns a prompt into a typed complexity decision. It keeps the laya inference boundary isolated, uses a deterministic mock by default, and can be switched to pretrained laya models when available.

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

The routing table is trivial → `glm-5.3-flash`, medium → `deepseek-v4.1-flash`, hard → `deepseek-v4-pro`. Classifications below 0.85 confidence escalate to the pro model. Triage blocks injection risk at 0.9 and reports urgency on a 1–5 scale.

The default `MockLayaBackend` is deterministic and needs no model download. To use the verified real backend, install the optional dependency and set `LAYA_BACKEND=real`:

```bash
uv sync --extra real
LAYA_BACKEND=real uv run uvicorn laya_router.service:app
```

`RealLayaBackend` lazily calls `laya.load("convaiinnovations/laya", subfolder="typed-decisions")` on its first prediction and then uses laya's `predict(state, questions)` interface. With `laya==0.3.4`, loading takes about 25 seconds and the first CPU inference about 1.4 seconds in this environment; loading is logged and `/health` stays fast. The `typed-decisions` checkpoint is the workflow-decision model (the `laya` and `multilingual` subfolders are also available). Real laya confidence values can be low (for example, 0.126), so the default 0.85 escalation gate will commonly route to `deepseek-v4-pro`; do not replace laya's own confidence with a fabricated value. Set `ROUTE_CONFIDENCE_GATE` to change the gate, for example `ROUTE_CONFIDENCE_GATE=0.7`.

## Benchmark

```bash
uv run python scripts/benchmark.py -n 8
```

The benchmark prints decisions and rough relative cost/latency savings against always selecting `deepseek-v4-pro`. Its units are illustrative rather than provider billing or measured wall-clock latency.
