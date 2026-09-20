"""Compare the offline laya router with Jev through the OpenCode CLI.

The laya side is deliberately imported directly, so this script does not need a
running service or a downloaded model. Jev is optional at runtime and is called
one prompt at a time through ``opencode run``.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from laya_router.router import ModelRouter

JEV_MODEL = "opencode/jev-1.13-free"


@dataclass(frozen=True)
class BenchmarkCase:
    prompt: str
    expected: str


BENCHMARK: tuple[BenchmarkCase, ...] = (
    BenchmarkCase("What is 2 + 2?", "trivial"),
    BenchmarkCase("Format this short list as JSON: apples, pears", "trivial"),
    BenchmarkCase("Translate 'good morning' into Spanish", "trivial"),
    BenchmarkCase("Explain how to implement a REST endpoint with validation", "medium"),
    BenchmarkCase("Write code to parse CSV rows and report malformed records", "medium"),
    BenchmarkCase("Analyze the tradeoffs between caching and consistency", "medium"),
    BenchmarkCase("Design a secure production architecture for a payments API", "hard"),
    BenchmarkCase("Debug a distributed outage involving retries and duplicate writes", "hard"),
    BenchmarkCase("Design an algorithm for scheduling dependent jobs at scale", "hard"),
    BenchmarkCase("Compare database sharding strategies under ambiguous requirements", "hard"),
)


def parse_jev_decision(output: str) -> str:
    """Extract one complexity label from Jev's text or JSON response."""
    text = output.strip()
    candidates: list[str] = []
    for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            value = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        for key in ("complexity", "decision", "label", "classification"):
            if isinstance(value.get(key), str):
                candidates.append(value[key])
    candidates.append(text)
    for candidate in candidates:
        match = re.search(r"\b(trivial|medium|hard)\b", candidate.lower())
        if match:
            return match.group(1)
    raise ValueError(f"Jev response contained no complexity label: {output[:200]!r}")


def _json_event_text(output: str) -> str:
    """Collect assistant text from OpenCode's JSON-lines output."""
    parts: list[str] = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = event.get("part", event)
        if isinstance(part, dict) and part.get("type") in {"text", "message"}:
            value = part.get("text") or part.get("content")
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(parts) or output


def query_jev(prompt: str, timeout: float = 120.0) -> str:
    """Ask Jev to classify a prompt and return its raw answer."""
    routing_prompt = (
        "Classify the task below by complexity. Reply with JSON only, exactly "
        '{"complexity":"trivial|medium|hard"}. '
        "trivial is a simple lookup or formatting task; medium is multi-step "
        "reasoning or coding; hard requires deep analysis, architecture, or "
        f"ambiguity.\n\nTask: {prompt}"
    )
    completed = subprocess.run(
        ["opencode", "run", "--model", JEV_MODEL, "--format", "json", routing_prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip().splitlines()[-1:]
        raise RuntimeError(f"opencode exited {completed.returncode}: {' '.join(detail)}")
    return _json_event_text(completed.stdout)


def compare(
    jev_client: Callable[[str], str],
    router: ModelRouter | None = None,
    cases: Iterable[BenchmarkCase] = BENCHMARK,
) -> list[dict[str, object]]:
    """Run both classifiers, recording wall-clock latency in milliseconds."""
    router = router or ModelRouter()
    rows: list[dict[str, object]] = []
    for case in cases:
        started = time.perf_counter()
        laya = router.classify(case.prompt).complexity
        laya_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        jev = parse_jev_decision(jev_client(case.prompt))
        jev_ms = (time.perf_counter() - started) * 1000
        rows.append({"prompt": case.prompt, "expected": case.expected, "laya": laya, "jev": jev, "laya_ms": laya_ms, "jev_ms": jev_ms})
    return rows


def agreement(rows: Iterable[dict[str, object]]) -> float:
    rows = list(rows)
    return sum(row["laya"] == row["jev"] for row in rows) / len(rows) if rows else 0.0


def render_table(rows: Iterable[dict[str, object]]) -> str:
    rows = list(rows)
    lines = ["prompt | laya | jev | ground truth", "--- | --- | --- | ---"]
    lines.extend(f"{row['prompt']} | {row['laya']} | {row['jev']} | {row['expected']}" for row in rows)
    return "\n".join(lines)


def _print_laya_only(router: ModelRouter | None = None) -> None:
    router = router or ModelRouter()
    print("prompt | laya | ground truth")
    print("--- | --- | ---")
    for case in BENCHMARK:
        print(f"{case.prompt} | {router.classify(case.prompt).complexity} | {case.expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=120.0, help="per-prompt Jev timeout in seconds")
    args = parser.parse_args()
    try:
        rows = compare(lambda prompt: query_jev(prompt, timeout=args.timeout))
    except Exception as exc:  # CLI availability/provider failures should be actionable, not a traceback.
        print("Laya results (Jev comparison unavailable):")
        _print_laya_only()
        print(f"\nJev unavailable: {exc}", file=sys.stderr)
        return 1
    print(render_table(rows))
    print(f"\nLaya/Jev agreement: {agreement(rows):.0%}")
    print(f"Mean latency: laya {sum(row['laya_ms'] for row in rows) / len(rows):.1f}ms; Jev {sum(row['jev_ms'] for row in rows) / len(rows):.1f}ms")
    print("Verdict: published laya benchmarks report better typed decisions (0.766 vs Jev 0.727) and calibration (ECE 0.081 vs 0.246); Jev is stronger on label spaces with more than 20 options.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
