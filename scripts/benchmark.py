"""Offline benchmark of routing decisions and rough cost/latency savings."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from laya_router.router import ModelRouter

SAMPLES = ["What is 2+2?", "Explain how to implement a REST endpoint", "Design a secure production architecture for this service", "Format this list as JSON"]
COST = {"glm-5.3-flash": 0.2, "deepseek-v4.1-flash": 0.5, "deepseek-v4-pro": 1.0}
LATENCY = {"glm-5.3-flash": 30, "deepseek-v4.1-flash": 80, "deepseek-v4-pro": 180}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=len(SAMPLES))
    args = parser.parse_args()
    router = ModelRouter()
    rows = [(prompt, router.route(prompt)) for prompt in (SAMPLES * ((args.n + len(SAMPLES) - 1) // len(SAMPLES)))[:args.n]]
    base_cost, base_latency = len(rows) * COST["deepseek-v4-pro"], len(rows) * LATENCY["deepseek-v4-pro"]
    used_cost = sum(COST[r["model"]] for _, r in rows); used_latency = sum(LATENCY[r["model"]] for _, r in rows)
    print("prompt | model | confidence | escalate")
    print("--- | --- | ---: | :---")
    for prompt, result in rows: print(f"{prompt[:42]} | {result['model']} | {result['confidence']:.2f} | {result['escalate']}")
    print(f"\nEstimated cost: {used_cost:.2f} vs {base_cost:.2f} always-pro (saved {base_cost-used_cost:.2f})")
    print(f"Estimated latency: {used_latency}ms vs {base_latency}ms always-pro (saved {base_latency-used_latency}ms)")

if __name__ == "__main__": main()
