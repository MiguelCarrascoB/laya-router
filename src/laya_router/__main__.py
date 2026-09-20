import argparse
import json

from .router import ModelRouter

parser = argparse.ArgumentParser(description="Route a task with laya")
parser.add_argument("command", choices=("route", "triage"))
parser.add_argument("text")
args = parser.parse_args()
result = ModelRouter().route(args.text) if args.command == "route" else ModelRouter().triage(args.text)
print(json.dumps(result, indent=2))
