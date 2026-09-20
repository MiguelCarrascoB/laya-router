import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from compare_jev import BenchmarkCase, agreement, compare, parse_jev_decision, render_table


def test_parse_jev_decision_accepts_json_and_prose():
    assert parse_jev_decision('{"complexity":"hard"}') == "hard"
    assert parse_jev_decision("Decision: medium") == "medium"


def test_compare_with_fake_jev_and_agreement():
    cases = (BenchmarkCase("easy", "trivial"), BenchmarkCase("hard design", "hard"))
    rows = compare(lambda prompt: '{"complexity":"trivial"}' if prompt == "easy" else "hard", cases=cases)
    assert [row["jev"] for row in rows] == ["trivial", "hard"]
    assert agreement(rows) == 1.0
    assert all(row["laya_ms"] >= 0 for row in rows)


def test_render_table_contains_decisions():
    table = render_table([{"prompt": "p", "laya": "medium", "jev": "hard", "expected": "medium"}])
    assert "prompt | laya | jev | ground truth" in table
    assert "p | medium | hard | medium" in table
