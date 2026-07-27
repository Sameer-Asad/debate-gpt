"""Day 6 eval report runner (PRD §9.4).

Runs the judge_consistency and position_bias evals (real LLM calls, not
mocked) and appends evaluation metrics and raw observations per run.

Run with:
    python tests/evals/eval_report.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from test_judge_consistency import run_consistency_eval
from test_position_bias import run_bias_eval

CSV_PATH = _REPO_ROOT / "eval_results.csv"
RAW_JSON_PATH = _REPO_ROOT / "eval_results_raw.json"


def _call_with_backoff(fn, retries: int = 3, base_delay: float = 2.0):
    """Exponential backoff + jitter wrapper (PRD §13: Groq rate limits
    flagged as Medium likelihood during eval runs)."""
    import random

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"  retrying after {delay:.1f}s ({exc})")
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def run_all_evals() -> tuple[list[dict], dict]:
    """Returns (summary_rows, raw_results)."""
    summary_rows: list[dict] = []
    raw_results: dict = {}

    print("Running judge_consistency eval (5 real judge calls)...")
    try:
        consistency = _call_with_backoff(run_consistency_eval)
        summary_rows.append(consistency)
        raw_results["judge_consistency"] = consistency["raw"]
    except Exception as exc:  # noqa: BLE001
        print(f"  judge_consistency eval raised: {exc}")
        summary_rows.append({
            "eval_name": "judge_consistency",
            "metric": "max_stdev_across_criteria",
            "value": None,
            "threshold": "< 1.5",
            "passed": False,
            "detail": {"exception": str(exc)},
        })
        raw_results["judge_consistency"] = {"exception": str(exc)}

    print("Running position_bias eval (2 real judge calls)...")
    try:
        bias = _call_with_backoff(run_bias_eval)
        summary_rows.append(bias)
        raw_results["position_bias"] = bias["raw"]
    except Exception as exc:  # noqa: BLE001
        print(f"  position_bias eval raised: {exc}")
        summary_rows.append({
            "eval_name": "position_bias",
            "metric": "max_pro_con_delta_after_swap_correction",
            "value": None,
            "threshold": "< 1.0",
            "passed": False,
            "detail": {"exception": str(exc)},
        })
        raw_results["position_bias"] = {"exception": str(exc)}

    return summary_rows, raw_results


def write_csv(rows: list[dict], run_timestamp: str) -> None:
    """Appends evaluation summary rows to CSV, creating the header if the file is new."""
    file_exists = CSV_PATH.exists()
    
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "eval_name", "metric", "value", "threshold", "pass_fail"])
        
        for row in rows:
            writer.writerow([
                run_timestamp,
                row["eval_name"],
                row["metric"],
                row["value"],
                row["threshold"],
                "PASS" if row["passed"] else "FAIL",
            ])


def write_raw_json(raw_results: dict, run_timestamp: str) -> None:
    """Appends raw evaluation results as a historical list inside the JSON file."""
    history = []
    
    if RAW_JSON_PATH.exists():
        try:
            with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    history = data
                elif isinstance(data, dict):
                    # Migration: Wrap legacy dict object into list structure
                    history = [{"timestamp": "legacy_run", "results": data}]
        except (json.JSONDecodeError, ValueError):
            history = []

    history.append({
        "timestamp": run_timestamp,
        "results": raw_results
    })

    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"{'eval_name':<20} {'value':<10} {'threshold':<10} {'result'}")
    print("-" * 60)
    for row in rows:
        result = "PASS" if row["passed"] else "FAIL"
        value_str = str(row["value"]) if row["value"] is not None else "ERROR"
        print(f"{row['eval_name']:<20} {value_str:<10} {value_str:<10}" if False else f"{row['eval_name']:<20} {value_str:<10} {row['threshold']:<10} {result}")
    print("=" * 60)
    print(f"\nAppended results to {CSV_PATH}")
    print(f"Appended results to {RAW_JSON_PATH}")


if __name__ == "__main__":
    current_time = datetime.now().isoformat()
    summary_rows, raw_results = run_all_evals()
    write_csv(summary_rows, current_time)
    write_raw_json(raw_results, current_time)
    print_summary(summary_rows)