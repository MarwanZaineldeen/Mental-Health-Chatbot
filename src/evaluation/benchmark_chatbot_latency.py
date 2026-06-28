from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "src" / "models"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))

from chatbot_pipeline import ChatbotPipeline


REPORT_DIR = PROJECT_ROOT / "reports" / "integrated_chatbot"

CASES = [
    {
        "name": "simple_greeting",
        "message": "hello",
        "history": [],
    },
    {
        "name": "mental_health_question",
        "message": "I feel anxious whenever I have to present at work. What can I do?",
        "history": [],
    },
    {
        "name": "contextual_follow_up",
        "message": "What should I do when it starts?",
        "history": [
            {"role": "user", "content": "I get panic symptoms before work presentations."},
            {"role": "assistant", "content": "That sounds stressful. We can make a small plan for that moment."},
        ],
    },
    {
        "name": "out_of_scope",
        "message": "Can you write a SQL query for me?",
        "history": [],
    },
    {
        "name": "gratitude",
        "message": "thank you",
        "history": [],
    },
]


def run_benchmark(repeat: int, source: str, top_k: int) -> dict[str, Any]:
    pipeline = ChatbotPipeline(retrieval_source=source, top_k=top_k)
    rows = []

    for case in CASES:
        for run_number in range(1, repeat + 1):
            output = pipeline.run(case["message"], history=case["history"])
            state = output["state"]
            timings = state.get("timings_ms", {})
            rows.append(
                {
                    "case": case["name"],
                    "run": run_number,
                    "route": state.get("route"),
                    "final_route": state.get("final_route"),
                    "final_intent": state.get("final_intent") or state.get("intent", {}).get("intent"),
                    "generation_skipped": state.get("generation_skipped", ""),
                    "safety_ms": timings.get("safety", 0.0),
                    "language_ms": timings.get("language", 0.0),
                    "emotion_ms": timings.get("emotion", 0.0),
                    "intent_ms": timings.get("intent", 0.0),
                    "parallel_analysis_ms": timings.get("parallel_analysis", 0.0),
                    "retrieval_ms": timings.get("retrieval", 0.0),
                    "generation_ms": timings.get("generation", 0.0),
                    "total_ms": timings.get("total", 0.0),
                    "retrieved_chunks": len(state.get("retrieval", {}).get("results", [])),
                    "response_chars": len(output.get("response", "")),
                    "suggested_questions": len(output.get("suggested_questions", [])),
                    "generation_error": state.get("generation_error", ""),
                }
            )

    summary = {
        "repeat": repeat,
        "source": source,
        "top_k": top_k,
        "average_total_ms": round(mean(row["total_ms"] for row in rows), 2),
        "average_generation_ms": round(mean(row["generation_ms"] for row in rows), 2),
        "rows": rows,
    }
    return summary


def save_report(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "latency_benchmark.json"
    csv_path = REPORT_DIR / "latency_benchmark.csv"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(report["rows"])

    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "average_total_ms": report["average_total_ms"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark integrated chatbot latency by pipeline step.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--source", choices=["both", "cci", "amod"], default="both")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_report(run_benchmark(repeat=args.repeat, source=args.source, top_k=args.top_k))
