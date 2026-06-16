from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "module_3_intent_classification"
DEFAULT_MODEL = "llama-3.1-8b-instant"

INTENTS = {
    "greeting",
    "goodbye",
    "gratitude",
    "asking_mental_health_question",
    "out_of_scope",
}

SYSTEM_PROMPT = """You classify user messages for a mental-health support chatbot.

Return exactly one intent:
- greeting
- goodbye
- gratitude
- asking_mental_health_question
- out_of_scope

Rules:
- If the message includes a mental-health concern, choose asking_mental_health_question even if it also includes greeting or thanks.
- Choose out_of_scope for non-mental-health requests.
- Return only valid JSON with keys: intent, confidence, reason.
- confidence must be a number from 0 to 1.
"""

FEW_SHOT_EXAMPLES = [
    ("hi", "greeting", "The user is only greeting the assistant."),
    ("thanks for listening", "gratitude", "The user is expressing thanks."),
    ("bye, talk later", "goodbye", "The user is ending the conversation."),
    (
        "I feel anxious every night and cannot sleep",
        "asking_mental_health_question",
        "The user describes anxiety and sleep difficulty.",
    ),
    (
        "hello, I feel hopeless today",
        "asking_mental_health_question",
        "Mental-health concern overrides the greeting.",
    ),
    ("what is the capital of France?", "out_of_scope", "The request is unrelated to mental health."),
]

TEST_CASES = [
    ("hello", "greeting"),
    ("good morning", "greeting"),
    ("thank you so much", "gratitude"),
    ("thanks, that helped", "gratitude"),
    ("bye", "goodbye"),
    ("see you later", "goodbye"),
    ("I feel depressed and alone", "asking_mental_health_question"),
    ("why do I panic before sleeping?", "asking_mental_health_question"),
    ("I am angry all the time and it scares me", "asking_mental_health_question"),
    ("hi, I feel anxious today", "asking_mental_health_question"),
    ("thanks, but I still feel hopeless", "asking_mental_health_question"),
    ("write me a SQL query", "out_of_scope"),
    ("who won the world cup?", "out_of_scope"),
    ("recommend a laptop", "out_of_scope"),
]


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class IntentClassifier:
    """Few-shot Groq intent classifier for chatbot routing."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        load_env_file()
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.temperature = temperature
        self.client = None

    def _get_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError("Set GROQ_API_KEY before running live intent classification.")

        if self.client is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise ImportError("Install the Groq SDK with `pip install groq`.") from exc
            self.client = Groq(api_key=self.api_key)

        return self.client

    @staticmethod
    def _build_user_prompt(text: str) -> str:
        examples = []
        for message, intent, reason in FEW_SHOT_EXAMPLES:
            examples.append(
                json.dumps(
                    {
                        "message": message,
                        "intent": intent,
                        "confidence": 0.95,
                        "reason": reason,
                    },
                    ensure_ascii=False,
                )
            )

        return (
            "Examples:\n"
            + "\n".join(examples)
            + "\n\nClassify this message:\n"
            + json.dumps({"message": text}, ensure_ascii=False)
        )

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        content = content.strip()
        match = re.search(r"\{.*\}", content, re.S)
        if match:
            content = match.group(0)
        return json.loads(content)

    @staticmethod
    def _normalize(result: dict[str, Any]) -> dict[str, Any]:
        intent = str(result.get("intent", "")).strip()
        invalid_intent = intent not in INTENTS
        if intent not in INTENTS:
            intent = "out_of_scope"

        try:
            confidence = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(confidence, 1.0))
        if invalid_intent:
            confidence = 0.0
        reason = str(result.get("reason", "")).strip()

        return {
            "intent": intent,
            "confidence": confidence,
            "reason": reason or "No reason provided.",
        }

    def classify(self, text: str) -> dict[str, Any]:
        clean_text = text.strip()
        if not clean_text:
            return {
                "intent": "out_of_scope",
                "confidence": 0.0,
                "reason": "Empty message.",
            }

        client = self._get_client()
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(clean_text)},
            ],
            temperature=self.temperature,
            max_completion_tokens=180,
            top_p=1,
        )

        content = completion.choices[0].message.content or "{}"
        try:
            return self._normalize(self._parse_json(content))
        except (json.JSONDecodeError, TypeError, ValueError):
            return {
                "intent": "out_of_scope",
                "confidence": 0.0,
                "reason": "The model returned an invalid JSON response.",
            }

    def evaluate(self, test_cases: list[tuple[str, str]] = TEST_CASES) -> dict[str, Any]:
        rows = []
        correct = 0

        for text, expected in test_cases:
            prediction = self.classify(text)
            predicted = prediction["intent"]
            is_correct = predicted == expected
            correct += int(is_correct)
            rows.append(
                {
                    "text": text,
                    "expected_intent": expected,
                    "predicted_intent": predicted,
                    "confidence": prediction["confidence"],
                    "correct": is_correct,
                    "reason": prediction["reason"],
                }
            )

        accuracy = correct / len(test_cases)
        return {"accuracy": accuracy, "num_cases": len(test_cases), "rows": rows}

    def save_reports(self, evaluation: dict[str, Any]) -> None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        with (REPORT_DIR / "test_cases.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "text",
                    "expected_intent",
                    "predicted_intent",
                    "confidence",
                    "correct",
                    "reason",
                ],
            )
            writer.writeheader()
            writer.writerows(evaluation["rows"])

        summary = {
            "model": self.model,
            "method": "few-shot LLM prompting with strict JSON output",
            "temperature": self.temperature,
            "intents": sorted(INTENTS),
            "accuracy": evaluation["accuracy"],
            "num_cases": evaluation["num_cases"],
            "routing_note": "Mental-health content overrides greeting or gratitude.",
        }
        (REPORT_DIR / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Module 3 intent classification.")
    parser.add_argument("text", nargs="?", default="I feel anxious and cannot sleep.")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    classifier = IntentClassifier(model=args.model)

    if args.evaluate:
        results = classifier.evaluate()
        classifier.save_reports(results)
        print(json.dumps({"accuracy": results["accuracy"], "num_cases": results["num_cases"]}, indent=2))
    else:
        print(json.dumps(classifier.classify(args.text), indent=2))
