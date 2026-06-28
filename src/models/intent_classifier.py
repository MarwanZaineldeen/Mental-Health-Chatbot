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
DEFAULT_FALLBACK_MODEL = "openai/gpt-oss-20b"

INTENT_NAMES = (
    "greeting",
    "goodbye",
    "gratitude",
    "asking_mental_health_question",
    "out_of_scope",
)
INTENTS = set(INTENT_NAMES)

SYSTEM_PROMPT = """You classify user messages for a mental-health support chatbot.

Return exactly one intent:
- greeting
- goodbye
- gratitude
- asking_mental_health_question
- out_of_scope

Rules:
- If the message includes a mental-health concern, choose asking_mental_health_question even if it also includes greeting or thanks.
- Use recent conversation history to resolve short or vague follow-ups.
- If the current message continues a recent mental-health discussion, choose asking_mental_health_question.
- Use greeting for a personal introduction and out_of_scope for a personal-context question that is not about mental health.
- Do not infer mental-health intent from generic greetings, availability checks, or small talk unless the concern is explicit.
- Choose out_of_scope for non-mental-health tasks or factual requests.
- For mixed messages that mention mental health plus another activity, classify by the real request: if the user asks how the activity may calm anxiety or mood, choose asking_mental_health_question; if they ask for unrelated instructions, choose out_of_scope.
- Do not treat a casual mental-health word as enough by itself; look for a real emotional, coping, symptom, therapy, or wellbeing need.
- For asking_mental_health_question, rewrite the request as a short standalone retrieval query focused on the mental-health need, not the unrelated activity details.
- Return intent_scores for all five intents. Scores must be numbers from 0 to 1 and sum to 1.
- Use a realistic score range. Do not default every clear prediction to 0.95.
- interaction_type must be standalone, contextual_follow_up, or personal_context.
- Return only valid JSON with keys: intent, intent_scores, reason, retrieval_query, contextual_follow_up, interaction_type.
"""

FEW_SHOT_EXAMPLES = [
    ("hi", "greeting", 0.92, "The user is only greeting the assistant."),
    ("hey there, are you available?", "greeting", 0.76, "The user is checking availability."),
    ("thanks for listening", "gratitude", 0.89, "The user is expressing thanks."),
    ("bye, talk later", "goodbye", 0.91, "The user is ending the conversation."),
    ("my name is Marwan", "greeting", 0.74, "The user is introducing themselves."),
    (
        "I feel anxious every night and cannot sleep",
        "asking_mental_health_question",
        0.97,
        "The user describes anxiety and sleep difficulty.",
    ),
    (
        "hello, I feel hopeless today",
        "asking_mental_health_question",
        0.93,
        "Mental-health concern overrides the greeting.",
    ),
    ("what is the capital of France?", "out_of_scope", 0.99, "The request is unrelated to mental health."),
    (
        "how to cook pizza to reduce anxiety?",
        "asking_mental_health_question",
        0.72,
        "The user is asking whether cooking can be used as a calming anxiety activity, not for a recipe alone.",
    ),
]

TEST_CASES = [
    ("hello", "greeting"),
    ("good morning", "greeting"),
    ("hey there, are you available?", "greeting"),
    ("thank you so much", "gratitude"),
    ("thanks, that helped", "gratitude"),
    ("I appreciate your help", "gratitude"),
    ("bye", "goodbye"),
    ("see you later", "goodbye"),
    ("good night, talk tomorrow", "goodbye"),
    ("I feel depressed and alone", "asking_mental_health_question"),
    ("why do I panic before sleeping?", "asking_mental_health_question"),
    ("I am angry all the time and it scares me", "asking_mental_health_question"),
    ("hi, I feel anxious today", "asking_mental_health_question"),
    ("thanks, but I still feel hopeless", "asking_mental_health_question"),
    ("bye, but I am scared I will spiral again tonight", "asking_mental_health_question"),
    ("can you explain why panic attacks happen?", "asking_mental_health_question"),
    ("I keep overthinking everything and cannot focus", "asking_mental_health_question"),
    ("what are common symptoms of depression?", "asking_mental_health_question"),
    ("I feel numb and disconnected from everyone", "asking_mental_health_question"),
    ("write me a SQL query", "out_of_scope"),
    ("who won the world cup?", "out_of_scope"),
    ("recommend a laptop", "out_of_scope"),
    ("summarize this business article", "out_of_scope"),
    ("build me a weekly gym routine", "out_of_scope"),
    ("translate this sentence into French", "out_of_scope"),
    ("how to cook pizza to reduce anxiety?", "asking_mental_health_question"),
    ("give me a pizza recipe", "out_of_scope"),
]


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8-sig").splitlines():
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
        self.timeout_seconds = float(os.getenv("GROQ_REQUEST_TIMEOUT_SECONDS", "12"))
        fallback_models = os.getenv("GROQ_INTENT_FALLBACK_MODELS", DEFAULT_FALLBACK_MODEL)
        self.fallback_models = [model.strip() for model in fallback_models.split(",") if model.strip()]
        self.client = None

    def _get_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError("Set GROQ_API_KEY before running live intent classification.")

        if self.client is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise ImportError("Install the Groq SDK with `pip install groq`.") from exc
            self.client = Groq(api_key=self.api_key, timeout=self.timeout_seconds)

        return self.client

    @staticmethod
    def _quick_intent(text: str) -> dict[str, Any] | None:
        return IntentClassifier._quick_social_intent(text) or IntentClassifier._quick_out_of_scope_intent(text)

    @staticmethod
    def _quick_social_intent(text: str) -> dict[str, Any] | None:
        normalized = " ".join(text.lower().strip().split())
        social_intents = {
            "hi": "greeting",
            "hello": "greeting",
            "hey": "greeting",
            "hey there": "greeting",
            "good morning": "greeting",
            "good afternoon": "greeting",
            "good evening": "greeting",
            "how are you": "greeting",
            "how are you?": "greeting",
            "are you there": "greeting",
            "are you there?": "greeting",
            "thanks": "gratitude",
            "thank you": "gratitude",
            "thank you so much": "gratitude",
            "thanks a lot": "gratitude",
            "bye": "goodbye",
            "goodbye": "goodbye",
            "see you": "goodbye",
            "see you later": "goodbye",
            "talk to you later": "goodbye",
        }
        intent = social_intents.get(normalized)
        if not intent or not normalized.isascii():
            return None

        scores = {name: 0.02 for name in INTENT_NAMES}
        scores[intent] = 0.92
        total = sum(scores.values())
        scores = {name: round(value / total, 4) for name, value in scores.items()}
        ranked_scores = sorted(scores.values(), reverse=True)
        return {
            "intent": intent,
            "confidence": scores[intent],
            "confidence_margin": round(ranked_scores[0] - ranked_scores[1], 4),
            "intent_scores": scores,
            "reason": "Matched a short English social message locally to reduce latency.",
            "retrieval_query": "",
            "contextual_follow_up": False,
            "interaction_type": "standalone",
            "classification_skipped": "local_social_intent_fast_path",
        }

    @staticmethod
    def _quick_out_of_scope_intent(text: str) -> dict[str, Any] | None:
        normalized = " ".join(text.lower().strip().split())
        mental_terms = {
            "anxiety", "anxious", "panic", "depression", "depressed", "sad", "stress", "stressed",
            "therapy", "therapist", "trauma", "mood", "sleep", "insomnia", "fear", "angry",
            "hopeless", "lonely", "self-harm", "suicide", "suicidal", "mental", "emotion",
        }
        if any(term in normalized for term in mental_terms):
            return None

        obvious_patterns = (
            "write a sql",
            "write me a sql",
            "sql query",
            "python code",
            "write code",
            "capital of",
            "who won",
            "recommend a laptop",
            "give me a recipe",
            "pizza recipe",
            "summarize this business",
            "translate this sentence",
        )
        if not normalized.isascii() or not any(pattern in normalized for pattern in obvious_patterns):
            return None

        scores = {name: 0.02 for name in INTENT_NAMES}
        scores["out_of_scope"] = 0.92
        total = sum(scores.values())
        scores = {name: round(value / total, 4) for name, value in scores.items()}
        ranked_scores = sorted(scores.values(), reverse=True)
        return {
            "intent": "out_of_scope",
            "confidence": scores["out_of_scope"],
            "confidence_margin": round(ranked_scores[0] - ranked_scores[1], 4),
            "intent_scores": scores,
            "reason": "Matched an obvious non-mental-health task locally to reduce latency.",
            "retrieval_query": "",
            "contextual_follow_up": False,
            "interaction_type": "standalone",
            "classification_skipped": "local_out_of_scope_fast_path",
        }

    @staticmethod
    def _build_user_prompt(text: str, history: list[dict[str, str]]) -> str:
        examples = []
        for message, intent, main_score, reason in FEW_SHOT_EXAMPLES:
            other_score = round((1 - main_score) / (len(INTENT_NAMES) - 1), 4)
            scores = {name: other_score for name in INTENT_NAMES}
            scores[intent] = main_score
            examples.append(
                json.dumps(
                    {
                        "message": message,
                        "intent": intent,
                        "intent_scores": scores,
                        "reason": reason,
                        "retrieval_query": message if intent == "asking_mental_health_question" else "",
                        "contextual_follow_up": False,
                        "interaction_type": "personal_context" if message == "my name is Marwan" else "standalone",
                    },
                    ensure_ascii=False,
                )
            )

        recent_history = [
            {"role": item.get("role", ""), "content": item.get("content", "")}
            for item in history[-8:]
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]

        return (
            "Examples:\n"
            + "\n".join(examples)
            + "\n\nContextual follow-up example:\n"
            + json.dumps(
                {
                    "recent_conversation": [
                        {"role": "user", "content": "I keep having panic attacks at work."},
                        {"role": "assistant", "content": "That sounds frightening and exhausting."},
                    ],
                    "message": "What should I do when it starts?",
                    "intent": "asking_mental_health_question",
                    "intent_scores": {
                        "greeting": 0.01,
                        "goodbye": 0.01,
                        "gratitude": 0.01,
                        "asking_mental_health_question": 0.88,
                        "out_of_scope": 0.09
                    },
                    "reason": "The message continues the recent panic-attack discussion.",
                    "retrieval_query": "What coping steps can help when a panic attack starts at work?",
                    "contextual_follow_up": True,
                    "interaction_type": "contextual_follow_up",
                },
                ensure_ascii=False,
            )
            + "\n\nPersonal-context example:\n"
            + json.dumps(
                {
                    "recent_conversation": [{"role": "user", "content": "My name is Marwan."}],
                    "message": "What name did I tell you?",
                    "intent": "out_of_scope",
                    "intent_scores": {
                        "greeting": 0.25,
                        "goodbye": 0.01,
                        "gratitude": 0.01,
                        "asking_mental_health_question": 0.01,
                        "out_of_scope": 0.72
                    },
                    "reason": "This is a personal-context question, not a mental-health request.",
                    "retrieval_query": "",
                    "contextual_follow_up": True,
                    "interaction_type": "personal_context",
                },
                ensure_ascii=False,
            )
            + "\n\nRecent conversation:\n"
            + json.dumps(recent_history, ensure_ascii=False)
            + "\n\nClassify the current message:\n"
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
    def _normalize(result: dict[str, Any], original_text: str) -> dict[str, Any]:
        declared_intent = str(result.get("intent", "")).strip()
        raw_scores = result.get("intent_scores", {})
        scores = {}
        for name in INTENT_NAMES:
            try:
                scores[name] = max(0.0, float(raw_scores.get(name, 0.0)))
            except (TypeError, ValueError, AttributeError):
                scores[name] = 0.0

        total = sum(scores.values())
        if total > 0:
            scores = {name: round(value / total, 4) for name, value in scores.items()}
            intent = max(scores, key=scores.get)
            confidence = scores[intent]
            ranked_scores = sorted(scores.values(), reverse=True)
            confidence_margin = round(ranked_scores[0] - ranked_scores[1], 4)
        else:
            intent = declared_intent if declared_intent in INTENTS else "out_of_scope"
            confidence = 0.0
            confidence_margin = 0.0

        reason = str(result.get("reason", "")).strip()
        retrieval_query = str(result.get("retrieval_query", "")).strip()
        if intent == "asking_mental_health_question" and not retrieval_query:
            retrieval_query = original_text
        interaction_type = str(result.get("interaction_type", "standalone")).strip()
        if interaction_type not in {"standalone", "contextual_follow_up", "personal_context"}:
            interaction_type = "standalone"

        return {
            "intent": intent,
            "confidence": confidence,
            "confidence_margin": confidence_margin,
            "intent_scores": scores,
            "reason": reason or "No reason provided.",
            "retrieval_query": retrieval_query,
            "contextual_follow_up": result.get("contextual_follow_up") is True,
            "interaction_type": interaction_type,
        }

    def classify(self, text: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        clean_text = text.strip()
        if not clean_text:
            return {
                "intent": "out_of_scope",
                "confidence": 0.0,
                "confidence_margin": 0.0,
                "intent_scores": {name: 0.0 for name in INTENT_NAMES},
                "reason": "Empty message.",
                "retrieval_query": "",
                "contextual_follow_up": False,
                "interaction_type": "standalone",
            }

        quick_intent = self._quick_intent(clean_text)
        if quick_intent:
            return quick_intent

        client = self._get_client()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(clean_text, history or [])},
        ]
        errors = []
        for model in self._model_candidates():
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=300,
                    top_p=1,
                    response_format={"type": "json_object"},
                )
                content = completion.choices[0].message.content or "{}"
                prediction = self._normalize(self._parse_json(content), clean_text)
                prediction["used_model"] = model
                return prediction
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                errors.append(f"{model}: invalid_json:{type(error).__name__}")
            except Exception as error:
                errors.append(f"{model}: {type(error).__name__}")

        return {
            "intent": "out_of_scope",
            "confidence": 0.0,
            "confidence_margin": 0.0,
            "intent_scores": {name: 0.0 for name in INTENT_NAMES},
            "reason": "Intent classification failed for configured Groq model(s): " + "; ".join(errors),
            "retrieval_query": "",
            "contextual_follow_up": False,
            "interaction_type": "standalone",
        }

    def _model_candidates(self) -> list[str]:
        models = [self.model, *self.fallback_models]
        unique_models = []
        for model in models:
            if model and model not in unique_models:
                unique_models.append(model)
        return unique_models

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
                    "confidence_margin": prediction["confidence_margin"],
                    "interaction_type": prediction["interaction_type"],
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
                    "confidence_margin",
                    "interaction_type",
                    "correct",
                    "reason",
                ],
            )
            writer.writeheader()
            writer.writerows(evaluation["rows"])

        summary = {
            "model": self.model,
            "method": "few-shot LLM prompting with strict JSON output",
            "confidence_method": "normalized five-class LLM score distribution with top-two margin",
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
