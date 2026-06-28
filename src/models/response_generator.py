from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

try:
    from google_ai_studio import GoogleAIStudioClient
except ModuleNotFoundError:
    from src.models.google_ai_studio import GoogleAIStudioClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_FALLBACK_MODEL = ""
EMOTION_LABELS = {"sadness", "joy", "love", "anger", "fear", "surprise"}
INTENT_LABELS = {"greeting", "goodbye", "gratitude", "asking_mental_health_question", "out_of_scope"}


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class ResponseGenerator:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        load_env_file()
        self.model = model or os.getenv("GROQ_RESPONSE_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = None
        self.max_tokens = int(os.getenv("GROQ_RESPONSE_MAX_TOKENS", "500"))
        self.temperature = float(os.getenv("GROQ_RESPONSE_TEMPERATURE", "0.55"))
        self.timeout_seconds = float(os.getenv("GROQ_REQUEST_TIMEOUT_SECONDS", "8"))
        self.max_retries = int(os.getenv("GROQ_MAX_RETRIES", "0"))
        self.context_top_k = int(os.getenv("LLM_CONTEXT_TOP_K", "5"))
        self.context_max_chars = int(os.getenv("LLM_CONTEXT_MAX_CHARS", "700"))
        self.history_turns = int(os.getenv("LLM_HISTORY_MESSAGES", "8"))
        fallback_models = os.getenv("GROQ_RESPONSE_FALLBACK_MODELS", "")
        self.fallback_models = [model.strip() for model in fallback_models.split(",") if model.strip()]
        self.google_client = GoogleAIStudioClient(
            model=os.getenv("GOOGLE_RESPONSE_MODEL") or os.getenv("GOOGLE_GENERATION_MODEL"),
            timeout_seconds=float(os.getenv("GOOGLE_REQUEST_TIMEOUT_SECONDS", self.timeout_seconds)),
        )

    def _get_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError("Set GROQ_API_KEY before generating chatbot responses.")

        if self.client is None:
            from groq import Groq

            self.client = Groq(api_key=self.api_key, timeout=self.timeout_seconds, max_retries=self.max_retries)

        return self.client

    def generate(self, state: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._user_prompt(state)},
        ]

        errors = []
        for model in self._model_candidates():
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    top_p=0.9,
                    response_format={"type": "json_object"},
                )
                content = completion.choices[0].message.content or "{}"
                result = self._parse_response(content)
                result["used_model"] = model
                self._enforce_review_labels(result, state)
                self._validate_generated_answer(result)
                return result
            except Exception as error:
                errors.append(f"{model}: {type(error).__name__}")

        if self.google_client.is_configured:
            try:
                content = self.google_client.generate_json(
                    system_prompt=self._system_prompt(),
                    user_prompt=self._user_prompt(state),
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                result = self._parse_response(content)
                result["used_model"] = f"google:{self.google_client.model}"
                self._enforce_review_labels(result, state)
                self._validate_generated_answer(result)
                return result
            except Exception as error:
                errors.append(f"google:{self.google_client.model}: {type(error).__name__}")

        raise RuntimeError("Response generation failed for configured model(s): " + "; ".join(errors))

    @staticmethod
    def _validate_generated_answer(result: dict[str, Any]) -> None:
        if not str(result.get("answer", "")).strip():
            raise ValueError("The model returned an empty answer.")

    def _model_candidates(self) -> list[str]:
        models = [self.model, *self.fallback_models]
        unique_models = []
        for model in models:
            if model and model not in unique_models:
                unique_models.append(model)
        return unique_models

    @staticmethod
    def _enforce_review_labels(result: dict[str, Any], state: dict[str, Any]) -> None:
        emotion_review = result.setdefault("emotion_review", {})
        if emotion_review.get("corrected_emotion") not in EMOTION_LABELS:
            emotion_review["corrected_emotion"] = state["emotion"].get("emotion", "unknown")
            emotion_review["matches_module_2"] = None
            emotion_review["reason"] = "Unsupported emotion review label; Module 2 output retained."
        elif emotion_review.get("corrected_emotion") == state["emotion"].get("emotion"):
            emotion_review["matches_module_2"] = True

        intent_review = result.setdefault("intent_review", {})
        if intent_review.get("corrected_intent") not in INTENT_LABELS:
            intent_review["corrected_intent"] = state["intent"].get("intent", "out_of_scope")
            intent_review["matches_module_3"] = None
            intent_review["reason"] = "Unsupported intent review label; Module 3 output retained."
        elif intent_review.get("corrected_intent") == state["intent"].get("intent"):
            intent_review["matches_module_3"] = True

        questions = result.get("suggested_questions", [])
        if not isinstance(questions, list):
            result["suggested_questions"] = []
            return

        clean_questions = []
        for question in questions:
            question = str(question).strip()
            question = ResponseGenerator._user_perspective_question(question)
            if question and len(question) <= 140:
                clean_questions.append(question)
        result["suggested_questions"] = clean_questions[:3]

    @staticmethod
    def _user_perspective_question(question: str) -> str:
        replacements = {
            "What are some other activities that help you relax?": "What activities can help me relax?",
            "What are some activities that help you relax?": "What activities can help me relax?",
            "How can you": "How can I",
            "How do you": "How do I",
            "What can you": "What can I",
            "What should you": "What should I",
            "Can you": "Can I",
            "you feel": "I feel",
            "your anxiety": "my anxiety",
            "your stress": "my stress",
            "your mood": "my mood",
            "your thoughts": "my thoughts",
            "your body": "my body",
            "your day": "my day",
            "yourself": "myself",
            "help you": "help me",
            "helps you": "helps me",
            "you can": "I can",
            "you might": "I might",
            "you could": "I could",
            "you should": "I should",
        }
        for old, new in replacements.items():
            question = question.replace(old, new)
        return question.strip()

    @staticmethod
    def _parse_response(content: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", content.strip(), re.S)
        if match:
            content = match.group(0)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {
                "language_review": {"matches_module_1": None, "corrected_language_code": None, "reason": "Invalid JSON."},
                "emotion_review": {"matches_module_2": None, "corrected_emotion": None, "reason": "Invalid JSON."},
                "intent_review": {"matches_module_3": None, "corrected_intent": None, "reason": "Invalid JSON."},
                "answer": content.strip(),
                "suggested_questions": [],
            }

        return {
            "language_review": parsed.get("language_review", {}),
            "emotion_review": parsed.get("emotion_review", {}),
            "intent_review": parsed.get("intent_review", {}),
            "answer": str(parsed.get("answer", "")).strip(),
            "suggested_questions": parsed.get("suggested_questions", []),
        }

    @staticmethod
    def _system_prompt() -> str:
        return """You are Nura, a warm mental wellness support chatbot.

Return only valid JSON with keys: language_review, emotion_review, intent_review, answer, suggested_questions.

Core rules:
- Answer in the same language as the user. If corrected_language_code is ar, the answer must be Arabic.
- Recheck language, emotion, and intent from the message and recent history.
- corrected_emotion must be one of: sadness, joy, love, anger, fear, surprise.
- corrected_intent must be one of: greeting, goodbye, gratitude, asking_mental_health_question, out_of_scope.
- Use retrieved context when available, but do not copy long passages. If context is weak, answer generally and gently.
- Use recent history to understand follow-ups and personal-context questions. If the user asks about a detail in history, answer with "you mentioned"; never claim permanent memory.
- If asked your name, say you are Nura. If asked whether you are human, a therapist, or a doctor, explain warmly that you are not a human or licensed professional.
- Do not diagnose, prescribe medication, or replace professional care. For immediate danger or self-harm, advise local emergency support immediately.
- Be supportive, practical, and clear. Give a thoughtful answer with enough detail to be genuinely useful, usually 2 to 4 short paragraphs when the user asks a real mental-health question.
- suggested_questions must be empty unless corrected_intent is asking_mental_health_question.
- For mental-health answers, include 2 short first-person suggested_questions the user can click, such as "How can I calm myself right now?"

JSON shape:
{
  "language_review": {"matches_module_1": true, "corrected_language_code": "en", "reason": "short reason"},
  "emotion_review": {"matches_module_2": true, "corrected_emotion": "fear", "reason": "short reason"},
  "intent_review": {"matches_module_3": true, "corrected_intent": "asking_mental_health_question", "reason": "short reason"},
  "answer": "final user-facing answer",
  "suggested_questions": ["How can I calm myself right now?", "What should I try next?"]
}
"""

    def _user_prompt(self, state: dict[str, Any]) -> str:
        compact_state = {
            "user_message": state["user_message"],
            "language": state["language"],
            "emotion": self._compact_emotion(state.get("emotion", {})),
            "intent": self._compact_intent(state.get("intent", {})),
            "retrieval": self._compact_retrieval(state.get("retrieval", {})),
            "conversation_history": state.get("conversation_history", [])[-self.history_turns :],
        }
        return (
            "Review the language, emotion, and intent again, then answer the user. "
            "If you correct the intent, make the answer match the corrected intent.\n\n"
            "Pipeline state:\n"
            + json.dumps(compact_state, ensure_ascii=False, indent=2)
        )

    @staticmethod
    def _compact_emotion(emotion: dict[str, Any]) -> dict[str, Any]:
        return {
            "emotion": emotion.get("emotion"),
            "confidence": emotion.get("confidence"),
            "is_confident": emotion.get("is_confident"),
        }

    @staticmethod
    def _compact_intent(intent: dict[str, Any]) -> dict[str, Any]:
        return {
            "intent": intent.get("intent"),
            "confidence": intent.get("confidence"),
            "reason": intent.get("reason"),
            "retrieval_query": intent.get("retrieval_query"),
            "contextual_follow_up": intent.get("contextual_follow_up"),
            "interaction_type": intent.get("interaction_type"),
        }

    def _compact_retrieval(self, retrieval: dict[str, Any]) -> dict[str, Any]:
        results = []
        for item in retrieval.get("results", [])[: self.context_top_k]:
            results.append(
                {
                    "rank": item.get("rank"),
                    "score": item.get("score"),
                    "source_type": item.get("source_type"),
                    "title": item.get("title"),
                    "topic": item.get("topic"),
                    "text": self._truncate_text(item.get("text", ""), self.context_max_chars),
                }
            )
        return {
            "enabled": retrieval.get("enabled", False),
            "source": retrieval.get("source"),
            "query": retrieval.get("query"),
            "results": results,
            "error": retrieval.get("error"),
        }

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        text = " ".join(str(text).split())
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "..."
