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
DEFAULT_FALLBACK_MODEL = "openai/gpt-oss-20b"
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
        self.max_tokens = int(os.getenv("GROQ_RESPONSE_MAX_TOKENS", "400"))
        self.temperature = float(os.getenv("GROQ_RESPONSE_TEMPERATURE", "0.55"))
        self.timeout_seconds = float(os.getenv("GROQ_REQUEST_TIMEOUT_SECONDS", "12"))
        fallback_models = os.getenv("GROQ_RESPONSE_FALLBACK_MODELS", DEFAULT_FALLBACK_MODEL)
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

            self.client = Groq(api_key=self.api_key, timeout=self.timeout_seconds)

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
        return """You are a supportive mental-health chatbot.

Rules:
- Answer in the same language as the user. This is mandatory for the final answer, not only for language_review.
- If corrected_language_code is ar, write the final answer in Arabic. Do not answer Arabic messages in English.
- Recheck language, emotion, and intent using the user message and recent history, not only the earlier module outputs.
- corrected_emotion must be one of: sadness, joy, love, anger, fear, surprise.
- corrected_intent must be one of: greeting, goodbye, gratitude, asking_mental_health_question, out_of_scope.
- Treat interaction_type as routing context, not as an intent label.
- Use recent conversation history to understand follow-ups and references to earlier messages.
- If the user asks about a personal detail from recent history, answer from recent history and keep corrected_intent as out_of_scope unless the current message asks for mental-health support.
- If the user asks whether you are a therapist, human, doctor, or real person, keep corrected_intent as out_of_scope and explain the boundary warmly.
- Never claim permanent memory. If a detail appears in recent history, say "you mentioned" it naturally.
- If the user shares their name, acknowledge it naturally without explaining memory capabilities.
- Use retrieved context as grounding when retrieval is enabled, but do not copy long passages.
- When retrieval is disabled, respond naturally using the current message and recent history.
- Do not reject a short follow-up merely because it is vague outside its conversation context.
- For mixed messages that mention mental health plus another activity, judge the real request carefully. If the user asks how an activity may support anxiety or mood, keep asking_mental_health_question. If the user mainly asks for unrelated instructions, mark out_of_scope.
- Do not present food, hobbies, or routines as treatments. Frame them only as possible calming activities when appropriate.
- For personal-context or capability questions, answer directly and warmly before inviting the user back to support if helpful.
- For genuinely unrelated requests, briefly explain the mental-health support scope without sounding mechanical.
- Do not diagnose, prescribe medication, or claim to replace a professional.
- Be warm, practical, and useful. Give enough detail to help the current question before suggesting anything else.
- Only include suggested_questions when corrected_intent is asking_mental_health_question. For greeting, goodbye, gratitude, personal-context, capability, or out_of_scope replies, return an empty suggested_questions list.
- For non-crisis mental-health answers, include two or three short suggested_questions that the user could click next. Keep them relevant and gentle.
- suggested_questions must be written from the user perspective as messages the user can send. Use first person: "How can I calm myself right now?" not "How can you calm yourself?"
- Avoid repeating the same suggested_questions across nearby turns. Make each suggestion match the latest user message and move the conversation forward.
- Do not make suggested questions the main content of the answer.
- If the message suggests immediate danger, self-harm, suicide, or harm to others, tell the user to contact local emergency services or the nearest emergency department immediately.
- If retrieved context is weak or unrelated, give a brief general supportive answer and suggest professional support.
- Return only valid JSON with keys: language_review, emotion_review, intent_review, answer, suggested_questions.

JSON schema:
{
  "language_review": {
    "matches_module_1": true,
    "corrected_language_code": "en",
    "reason": "short explanation"
  },
  "emotion_review": {
    "matches_module_2": true,
    "corrected_emotion": "fear",
    "reason": "short explanation"
  },
  "intent_review": {
    "matches_module_3": true,
    "corrected_intent": "asking_mental_health_question",
    "reason": "short explanation"
  },
  "answer": "final user-facing answer",
  "suggested_questions": ["How can I calm myself right now?", "What should I try when this feeling comes back?"]
}
"""

    @staticmethod
    def _user_prompt(state: dict[str, Any]) -> str:
        compact_state = {
            "user_message": state["user_message"],
            "language": state["language"],
            "emotion": state["emotion"],
            "intent": state["intent"],
            "retrieval": state["retrieval"],
            "conversation_history": state.get("conversation_history", []),
        }
        return (
            "Review the language, emotion, and intent again, then answer the user. "
            "If you correct the intent, make the answer match the corrected intent.\n\n"
            "Pipeline state:\n"
            + json.dumps(compact_state, ensure_ascii=False, indent=2)
        )
