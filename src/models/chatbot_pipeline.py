from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from emotion_classifier import EmotionClassifier
from intent_classifier import IntentClassifier
from language_classifier import LanguageDetector
from response_generator import ResponseGenerator
from safety_router import crisis_reply, detect_crisis
from src.retrieval.retrieval_engine import RetrievalEngine


class ChatbotPipeline:
    def __init__(self, retrieval_source: str = "both", top_k: int = 5, retrieval_collection: str | None = None) -> None:
        self.retrieval_source = retrieval_source
        self.top_k = top_k
        self.retrieval_collection = retrieval_collection
        self.language_detector = LanguageDetector()
        self.language_detector.load_model()
        self.emotion_classifier = EmotionClassifier()
        self.intent_classifier = IntentClassifier()
        self.retrieval_engine = None
        self.response_generator = ResponseGenerator()

    def run(self, user_message: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        total_started = time.perf_counter()
        clean_message = user_message.strip()
        if not clean_message:
            return {"response": "Please enter a message.", "state": {"timings_ms": {"total": 0.0}}}

        conversation_history = self._prepare_history(clean_message, history or [])
        state = self._analyze(clean_message, conversation_history)
        state["conversation_history"] = conversation_history
        state.setdefault("timings_ms", {})["history_messages"] = len(conversation_history)
        if state["intent"].get("intent") in {"greeting", "gratitude", "goodbye"} and self._is_simple_english_social_message(clean_message):
            state["language"] = {
                "language_code": "en",
                "language_name": "English",
                "confidence": 1.0,
                "is_confident": True,
                "message": "Short English social message normalized before response.",
            }
        language_code = state["language"].get("language_code", "en")

        if state["safety"]["is_crisis"]:
            state["route"] = "crisis"
            state["timings_ms"]["total"] = self._elapsed_ms(total_started)
            return {"response": crisis_reply(language_code), "state": state}

        intent = state["intent"]["intent"]
        use_retrieval = intent == "asking_mental_health_question"
        state["route"] = "rag" if use_retrieval else "direct_response"
        state["retrieval"] = {
            "enabled": use_retrieval,
            "source": self.retrieval_source,
            "top_k": self.top_k,
            "results": [],
        }
        if use_retrieval:
            retrieval_query = state["intent"].get("retrieval_query") or clean_message
            state["retrieval"]["query"] = retrieval_query
            retrieval_started = time.perf_counter()
            try:
                state["retrieval"]["results"] = self._retrieve(retrieval_query)
            except Exception as error:
                state["retrieval"]["error"] = f"{type(error).__name__}"
            finally:
                state["timings_ms"]["retrieval"] = self._elapsed_ms(retrieval_started)

        quick_response = self._quick_direct_response(state, total_started)
        if quick_response:
            return quick_response

        generation_started = time.perf_counter()
        try:
            generated = self.response_generator.generate(state)
            state["timings_ms"]["generation"] = self._elapsed_ms(generation_started)
            if generated.get("used_model"):
                state["response_model"] = generated["used_model"]
            state["llm_review"] = {
                "language": generated.get("language_review", {}),
                "emotion": generated.get("emotion_review", {}),
                "intent": generated.get("intent_review", {}),
            }
            corrected_intent = state["llm_review"]["intent"].get("corrected_intent")
            state["final_intent"] = corrected_intent or state["intent"].get("intent")
            state["final_route"] = "rag" if state["final_intent"] == "asking_mental_health_question" else "direct_response"
            if state["final_intent"] == "asking_mental_health_question":
                state["suggested_questions"] = generated.get("suggested_questions", [])
            else:
                state["suggested_questions"] = []
            response = generated.get("answer") or "I am here with you, but I could not generate a complete response."
        except RuntimeError as error:
            state["timings_ms"]["generation"] = self._elapsed_ms(generation_started)
            state["llm_review"] = {"language": {}, "emotion": {}, "intent": {}}
            state["final_intent"] = state["intent"].get("intent")
            state["final_route"] = state.get("route", "direct_response")
            state["suggested_questions"] = []
            state["generation_error"] = str(error)
            response = (
                "I'm here with you \u2764\ufe0f. I had trouble reaching the response model just now, but we can still take one small step: "
                "pause, take a slow breath, and tell me what feels most urgent in this moment. "
                "If you feel unsafe or might hurt yourself, contact local emergency support or someone you trust right away."
            )
        except Exception as error:
            state["timings_ms"]["generation"] = self._elapsed_ms(generation_started)
            state["llm_review"] = {"language": {}, "emotion": {}, "intent": {}}
            state["final_intent"] = state["intent"].get("intent")
            state["final_route"] = state.get("route", "direct_response")
            state["suggested_questions"] = []
            state["generation_error"] = f"{type(error).__name__}: {error}"
            response = (
                "I'm here with you \u2764\ufe0f. Something interrupted my full response, but you do not have to hold this alone. "
                "Try sending it again in a moment, or tell me the smallest part you want help with first."
            )

        state["timings_ms"]["total"] = self._elapsed_ms(total_started)
        return {"response": response, "suggested_questions": state.get("suggested_questions", []), "state": state}

    def _quick_direct_response(self, state: dict[str, Any], total_started: float) -> dict[str, Any] | None:
        intent = state["intent"].get("intent")
        message = state.get("user_message", "")
        language = state["language"].get("language_code", "en")
        interaction_type = state["intent"].get("interaction_type")
        is_social = intent in {"greeting", "gratitude", "goodbye"} and self._is_simple_english_social_message(message)
        is_standalone_boundary = (
            intent == "out_of_scope"
            and interaction_type != "personal_context"
            and language == "en"
            and state["intent"].get("classification_skipped") == "local_out_of_scope_fast_path"
        )
        if not is_social and not is_standalone_boundary:
            return None

        responses = {
            "greeting": "Hi, I'm Nura. I'm here with you. What would feel helpful to talk through today?",
            "gratitude": "You're very welcome. I'm glad this felt helpful. What would you like to focus on next?",
            "goodbye": "Take care of yourself. I'll be here whenever you want to talk again.",
            "out_of_scope": "I want to keep Nura focused on emotional support and mental wellness. If something is weighing on you, tell me what's been going on, and I'll help you sort through it gently.",
        }
        state["final_intent"] = intent
        state["final_route"] = "direct_response"
        state["suggested_questions"] = []
        fast_path_reason = (
            "Fast response used for a simple English conversational intent."
            if is_social
            else "Fast response used for a standalone out-of-scope boundary."
        )
        state["llm_review"] = {
            "language": {"matches_module_1": True, "reason": fast_path_reason},
            "emotion": {"matches_module_2": True, "reason": "Emotion output retained; no final LLM review needed."},
            "intent": {"matches_module_3": True, "corrected_intent": intent, "reason": fast_path_reason},
        }
        state["generation_skipped"] = "direct_response_fast_path"
        state["timings_ms"]["generation"] = 0.0
        state["timings_ms"]["total"] = self._elapsed_ms(total_started)
        return {"response": responses[intent], "suggested_questions": [], "state": state}

    def _analyze(self, message: str, history: list[dict[str, str]]) -> dict[str, Any]:
        timings: dict[str, float] = {}
        safety_started = time.perf_counter()
        safety = detect_crisis(message)
        timings["safety"] = self._elapsed_ms(safety_started)

        analysis_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            language_future = executor.submit(self._timed_call, "language", self._safe_language, message)
            emotion_future = executor.submit(self._timed_call, "emotion", self._safe_emotion, message)
            if safety["is_crisis"]:
                intent_future = None
            else:
                intent_future = executor.submit(self._timed_call, "intent", self._safe_intent, message, history)

            language, timings["language"] = language_future.result()
            emotion, timings["emotion"] = emotion_future.result()
            if intent_future is None:
                intent = self._crisis_intent(message)
                timings["intent"] = 0.0
            else:
                intent, timings["intent"] = intent_future.result()
        timings["parallel_analysis"] = self._elapsed_ms(analysis_started)

        return {
            "user_message": message,
            "language": language,
            "emotion": emotion,
            "intent": intent,
            "safety": safety,
            "retrieval": {"enabled": False, "results": []},
            "timings_ms": timings,
        }

    def _crisis_intent(self, message: str) -> dict[str, Any]:
        return {
            "intent": "asking_mental_health_question",
            "confidence": 0.0,
            "confidence_margin": 0.0,
            "intent_scores": {},
            "reason": "Crisis guardrail matched before live intent classification.",
            "retrieval_query": message,
            "contextual_follow_up": False,
            "interaction_type": "standalone",
            "classification_skipped": True,
        }

    def _safe_intent(self, message: str, history: list[dict[str, str]]) -> dict[str, Any]:
        try:
            return self.intent_classifier.classify(message, history=history)
        except Exception as error:
            return self._fallback_intent(message, error)

    def set_retrieval_collection(self, collection_name: str | None) -> None:
        if collection_name != self.retrieval_collection:
            self.retrieval_collection = collection_name
            self.retrieval_engine = None

    def _safe_language(self, message: str) -> dict[str, Any]:
        try:
            return self.language_detector.predict_with_confidence(message)
        except Exception as error:
            return {
                "language_code": "en",
                "language_name": "English",
                "confidence": 0.0,
                "is_confident": False,
                "message": f"Language detection unavailable: {type(error).__name__}",
            }

    def _safe_emotion(self, message: str) -> dict[str, Any]:
        try:
            return self.emotion_classifier.predict_with_confidence(message)
        except Exception as error:
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "is_confident": False,
                "scores": {},
                "message": f"Emotion detection unavailable: {type(error).__name__}",
            }

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)

    def _timed_call(self, name: str, func: Any, *args: Any) -> tuple[Any, float]:
        started = time.perf_counter()
        return func(*args), self._elapsed_ms(started)

    @staticmethod
    def _is_simple_english_social_message(message: str) -> bool:
        text = " ".join(message.lower().strip().split())
        english_social = {
            "hi",
            "hello",
            "hey",
            "hey there",
            "good morning",
            "good afternoon",
            "good evening",
            "thanks",
            "thank you",
            "thank you so much",
            "thanks a lot",
            "bye",
            "goodbye",
            "see you",
            "see you later",
            "talk to you later",
            "how are you",
            "how are you?",
            "are you there",
            "are you there?",
        }
        return text in english_social and text.isascii()

    def _fallback_intent(self, message: str, error: Exception) -> dict[str, Any]:
        return {
            "intent": "out_of_scope",
            "confidence": 0.0,
            "confidence_margin": 0.0,
            "intent_scores": {},
            "reason": f"Intent classification unavailable: {type(error).__name__}.",
            "retrieval_query": message,
            "contextual_follow_up": False,
            "interaction_type": "standalone",
        }

    @staticmethod
    def _prepare_history(message: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
        clean_history = [
            {"role": item.get("role", ""), "content": str(item.get("content", "")).strip()}
            for item in history
            if item.get("role") in {"user", "assistant"} and str(item.get("content", "")).strip()
        ]
        if clean_history and clean_history[-1]["role"] == "user" and clean_history[-1]["content"] == message:
            clean_history.pop()
        return clean_history[-8:]

    def warmup(self, include_retrieval: bool = False) -> dict[str, Any]:
        status: dict[str, Any] = {
            "language_model": "loaded",
            "emotion_model": "loaded",
            "intent_classifier": "ready",
            "response_generator": "ready",
            "retrieval": "not_loaded",
        }
        if include_retrieval:
            try:
                self._retrieve("anxiety coping skills")
                status["retrieval"] = "loaded"
            except Exception as error:
                status["retrieval"] = f"unavailable:{type(error).__name__}"
        return status

    def _retrieve(self, message: str) -> list[dict[str, Any]]:
        if self.retrieval_engine is None:
            self.retrieval_engine = RetrievalEngine(collection_name=self.retrieval_collection)
        return self.retrieval_engine.search(message, source=self.retrieval_source, top_k=self.top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the integrated mental-health chatbot pipeline.")
    parser.add_argument("message", nargs="?", default="I feel anxious and cannot sleep.")
    parser.add_argument("--source", choices=["both", "cci", "amod"], default="both")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = ChatbotPipeline(retrieval_source=args.source, top_k=args.top_k)
    output = pipeline.run(args.message)
    print(json.dumps(output, indent=2, ensure_ascii=False))
