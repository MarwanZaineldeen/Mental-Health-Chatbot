from __future__ import annotations

import argparse
import json
from typing import Any

from emotion_classifier import EmotionClassifier
from intent_classifier import IntentClassifier
from language_classifier import LanguageDetector


class MessageRouter:
    """Current end-to-end routing flow for Modules 1-3."""

    def __init__(self) -> None:
        self.language_detector = LanguageDetector()
        self.language_detector.load_model()
        self.emotion_classifier = EmotionClassifier()
        self.intent_classifier = IntentClassifier()

    def analyze(self, text: str) -> dict[str, Any]:
        clean_text = text.strip()
        if not clean_text:
            return {"message": "Please enter a user message."}

        return {
            "text": clean_text,
            "language": self.language_detector.predict_with_confidence(clean_text),
            "emotion": self.emotion_classifier.predict_with_confidence(clean_text),
            "intent": self.intent_classifier.classify(clean_text),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the current Modules 1-3 routing flow.")
    parser.add_argument("text", nargs="?", default="hi, I feel anxious and cannot sleep")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    router = MessageRouter()
    print(json.dumps(router.analyze(args.text), indent=2, ensure_ascii=False))
