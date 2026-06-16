from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "src" / "models" / "saved_emotion_model"


def _load_transformer_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Module 2 requires torch and transformers. Install them with "
            "`python -m pip install -r requirements.txt`, or run the Colab notebook."
        ) from exc

    return torch, AutoModelForSequenceClassification, AutoTokenizer


class EmotionClassifier:
    """Transformer emotion classifier with confidence and simple word-occlusion explanations."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.torch = None
        self.tokenizer = None
        self.model = None
        self.id2label: dict[int, str] = {}

    def load_model(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Emotion model not found at {self.model_dir}. "
                "Train it first with notebooks/module_2_emotion_training.ipynb."
            )

        torch, model_cls, tokenizer_cls = _load_transformer_stack()
        self.torch = torch
        self.tokenizer = tokenizer_cls.from_pretrained(self.model_dir)
        self.model = model_cls.from_pretrained(self.model_dir)
        self.model.eval()

        config_labels = self.model.config.id2label
        self.id2label = {int(key): value for key, value in config_labels.items()}

    def predict_with_confidence(self, text: str) -> dict[str, Any]:
        clean_text = text.strip()
        if not clean_text:
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "is_confident": False,
                "message": "Please enter text to classify.",
            }

        if self.model is None or self.tokenizer is None or self.torch is None:
            self.load_model()

        inputs = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = self.torch.softmax(logits, dim=-1)[0]

        best_index = int(probabilities.argmax().item())
        confidence = float(probabilities[best_index].item())

        return {
            "emotion": self.id2label.get(best_index, str(best_index)),
            "confidence": confidence,
            "is_confident": confidence >= 0.60,
            "message": None,
        }

    def explain(self, text: str, top_k: int = 8) -> dict[str, Any]:
        """Estimate influential words by measuring confidence drop after removing each word."""
        base_prediction = self.predict_with_confidence(text)
        target_emotion = base_prediction["emotion"]
        base_confidence = base_prediction["confidence"]
        words = re.findall(r"\b[\w']+\b", text)

        impacts = []
        for index, word in enumerate(words):
            reduced_words = words[:index] + words[index + 1 :]
            if not reduced_words:
                continue

            reduced_text = " ".join(reduced_words)
            reduced_prediction = self.predict_with_confidence(reduced_text)
            confidence_drop = base_confidence - (
                reduced_prediction["confidence"]
                if reduced_prediction["emotion"] == target_emotion
                else 0.0
            )

            impacts.append(
                {
                    "word": word,
                    "impact": round(float(confidence_drop), 4),
                }
            )

        impacts = sorted(impacts, key=lambda item: item["impact"], reverse=True)
        return {
            "prediction": base_prediction,
            "top_evidence": impacts[:top_k],
            "method": "word occlusion: larger impact means removing the word reduced confidence more",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Module 2 emotion inference.")
    parser.add_argument("text", nargs="?", default="I feel anxious and overwhelmed today.")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    classifier = EmotionClassifier(model_dir=args.model_dir)
    output = classifier.explain(args.text) if args.explain else classifier.predict_with_confidence(args.text)
    print(json.dumps(output, indent=2))
