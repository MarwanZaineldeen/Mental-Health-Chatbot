from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "src" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "module_1_language_detection"
DEFAULT_MODEL_PATH = MODEL_DIR / "saved_lang_model.pkl"
DEFAULT_HF_MODEL_FILENAME = "saved_lang_model.pkl"

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


class LanguageDetector:
    """Traditional NLP language detector using character TF-IDF and Naive Bayes."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.65,
    ) -> None:
        self.model_path = Path(model_path or os.getenv("LANGUAGE_MODEL_PATH", DEFAULT_MODEL_PATH))
        self.confidence_threshold = confidence_threshold
        self.pipeline = self._build_pipeline()

    @staticmethod
    def _build_pipeline() -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 4),
                        max_features=50000,
                        lowercase=True,
                    ),
                ),
                ("clf", MultinomialNB()),
            ]
        )

    @staticmethod
    def _load_dataset(path: str | Path) -> Any:
        import pandas as pd

        df = pd.read_csv(path)
        required_columns = {"text", "labels"}
        missing_columns = required_columns.difference(df.columns)

        if missing_columns:
            raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

        df = df.dropna(subset=["text", "labels"]).copy()
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"] != ""]
        return df

    def train(
        self,
        train_path: str | Path = DATA_DIR / "lang_train.csv",
        validation_path: str | Path = DATA_DIR / "lang_val.csv",
        test_path: str | Path = DATA_DIR / "lang_test.csv",
    ) -> dict[str, Any]:
        train_df = self._load_dataset(train_path)
        validation_df = self._load_dataset(validation_path)
        test_df = self._load_dataset(test_path)

        print("Training character n-gram TF-IDF language detector...")
        self.pipeline.fit(train_df["text"], train_df["labels"])

        validation_metrics = self.evaluate(validation_df, "validation")
        test_metrics = self.evaluate(test_df, "test")

        self.save_model()
        self.save_reports(validation_metrics, test_metrics)

        return {
            "validation": validation_metrics,
            "test": test_metrics,
            "model_path": str(self.model_path),
        }

    def evaluate(self, df: Any, split_name: str) -> dict[str, Any]:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        predictions = self.pipeline.predict(df["text"])
        labels = sorted(df["labels"].unique())
        report_dict = classification_report(
            df["labels"],
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        report_text = classification_report(
            df["labels"],
            predictions,
            labels=labels,
            zero_division=0,
        )
        matrix = confusion_matrix(df["labels"], predictions, labels=labels)

        accuracy = accuracy_score(df["labels"], predictions)
        print(f"{split_name.title()} accuracy: {accuracy * 100:.2f}%")

        return {
            "split": split_name,
            "accuracy": accuracy,
            "labels": labels,
            "classification_report": report_dict,
            "classification_report_text": report_text,
            "confusion_matrix": matrix.tolist(),
        }

    def save_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"Saved model to {self.model_path}")

    def save_reports(self, validation_metrics: dict[str, Any], test_metrics: dict[str, Any]) -> None:
        import pandas as pd

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        summary = {
            "model": "Character n-gram TF-IDF + Multinomial Naive Bayes",
            "vectorizer": {
                "analyzer": "char_wb",
                "ngram_range": [2, 4],
                "max_features": 50000,
                "lowercase": True,
            },
            "classifier": "MultinomialNB",
            "confidence_threshold": self.confidence_threshold,
            "validation_accuracy": validation_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "languages": LANGUAGE_NAMES,
        }

        (REPORTS_DIR / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        for metrics in (validation_metrics, test_metrics):
            split = metrics["split"]
            labels = metrics["labels"]

            (REPORTS_DIR / f"{split}_classification_report.txt").write_text(
                metrics["classification_report_text"],
                encoding="utf-8",
            )

            pd.DataFrame(metrics["classification_report"]).transpose().to_csv(
                REPORTS_DIR / f"{split}_classification_report.csv",
                encoding="utf-8",
            )

            pd.DataFrame(
                metrics["confusion_matrix"],
                index=labels,
                columns=labels,
            ).to_csv(REPORTS_DIR / f"{split}_confusion_matrix.csv", encoding="utf-8")

        print(f"Saved evaluation reports to {REPORTS_DIR}")

    def load_model(self) -> None:
        model_path = self._resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                "Language model is not available. Train Module 1 locally, or set "
                "LANGUAGE_MODEL_REPO_ID to a Hugging Face model repository."
            )
        self.pipeline = joblib.load(model_path)

    def _resolve_model_path(self) -> Path:
        if self.model_path.exists():
            return self.model_path

        repo_id = os.getenv("LANGUAGE_MODEL_REPO_ID")
        if not repo_id:
            return self.model_path

        filename = os.getenv("LANGUAGE_MODEL_FILENAME", DEFAULT_HF_MODEL_FILENAME)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Install huggingface_hub to load the language model from Hugging Face Hub.") from exc

        return Path(hf_hub_download(repo_id=repo_id, filename=filename))

    def predict(self, text: str) -> str:
        return self.predict_with_confidence(text)["language_code"]

    def predict_with_confidence(self, text: str) -> dict[str, Any]:
        clean_text = text.strip()
        if len(clean_text) < 3:
            return {
                "language_code": "unknown",
                "language_name": "Unknown",
                "confidence": 0.0,
                "is_confident": False,
                "message": "Please enter at least 3 characters.",
            }

        probabilities = self.pipeline.predict_proba([clean_text])[0]
        best_index = int(probabilities.argmax())
        language_code = str(self.pipeline.classes_[best_index])
        confidence = float(probabilities[best_index])

        return {
            "language_code": language_code,
            "language_name": LANGUAGE_NAMES.get(language_code, language_code.upper()),
            "confidence": confidence,
            "is_confident": confidence >= self.confidence_threshold,
            "message": None,
        }


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Module 1 language detector.")
    parser.add_argument("--train-path", default=DATA_DIR / "lang_train.csv", type=Path)
    parser.add_argument("--validation-path", default=DATA_DIR / "lang_val.csv", type=Path)
    parser.add_argument("--test-path", default=DATA_DIR / "lang_test.csv", type=Path)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    _configure_console()
    args = parse_args()

    detector = LanguageDetector(model_path=args.model_path)
    results = detector.train(args.train_path, args.validation_path, args.test_path)

    sample_texts = [
        "I feel anxious and need someone to talk to.",
        "أنا أشعر بالقلق وأحتاج إلى المساعدة.",
        "Je me sens stressé aujourd'hui.",
    ]

    print("\nSample predictions:")
    for sample in sample_texts:
        prediction = detector.predict_with_confidence(sample)
        print(
            f"- {sample!r} -> {prediction['language_name']} "
            f"({prediction['confidence'] * 100:.1f}%)"
        )

    print(f"\nFinal test accuracy: {results['test']['accuracy'] * 100:.2f}%")
