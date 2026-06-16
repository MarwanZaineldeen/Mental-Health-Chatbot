from pathlib import Path
import os

import gradio as gr

from emotion_classifier import DEFAULT_MODEL_DIR, EmotionClassifier


classifier = EmotionClassifier()

THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="rose",
    neutral_hue="zinc",
    radius_size="sm",
)

CSS = """
.emotion-shell {
    max-width: 980px;
    margin: 0 auto;
}
.status-box {
    border-left: 4px solid #0f766e;
    padding: 12px 14px;
    background: #f8fafc;
}
.missing-box {
    border-left: 4px solid #be123c;
    padding: 12px 14px;
    background: #fff1f2;
}
"""


def model_status() -> str:
    model_dir = Path(DEFAULT_MODEL_DIR)
    if model_dir.exists():
        return f"<div class='status-box'>Model ready: <code>{model_dir}</code></div>"
    return (
        "<div class='missing-box'>Emotion model is not available locally yet. "
        "Run the Module 2 Colab notebook, then copy the generated "
        "<code>saved_emotion_model</code> folder to "
        f"<code>{model_dir}</code>.</div>"
    )


def predict_emotion(text: str) -> tuple[dict, str]:
    try:
        result = classifier.explain(text or "", top_k=6)
        emotion = result["prediction"]["emotion"]
        confidence = result["prediction"]["confidence"]
        status = f"<div class='status-box'>Predicted <b>{emotion}</b> with {confidence:.1%} confidence.</div>"
        return result, status
    except FileNotFoundError:
        return (
            {
                "error": "Emotion model not found locally.",
                "expected_model_path": str(DEFAULT_MODEL_DIR),
                "next_step": "Run notebooks/module_2_emotion_training.ipynb in Colab and copy src/models/saved_emotion_model back into this project.",
            },
            model_status(),
        )
    except ImportError as exc:
        return (
            {
                "error": "Missing Python dependency.",
                "details": str(exc),
                "next_step": "Install dependencies with python -m pip install -r requirements.txt.",
            },
            "<div class='missing-box'>Missing dependency. Install project requirements.</div>",
        )


with gr.Blocks(title="Emotion Classifier") as interface:
    with gr.Column(elem_classes=["emotion-shell"]):
        gr.Markdown(
            """
            # Emotion Classification
            DistilBERT-based emotion analysis with confidence and word-level evidence.
            """
        )
        status = gr.HTML(value=model_status())

        with gr.Row():
            with gr.Column(scale=5):
                text_input = gr.Textbox(
                    lines=7,
                    label="User message",
                    placeholder="Example: I feel overwhelmed and I cannot sleep.",
                )
                analyze_button = gr.Button("Analyze emotion", variant="primary")
            with gr.Column(scale=4):
                result_output = gr.JSON(label="Prediction")
                summary_output = gr.HTML()

        analyze_button.click(
            fn=predict_emotion,
            inputs=text_input,
            outputs=[result_output, summary_output],
        )


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    interface.launch(theme=THEME, css=CSS, server_port=port)
