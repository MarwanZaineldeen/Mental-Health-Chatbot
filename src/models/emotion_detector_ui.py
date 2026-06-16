import gradio as gr

from emotion_classifier import EmotionClassifier


classifier = EmotionClassifier()


def predict_emotion(text: str) -> dict:
    return classifier.explain(text or "", top_k=6)


interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Type an English mental-health related message...",
        label="User Message",
    ),
    outputs=gr.JSON(label="Emotion Result"),
    title="Module 2: Emotion Classification",
    description="DistilBERT emotion classifier with confidence and word-occlusion explanation.",
    flagging_mode="never",
)


if __name__ == "__main__":
    interface.launch()
