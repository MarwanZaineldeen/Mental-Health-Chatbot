import gradio as gr

from intent_classifier import IntentClassifier


classifier = IntentClassifier()


def predict_intent(text: str) -> dict:
    return classifier.classify(text or "")


interface = gr.Interface(
    fn=predict_intent,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Type a user message...",
        label="User Message",
    ),
    outputs=gr.JSON(label="Intent Result"),
    title="Module 3: Intent Classification",
    description="Few-shot Groq intent classifier for chatbot routing.",
    flagging_mode="never",
)


if __name__ == "__main__":
    interface.launch()
