# RAG-Based Mental Health Support Chatbot

This repository contains the module work for the NLP final project.

## Module 1: Language Detection

The language detector is implemented with traditional NLP:

- Vectorizer: character-level TF-IDF with `char_wb` n-grams from 2 to 4 characters
- Classifier: Multinomial Naive Bayes
- Dataset: `papluca/language-identification`
- Supported languages: Arabic, Bulgarian, German, Greek, English, Spanish, French, Hindi, Italian, Japanese, Dutch, Polish, Portuguese, Russian, Swahili, Thai, Turkish, Urdu, Vietnamese, Chinese

### Train and Evaluate

```bash
.\.venv\Scripts\python.exe src\models\language_classifier.py
```

This trains the model, saves it to `src/models/saved_lang_model.pkl`, and writes reports to:

```text
reports/module_1_language_detection/
```

### Run the UI

```bash
.\.venv\Scripts\python.exe src\models\language_detector_ui.py
```

The UI returns the detected language, confidence, and whether the prediction passed the confidence threshold.

## Module 2: Emotion Classification

The emotion classifier uses a fine-tuned transformer:

- Base model: `distilbert-base-uncased`
- Dataset: `dair-ai/emotion`
- Labels: sadness, joy, love, anger, fear, surprise
- Training target: run on Colab T4 using `notebooks/module_2_emotion_training.ipynb`

DistilBERT is used because it keeps most of BERT's language understanding while being smaller and faster, which makes it a better fit for a student project that needs GPU training but practical local inference.

After training in Colab, the notebook saves:

```text
src/models/saved_emotion_model/
reports/module_2_emotion_classification/
```

The local inference class returns the predicted emotion, confidence, and a simple word-occlusion explanation showing which words most affected the predicted emotion.

### Module Integration Plan

The final chatbot will analyze each user message in this order:

```text
User message -> Language Detection -> Emotion Classification -> Intent Classification -> RAG/direct response
```

Module 1 decides the language for routing and response language. Module 2 adds emotional context so later response generation can be gentler for sadness/fear/anger and more direct for neutral informational requests. Crisis handling should still be implemented as a separate safety route later, not inferred from emotion alone.

Run emotion inference after exporting the trained model:

```bash
.\.venv\Scripts\python.exe src\models\emotion_classifier.py "I feel anxious and overwhelmed" --explain
```

Run the emotion UI:

```bash
.\.venv\Scripts\python.exe src\models\emotion_detector_ui.py
```

### Install Dependencies

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
