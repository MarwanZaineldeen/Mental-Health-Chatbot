# Nura: Mental Health Support Chatbot

Nura is your gentle mental wellness companion: an end-to-end mental-health support chatbot built with a modular NLP and RAG architecture. The system detects the user's language, emotion, and intent, applies safety routing, retrieves relevant mental-health context from a Qdrant vector database, and generates a supportive response through Groq.

The project is designed to be explainable, testable, and suitable for a professional portfolio: each module can run independently, produces reports, and is integrated into a FastAPI chatbot interface.

## What The System Does

```text
User message
  -> Language detection
  -> Emotion classification
  -> Safety guardrail
  -> Context-aware intent classification
  -> RAG retrieval when needed
  -> LLM response generation
  -> Same-language supportive answer
```

Key features:

- Multilingual language detection with confidence scores.
- Transformer-based emotion classification with word-level explainability.
- Context-aware LLM intent routing with five-class score distributions.
- Crisis-aware guardrail that bypasses normal RAG when urgent risk is detected.
- RAG retrieval over two mental-health knowledge sources.
- Qdrant Cloud vector database with source filtering.
- E5 multilingual embeddings for cross-lingual retrieval.
- FastAPI backend with a branded Nura production UI, light/dark mode, local saved chats, and a developer testing UI.
- Context-aware follow-up routing using recent browser-session history.
- Final LLM review of language, emotion, and intent predictions.
- Clean reports for every major module.

## Modules

### Module 1: Language Detection

- Dataset: `papluca/language-identification`
- Model: character-level TF-IDF with Multinomial Naive Bayes
- Supported languages: 20 languages including English, Arabic, French, Spanish, German, Chinese, Japanese, Hindi, Urdu, and others
- Report folder: `reports/module_1_language_detection/`

Run:

```powershell
.\.venv\Scripts\python.exe src\data\fetch_language_data.py
.\.venv\Scripts\python.exe src\models\language_classifier.py
.\.venv\Scripts\python.exe src\models\language_detector_ui.py
```

### Module 2: Emotion Classification

- Dataset: `dair-ai/emotion`
- Model: fine-tuned `distilbert-base-uncased`
- Labels: sadness, joy, love, anger, fear, surprise
- Explainability: word-occlusion impact scores
- Report folder: `reports/module_2_emotion_classification/`

The trained model folder is intentionally ignored by Git:

```text
src/models/saved_emotion_model/
```

Run:

```powershell
.\.venv\Scripts\python.exe src\models\emotion_classifier.py "I feel anxious and overwhelmed" --explain
.\.venv\Scripts\python.exe src\models\emotion_detector_ui.py
```

### Module 3: Intent Classification

- Model: Groq `llama-3.1-8b-instant`
- Method: few-shot classification prompt with strict JSON parsing
- Intents: greeting, goodbye, gratitude, asking_mental_health_question, out_of_scope
- Report folder: `reports/module_3_intent_classification/`

Run:

```powershell
.\.venv\Scripts\python.exe src\models\intent_classifier.py "I feel anxious every night"
.\.venv\Scripts\python.exe src\models\intent_classifier.py --evaluate
.\.venv\Scripts\python.exe src\models\intent_detector_ui.py
```

### Module 4: RAG Retrieval

Knowledge sources:

- `cci`: Centre for Clinical Interventions information sheets, cleaned from PDFs and grouped into structure-aware chunks of at most 400 words.
- `amod`: cleaned counseling Q&A pairs from `Amod/mental_health_counseling_conversations`.

Retrieval stack:

- Embedding model: `intfloat/multilingual-e5-base`
- Vector database: Qdrant Cloud
- Collection: `mental_health_rag_v2`
- Retrieval modes:
  - `both`: Balanced Support
  - `cci`: Educational Guidance
  - `amod`: Counseling Style

Build corpora and vector index:

```powershell
.\.venv\Scripts\python.exe src\retrieval\build_cci_corpus.py
.\.venv\Scripts\python.exe src\retrieval\build_amod_qa_corpus.py
.\.venv\Scripts\python.exe src\retrieval\build_vector_index.py --recreate
```

Test retrieval:

```powershell
.\.venv\Scripts\python.exe src\retrieval\retrieval_engine.py "I feel anxious and cannot sleep" --source both --top-k 8
.\.venv\Scripts\python.exe src\retrieval\retrieval_tester_ui.py
```

## FastAPI Chatbot

Run the integrated chatbot:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api_app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Available pages:

- `/` production chatbot UI
- `/developer` developer UI with pipeline state and vector index switching
- `/docs` FastAPI API documentation

API endpoints:

- `GET /health`
- `POST /chat`

Example request:

```json
{
  "message": "I feel anxious every night and cannot sleep",
  "source": "both",
  "top_k": 8,
  "history": []
}
```

## Environment Setup

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a local `.env` file from `.env.example`:

```text
GROQ_API_KEY=your_groq_api_key_here
LANGUAGE_MODEL_REPO_ID=your_hf_username/language-detector-model
LANGUAGE_MODEL_FILENAME=saved_lang_model.pkl
EMOTION_MODEL_ID=your_hf_username/emotion-detector-model
QDRANT_URL=https://your-cluster-url.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION=mental_health_rag_v2
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-base
EMBEDDING_BATCH_SIZE=2
TORCH_NUM_THREADS=1
```

The real `.env` file is ignored by Git and should never be committed.

## Repository Structure

```text
src/
  api_app.py
  data/
    fetch_language_data.py
  models/
    language_classifier.py
    emotion_classifier.py
    intent_classifier.py
    chatbot_pipeline.py
    safety_router.py
    response_generator.py
  retrieval/
    build_cci_corpus.py
    build_amod_qa_corpus.py
    embedding_model.py
    build_vector_index.py
    retrieval_engine.py

notebooks/
  module_1_language_detection.ipynb
  module_2_emotion_training.ipynb
  module_3_intent_classification.ipynb
  module_4_amod_dataset_exploration.ipynb

reports/
  module_1_language_detection/
  module_2_emotion_classification/
  module_3_intent_classification/
  module_4_rag_retrieval/
  integrated_chatbot/
```

## Reports

Each module writes its own evaluation or data-preparation report:

- Language metrics and confusion matrices.
- Emotion classification metrics and explanation examples.
- Intent test cases and accuracy summary.
- CCI corpus summary, Amod dataset summary, Qdrant index summary, and chunking comparison report.
- Integrated chatbot edge-case report for continued chat, mixed-scope queries, multilingual inputs, crisis routing, and suggested questions.

These reports make the project easier to review, debug, and present.

## Deployment Notes

The current app runs locally through FastAPI and is prepared for a Hugging Face Docker Space.

See `DEPLOYMENT.md` for the full Hugging Face Spaces checklist.

Recommended Hugging Face Space setup:

1. Create a new Space with `Docker` as the SDK.
2. Push this repository content to the Space repository.
3. Add the required secrets in the Space settings:

```text
GROQ_API_KEY
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION
LANGUAGE_MODEL_REPO_ID
LANGUAGE_MODEL_FILENAME
EMOTION_MODEL_ID
```

The Dockerfile runs:

```text
uvicorn src.api_app:app --host 0.0.0.0 --port 7860
```

Model artifact policy:

- Upload `saved_lang_model.pkl` to a Hugging Face model repository and set `LANGUAGE_MODEL_REPO_ID`.
- Upload the trained `saved_emotion_model/` files to another Hugging Face model repository and set `EMOTION_MODEL_ID`.
- Keep API keys, model secrets, local data, and local caches outside Git.
- Keep the production UI at `/` and the developer UI at `/developer`.

## Safety Note

This chatbot is for educational and supportive use only. It does not diagnose, replace therapy, prescribe treatment, or handle emergencies as a clinical service. Crisis-like messages are routed to immediate-support guidance and should encourage contacting local emergency services or crisis resources.
