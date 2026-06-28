# Deployment Notes

Nura is container-ready through the project Dockerfile. The container reads the hosting platform `PORT` environment variable and falls back to `7860`, the default Hugging Face Docker Space port.

## Required Secrets

Set these in Render, Hugging Face Spaces, or any other hosting platform:

```text
GROQ_API_KEY
GOOGLE_API_KEY
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION
LANGUAGE_MODEL_REPO_ID
LANGUAGE_MODEL_FILENAME
EMOTION_MODEL_ID
```

Recommended values:

```text
LANGUAGE_MODEL_FILENAME=saved_lang_model.pkl
QDRANT_COLLECTION=mental_health_rag_v2
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-base
GROQ_RESPONSE_MAX_TOKENS=500
GROQ_RESPONSE_TEMPERATURE=0.55
GROQ_INTENT_MODEL=llama-3.1-8b-instant
GROQ_RESPONSE_MODEL=llama-3.1-8b-instant
GROQ_INTENT_FALLBACK_MODELS=
GROQ_RESPONSE_FALLBACK_MODELS=
GROQ_REQUEST_TIMEOUT_SECONDS=8
GROQ_MAX_RETRIES=0
GOOGLE_INTENT_MODEL=gemini-2.5-flash
GOOGLE_RESPONSE_MODEL=gemini-2.5-flash
GOOGLE_REQUEST_TIMEOUT_SECONDS=12
LLM_CONTEXT_TOP_K=5
LLM_CONTEXT_MAX_CHARS=700
LLM_HISTORY_MESSAGES=8
TORCH_NUM_THREADS=1
NURA_WARMUP_ON_START=false
NURA_WARMUP_RETRIEVAL=false
```


## Latency Settings

For Hugging Face Spaces or Render, the most useful latency controls are:

- Keep the Space/app warm before a demo. Cold starts are usually slower than normal requests.
- Use the production UI for users and `/developer` only for debugging, because the developer page returns full pipeline state.
- Keep `GROQ_RESPONSE_MAX_TOKENS` around `500` for fuller answers without making responses too long.
- Set `GROQ_INTENT_FALLBACK_MODELS` and `GROQ_RESPONSE_FALLBACK_MODELS` to another reliable Groq model such as `openai/gpt-oss-20b`. Use comma-separated lists if you want more than one fallback.
- `GROQ_REQUEST_TIMEOUT_SECONDS=8` and `GROQ_MAX_RETRIES=0` stop one slow Groq call from blocking the whole request for too long before trying a fallback.
- Google AI Studio is only used as a second-provider fallback when all configured Groq models fail. Set `GOOGLE_API_KEY` in hosting secrets to enable it.
- `EMBEDDING_BATCH_SIZE` mainly matters when building indexes or embedding multiple texts. It does not speed up a single user query much.
- The app already runs language, emotion, and intent analysis in parallel.

Latency can be measured with:

```powershell
.\.venv\Scripts\python.exe src\evaluation\benchmark_chatbot_latency.py --repeat 1 --source both --top-k 5
```

The report is saved under `reports/integrated_chatbot/latency_benchmark.*`.

## Warmup Strategy

The app exposes `/warmup`. Calling it loads the cached chatbot pipeline before the first real chat request. Use `/warmup?retrieval=true` when you also want to initialize the E5 retrieval model and Qdrant search path.

For demos, you can set:

```text
NURA_WARMUP_ON_START=true
NURA_WARMUP_RETRIEVAL=false
```

Keep retrieval warmup disabled on small free hardware unless you need the first RAG question to be faster, because loading E5 during startup uses more RAM and can slow the Space wake-up.

## Model Artifacts

Keep model artifacts outside Git:

- Upload `saved_lang_model.pkl` to a Hugging Face model repository.
- Upload the trained `saved_emotion_model/` folder to another Hugging Face model repository.
- Point the app to those repositories with `LANGUAGE_MODEL_REPO_ID` and `EMOTION_MODEL_ID`.

## Render Notes

Use the Docker runtime. Render injects `PORT`, so the same Dockerfile can run there without changing the code:

```text
uvicorn src.api_app:app --host 0.0.0.0 --port $PORT
```

## Hugging Face Space Notes

Use a Docker Space and add the same secrets in Space settings. If the Space uses a custom repo name such as `Nura`, push the same project files to that Space remote.

## Local Container Test

```powershell
docker build -t nura .
docker run --env-file .env -p 7860:7860 nura
```

Then open:

```text
http://127.0.0.1:7860
```
