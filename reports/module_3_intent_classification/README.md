# Module 3 Reports

Module 3 classifies user messages into routing intents with few-shot Groq prompting.

Intents:

- `greeting`
- `goodbye`
- `gratitude`
- `asking_mental_health_question`
- `out_of_scope`

Run live evaluation after setting `GROQ_API_KEY`:

```bash
python src/models/intent_classifier.py --evaluate
```

Generated files:

- `metrics_summary.json`
- `test_cases.csv`

The committed `metrics_summary.json` and `test_cases.csv` show the latest live evaluation results.
