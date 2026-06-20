# Mental Health Chatbot - Project V1
 
A modular AI-powered mental health chatbot pipeline built with Python. The system detects language, classifies emotions and intents, retrieves relevant context, and generates empathetic responses.
 
---
 
## Important Notes Before You Start
 
**Models are NOT included** in this repository due to large file sizes. You must run each module first to train and generate the models locally before running the chatbot.
 
**This code lives on the `Project_V1` branch** — after cloning, make sure to switch to it
 
---
 
## Getting Started
 
### 1. Clone the repository
```bash
git clone https://github.com/MarwanZaineldeen/Mental-Health-Chatbot.git
cd Mental-Health-Chatbot
```
 
### 2. Switch to the correct branch
```bash
git checkout Project_V1
```
Without this step you will be on the main branch and will NOT see this code.
 
### 3. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Mac/Linux
venv\Scripts\activate           # On Windows
```
 
### 4. Install dependencies
```bash
pip install -r requirements.txt
```
 
Or install manually:
```bash
pip install torch transformers scikit-learn pandas numpy datasets huggingface-hub sentence-transformers faiss-cpu langchain openai python-dotenv flask gradio
```
 
### 5. Set up environment variables
```bash
cp .env.example .env
# Open .env and fill in your configuration
```
 
---
 
## Step-by-Step: Run Modules to Generate Models
 
Since models are not pushed to GitHub, you must run each module in order to train and save them locally before using the chatbot.
 
### Step 1 - Language Detection Model
```bash
python Module1.py
```
Trains and saves the language detection model.
 
### Step 2 - Emotion Classification Model
```bash
python Module2.py
```
Trains and saves the emotion classification model.
 
### Step 3 - Intent Classification Model
```bash
python Module3.py
```
Trains and saves the intent classification model.
 
### Step 4 - RAG Retrieval & Response Model
```bash
python Module4.py
```
Builds the retrieval index and saves the response generation model.
 
---
 
## Run the Chatbot
 
Once all 4 modules have been run and models are saved, start the chatbot:
 
```bash
python inference.py
```
 
Or run the full integrated pipeline:
 
```bash
python integration.py
```
 
---
 
## Project Structure
 
```
Mental-Health-Chatbot/
│
├── Module1.py          # Language Detection
├── Module2.py          # Emotion Classification
├── Module3.py          # Intent Classification
├── Module4.py          # RAG Retrieval & Response Generation
├── integration.py      # Full Pipeline (connects all modules)
└── inference.py        # Run the chatbot here
```
 
---
 
## Modules Overview
 
| File | Role |
|------|------|
| Module1.py | Detects the language of the user input (Arabic / English) |
| Module2.py | Classifies the emotion in the message (sadness, anxiety, anger, etc.) |
| Module3.py | Identifies the user intent (seeking support, venting, asking for advice, etc.) |
| Module4.py | Retrieves relevant context using RAG and prepares the response |
| integration.py | Connects all 4 modules into one end-to-end pipeline |
| inference.py | Main entry point — run this to chat with the bot |
 
---
 
## Requirements
 
- Python 3.8+
- torch
- transformers
- scikit-learn
- pandas
- numpy
- datasets
- huggingface-hub
- sentence-transformers
- faiss-cpu
- langchain
- openai
- python-dotenv
- flask
- gradio
---

## API Testing

The chatbot exposes a REST API endpoint that can be tested using Postman or any HTTP client.

### Endpoint

```
POST http://127.0.0.1:7860/chat
```
### Postman Test Screenshot
<img width="1918" height="1017" alt="image" src="https://github.com/user-attachments/assets/b1d5a871-1765-491b-8cab-dcf5337f36e0" />
