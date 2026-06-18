FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TOKENIZERS_PARALLELISM=false
ENV TORCH_NUM_THREADS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-space.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements-space.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "src.api_app:app", "--host", "0.0.0.0", "--port", "7860"]
