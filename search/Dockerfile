# Search API image: FastAPI + ONNX MiniLM, no PyTorch. Build from the repository root:
#   docker build -f search/Dockerfile -t robotics-search .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/models/minilm

WORKDIR /app
COPY search/requirements.txt search/requirements.txt
RUN python -m pip install --no-cache-dir --disable-pip-version-check -r search/requirements.txt

COPY search/__init__.py search/embedder.py search/failure_types.py search/api.py search/
# Model fetched at build time at a pinned revision and checked against its SHA-256.
RUN python -c "from pathlib import Path; from search.embedder import download_model; download_model(Path('/models/minilm'))" \
    && rm -rf /models/minilm/.cache

RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
CMD ["uvicorn", "search.api:app", "--host", "0.0.0.0", "--port", "8000"]
