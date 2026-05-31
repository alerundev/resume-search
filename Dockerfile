FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set writable cache directories
ENV HF_HOME=/app/.cache/huggingface
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed

# Pre-download fastembed ONNX model during build (only ~120MB, no torch needed)
RUN mkdir -p /app/.cache/fastembed && \
    python -c "from fastembed import TextEmbedding; m = TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); list(m.embed(['test']))"

# Copy app source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
