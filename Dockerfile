FROM python:3.11-slim

# System dependencies for PDF + OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data uploads chroma_db

# Non-root user for security
RUN useradd -m -u 1000 docai && chown -R docai:docai /app
USER docai

# Pre-download HuggingFace models at build time to avoid cold start
RUN python -c "
from transformers import pipeline
import torch
device = 0 if torch.cuda.is_available() else -1
print('Pre-loading models...')
pipeline('text2text-generation', model='google/flan-t5-large', device=device)
pipeline('question-answering', model='distilbert-base-cased-distilled-squad', device=device)
pipeline('summarization', model='sshleifer/distilbart-cnn-12-6', device=device)
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('Models cached.')
" || echo "Model pre-download skipped (will download at runtime)"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "server.py"]
