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

# Install Python dependencies (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download sentence-transformer embedding model at build time
# This avoids a slow cold start on first document upload
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('Embedding model cached.')" \
    || echo "Embedding model download skipped (will download at runtime)"

# Copy application code
COPY . .

# Create persistent data directories
RUN mkdir -p data uploads chroma_db

# Non-root user for security
RUN useradd -m -u 1000 docai && chown -R docai:docai /app
USER docai

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "server.py"]
