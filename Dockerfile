# ============================================================
# Stage 1 : Build Next.js
# ============================================================
FROM node:20-bookworm AS frontend-builder

WORKDIR /frontend

# Better Docker cache
COPY frontend/package*.json ./

RUN npm ci

COPY frontend .

RUN npm run build


# ============================================================
# Stage 2 : Python Runtime
# ============================================================
FROM python:3.10-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ------------------------------------------------------------
# Linux Packages
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    build-essential \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Install NodeJS
# ------------------------------------------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y nodejs

# ------------------------------------------------------------
# Python Requirements
# ------------------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip

# Install PyTorch compiled for CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install the remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt
# ------------------------------------------------------------
# SpaCy
# ------------------------------------------------------------
RUN python -m spacy download en_core_web_sm

# ------------------------------------------------------------
# Backend
# ------------------------------------------------------------
COPY backend /app/backend

# ------------------------------------------------------------
# Frontend
# ------------------------------------------------------------
COPY --from=frontend-builder /frontend /app/frontend

# ------------------------------------------------------------
# Startup Script
# ------------------------------------------------------------
COPY start.sh /app/start.sh

RUN sed -i 's/\r$//' /app/start.sh && \
    chmod +x /app/start.sh

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
ENV BACKEND_URL=http://127.0.0.1:7861
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
ENV TORCH_HOME=/root/.cache/torch
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 7860

# ------------------------------------------------------------
# Healthcheck
# ------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s CMD curl -f http://localhost:7860 || exit 1

CMD ["/app/start.sh"]
