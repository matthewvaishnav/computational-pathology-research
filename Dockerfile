# Multi-stage GPU-enabled Docker for real-time WSI streaming
# Production-optimized with CUDA, health checks, monitoring

# ============================================================================
# Stage 1: Base CUDA runtime with Python
# ============================================================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS base

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

# Python setup
RUN ln -s /usr/bin/python3.11 /usr/bin/python
RUN python -m pip install --upgrade pip

WORKDIR /app

# ============================================================================
# Stage 2: Dependencies
# ============================================================================
FROM base AS deps

# Copy requirements first (cache layer)
COPY requirements.txt pyproject.toml ./

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional streaming deps
RUN pip install \
    reportlab \
    pydicom \
    pynetdicom \
    fhir.resources \
    redis \
    prometheus-client \
    uvicorn[standard] \
    fastapi \
    websockets

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM deps AS app

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY checkpoints/ ./checkpoints/

# Install package in development mode
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 histocore && \
    chown -R histocore:histocore /app
USER histocore

# Environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV HISTOCORE_LOG_LEVEL=INFO
ENV HISTOCORE_MAX_MEMORY_GB=8
ENV HISTOCORE_BATCH_SIZE=32

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "src.streaming.web_dashboard", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# Stage 4: Production (final)
# ============================================================================
FROM app AS production

# Production optimizations
ENV PYTHONOPTIMIZE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy production config
COPY docker/production.env .env

# Labels for metadata
LABEL maintainer="HistoCore Team"
LABEL version="1.0.0"
LABEL description="Real-time WSI streaming with GPU acceleration"
LABEL gpu.required="true"
LABEL gpu.memory=">=4GB"

# Production command with gunicorn
CMD ["gunicorn", "src.streaming.web_dashboard:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "300", \
     "--keep-alive", "2"]