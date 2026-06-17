# Multi-stage build for HistoCore production deployment
FROM python:3.9-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopenslide0 \
    libvips42 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY experiments/ experiments/
COPY scripts/ scripts/
COPY configs/ configs/

# Create non-root user
RUN useradd -m -u 1000 histocore && \
    chown -R histocore:histocore /app

USER histocore

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Default command
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# GPU-enabled variant
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu

# Install Python
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    libopenslide0 \
    libvips42 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY experiments/ experiments/
COPY scripts/ scripts/
COPY configs/ configs/

# Create non-root user
RUN useradd -m -u 1000 histocore && \
    chown -R histocore:histocore /app

USER histocore

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')"

# Default command
CMD ["python3", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
