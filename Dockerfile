# Protein Design MCP Server - All-in-One Docker Image
#
# This Dockerfile creates an image with RFdiffusion, ProteinMPNN,
# ESMFold, and the MCP server. Model weights are downloaded at runtime.
#
# Build: docker build -t protein-design-mcp .
# Run:   docker-compose up

# =============================================================================
# Stage 1: Base image with CUDA and Python
# =============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# =============================================================================
# Stage 2: Install RFdiffusion and ProteinMPNN (code only, no weights)
# =============================================================================
FROM base AS tools

WORKDIR /opt

# Install PyTorch first (required for se3-transformer)
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Clone RFdiffusion and install dependencies manually
RUN git clone --depth 1 https://github.com/RosettaCommons/RFdiffusion.git \
    && mkdir -p RFdiffusion/models

# Install se3-transformer from RFdiffusion's env
RUN cd RFdiffusion/env/SE3Transformer \
    && pip install --no-cache-dir -e .

# Install RFdiffusion dependencies (without full package install)
RUN pip install --no-cache-dir \
    hydra-core \
    omegaconf \
    icecream \
    pyrsistent \
    e3nn \
    dgl

# Clone ProteinMPNN
RUN git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git \
    && mkdir -p ProteinMPNN/vanilla_model_weights

# =============================================================================
# Stage 3: Install Python dependencies and MCP server
# =============================================================================
FROM tools AS app

WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package with all dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Install additional runtime dependencies
RUN pip install --no-cache-dir \
    fair-esm>=2.0.0 \
    aiohttp \
    tqdm

# Copy docker scripts
COPY docker/ ./docker/
RUN chmod +x docker/entrypoint.sh docker/download_models.py

# =============================================================================
# Stage 4: Final image
# =============================================================================
FROM app AS final

# Set environment variables
ENV RFDIFFUSION_PATH=/opt/RFdiffusion
ENV PROTEINMPNN_PATH=/opt/ProteinMPNN
ENV MODELS_DIR=/models
ENV CACHE_DIR=/cache
ENV TORCH_HOME=/models/esm
ENV PYTHONUNBUFFERED=1

# Create volume mount points
RUN mkdir -p /models /data /cache

# Set working directory
WORKDIR /app

# Expose volumes
VOLUME ["/models", "/data", "/cache"]

# Set entrypoint
ENTRYPOINT ["/app/docker/entrypoint.sh"]

# Default command (starts MCP server)
CMD ["server"]

# Labels
LABEL org.opencontainers.image.title="Protein Design MCP Server"
LABEL org.opencontainers.image.description="MCP server for protein binder design with RFdiffusion, ProteinMPNN, and ESMFold"
LABEL org.opencontainers.image.source="https://github.com/jasonkim8652/protein-design-mcp"
LABEL org.opencontainers.image.licenses="MIT"
