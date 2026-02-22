# Protein Design MCP Server - All-in-One Docker Image
#
# Includes: RFdiffusion, ProteinMPNN, ESMFold, ColabFold/AlphaFold2,
#            ESM2 Scorer, OpenMM, and the MCP server.
#
# Build: docker build -t protein-design-mcp .
# Run:   docker-compose up
#
# GPU mode (default):  Full pipeline including RFdiffusion
# CPU mode:            Set DEVICE=cpu — runs ESMFold, ESM2, ProteinMPNN, OpenMM
#                      (RFdiffusion disabled, AlphaFold2 uses API backend)

# =============================================================================
# Stage 1: Base image with CUDA and Python
# =============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

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

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# =============================================================================
# Stage 2: Install computational backends
# =============================================================================
FROM base AS tools

WORKDIR /opt

# PyTorch (CUDA 11.8 — also works on CPU)
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# --- RFdiffusion ---
RUN git clone --depth 1 https://github.com/RosettaCommons/RFdiffusion.git \
    && mkdir -p RFdiffusion/models

RUN cd RFdiffusion/env/SE3Transformer \
    && pip install --no-cache-dir -e .

RUN cd /opt/RFdiffusion && pip install --no-cache-dir -e .

# Pin e3nn==0.5.1 (torch 2.0.x compat, before torch.compiler)
# Install dgl==1.1.3 CUDA 11.8 build (PyPI's dgl is CPU-only)
RUN pip install --no-cache-dir \
    hydra-core omegaconf icecream pyrsistent \
    "e3nn==0.5.1" "numpy<2" \
    && pip install --no-cache-dir "dgl==1.1.3" \
    -f https://data.dgl.ai/wheels/cu118/repo.html

# --- ProteinMPNN ---
RUN git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git \
    && mkdir -p ProteinMPNN/vanilla_model_weights

# --- ColabFold / AlphaFold2 ---
RUN pip install --no-cache-dir \
    "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold.git" \
    && mkdir -p /root/.cache/colabfold

RUN ln -sf $(python -c "import colabfold; print(colabfold.__path__[0])")/batch.py \
    /usr/local/bin/colabfold_batch || true

# --- OpenFold v1.0.1 (ESMFold dependency, PYTHONPATH only — no C++ build) ---
RUN git clone --depth 1 --branch v1.0.1 \
    https://github.com/aqlaboratory/openfold.git /opt/openfold \
    && pip install --no-cache-dir biopython dm-tree ml-collections modelcif einops 2>/dev/null || true

# --- OpenMM (energy minimization, AMBER14 force fields) ---
RUN pip install --no-cache-dir openmm

# =============================================================================
# Stage 3: Install MCP server
# =============================================================================
FROM tools AS app

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e ".[dev]"

RUN pip install --no-cache-dir \
    fair-esm>=2.0.0 \
    aiohttp \
    tqdm

COPY docker/ ./docker/
RUN chmod +x docker/entrypoint.sh docker/download_models.py

# =============================================================================
# Stage 4: Final image
# =============================================================================
FROM app AS final

# Tool paths
ENV RFDIFFUSION_PATH=/opt/RFdiffusion
ENV PROTEINMPNN_PATH=/opt/ProteinMPNN
ENV PYTHONPATH="/opt/openfold:${PYTHONPATH}"

# ColabFold
ENV COLABFOLD_PATH=/usr/local/bin/colabfold_batch
ENV COLABFOLD_BACKEND=api

# Storage
ENV MODELS_DIR=/models
ENV CACHE_DIR=/cache
ENV TORCH_HOME=/models/esm
ENV COLABFOLD_WEIGHTS_DIR=/models/colabfold

# Performance
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1
ENV DS_BUILD_OPS=0

# Device mode: "auto" (default), "cuda", or "cpu"
ENV DEVICE=auto

RUN mkdir -p /models /data /cache

WORKDIR /app

VOLUME ["/models", "/data", "/cache"]

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["server"]

LABEL org.opencontainers.image.title="Protein Design MCP Server"
LABEL org.opencontainers.image.description="MCP server for protein design with RFdiffusion, ProteinMPNN, ESMFold, AlphaFold2, ESM2, and OpenMM"
LABEL org.opencontainers.image.source="https://github.com/jasonkim8652/protein-design-mcp"
LABEL org.opencontainers.image.licenses="Apache-2.0"
