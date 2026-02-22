#!/bin/bash
set -e

echo "==============================================" >&2
echo "Protein Design MCP Server" >&2
echo "==============================================" >&2

# Set default paths
export MODELS_DIR="${MODELS_DIR:-/models}"
export RFDIFFUSION_PATH="${RFDIFFUSION_PATH:-/opt/RFdiffusion}"
export PROTEINMPNN_PATH="${PROTEINMPNN_PATH:-/opt/ProteinMPNN}"
export CACHE_DIR="${CACHE_DIR:-/cache}"
export TORCH_HOME="${TORCH_HOME:-${MODELS_DIR}/esm}"
export DEVICE="${DEVICE:-auto}"

# Auto-detect device
if [ "$DEVICE" = "auto" ]; then
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi
export DEVICE

# Create directories if they don't exist
mkdir -p "$MODELS_DIR" "$CACHE_DIR"

# Check if models need to be downloaded
# Default: skip eager download — weights are fetched lazily on first tool call
SKIP_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-true}"

if [ "$SKIP_DOWNLOAD" != "true" ]; then
    echo "Checking model weights..." >&2

    RFD_WEIGHTS="$MODELS_DIR/RFdiffusion/models/Complex_base_ckpt.pt"
    if [ ! -f "$RFD_WEIGHTS" ]; then
        echo "RFdiffusion weights not found, downloading..." >&2
        python /app/docker/download_models.py >&2
    else
        echo "Model weights found, skipping download." >&2
    fi
fi

# Print configuration
echo "" >&2
echo "Configuration:" >&2
echo "  DEVICE: $DEVICE" >&2
echo "  MODELS_DIR: $MODELS_DIR" >&2
echo "  RFDIFFUSION_PATH: $RFDIFFUSION_PATH" >&2
echo "  PROTEINMPNN_PATH: $PROTEINMPNN_PATH" >&2
echo "  CACHE_DIR: $CACHE_DIR" >&2
if [ "$DEVICE" = "cpu" ]; then
    echo "" >&2
    echo "  CPU mode: RFdiffusion disabled, using ESMFold/ESM2/ProteinMPNN/OpenMM" >&2
fi
echo "" >&2

# Handle different commands
case "$1" in
    "server"|"")
        echo "Starting MCP server..." >&2
        echo "==============================================" >&2
        exec python -m protein_design_mcp.server
        ;;
    "download")
        echo "Downloading model weights..." >&2
        exec python /app/docker/download_models.py
        ;;
    "bash"|"sh")
        exec /bin/bash
        ;;
    *)
        exec "$@"
        ;;
esac
