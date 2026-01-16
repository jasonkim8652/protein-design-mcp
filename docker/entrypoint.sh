#!/bin/bash
set -e

echo "=============================================="
echo "Protein Design MCP Server"
echo "=============================================="

# Set default paths
export MODELS_DIR="${MODELS_DIR:-/models}"
export RFDIFFUSION_PATH="${RFDIFFUSION_PATH:-/opt/RFdiffusion}"
export PROTEINMPNN_PATH="${PROTEINMPNN_PATH:-/opt/ProteinMPNN}"
export CACHE_DIR="${CACHE_DIR:-/cache}"
export TORCH_HOME="${MODELS_DIR}/esm"

# Create directories if they don't exist
mkdir -p "$MODELS_DIR" "$CACHE_DIR"

# Check if models need to be downloaded
SKIP_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-false}"

if [ "$SKIP_DOWNLOAD" != "true" ]; then
    echo "Checking model weights..."

    # Check for RFdiffusion weights
    RFD_WEIGHTS="$MODELS_DIR/RFdiffusion/models/Complex_base_ckpt.pt"
    if [ ! -f "$RFD_WEIGHTS" ]; then
        echo "RFdiffusion weights not found, downloading..."
        python /app/docker/download_models.py
    else
        echo "Model weights found, skipping download."
    fi
fi

# Print configuration
echo ""
echo "Configuration:"
echo "  MODELS_DIR: $MODELS_DIR"
echo "  RFDIFFUSION_PATH: $RFDIFFUSION_PATH"
echo "  PROTEINMPNN_PATH: $PROTEINMPNN_PATH"
echo "  CACHE_DIR: $CACHE_DIR"
echo ""

# Handle different commands
case "$1" in
    "server"|"")
        echo "Starting MCP server..."
        echo "=============================================="
        exec python -m protein_design_mcp.server
        ;;
    "download")
        echo "Downloading model weights..."
        exec python /app/docker/download_models.py
        ;;
    "bash"|"sh")
        exec /bin/bash
        ;;
    *)
        # Pass through any other command
        exec "$@"
        ;;
esac
