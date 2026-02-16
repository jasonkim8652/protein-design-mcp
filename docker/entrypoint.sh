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
export TORCH_HOME="${MODELS_DIR}/esm"

# Create directories if they don't exist
mkdir -p "$MODELS_DIR" "$CACHE_DIR"

# Check if models need to be downloaded
SKIP_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-false}"

if [ "$SKIP_DOWNLOAD" != "true" ]; then
    echo "Checking model weights..." >&2

    # Check for RFdiffusion weights
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
echo "  MODELS_DIR: $MODELS_DIR" >&2
echo "  RFDIFFUSION_PATH: $RFDIFFUSION_PATH" >&2
echo "  PROTEINMPNN_PATH: $PROTEINMPNN_PATH" >&2
echo "  CACHE_DIR: $CACHE_DIR" >&2
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
        # Pass through any other command
        exec "$@"
        ;;
esac
