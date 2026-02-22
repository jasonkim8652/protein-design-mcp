#!/usr/bin/env bash
# Protein Design MCP Server — Quick Setup
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/jasonkim8652/protein-design-mcp/main/setup.sh | bash
#
#   # Or with options:
#   curl -sSL ... | bash -s -- --cpu        # CPU-only mode
#   curl -sSL ... | bash -s -- --gpu 0      # Specific GPU
#   curl -sSL ... | bash -s -- --dir /path  # Custom install directory

set -euo pipefail

# Defaults
INSTALL_DIR="$HOME/protein-design-mcp"
MODE="gpu"
GPU_ID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)       MODE="cpu"; shift ;;
        --gpu)       GPU_ID="$2"; shift 2 ;;
        --dir)       INSTALL_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: setup.sh [--cpu] [--gpu ID] [--dir PATH]"
            echo ""
            echo "Options:"
            echo "  --cpu        CPU-only mode (no NVIDIA GPU required)"
            echo "  --gpu ID     Use specific GPU (default: all available)"
            echo "  --dir PATH   Installation directory (default: ~/protein-design-mcp)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "  Protein Design MCP Server — Setup"
echo "=============================================="
echo ""
echo "  Install dir:  $INSTALL_DIR"
echo "  Mode:         $MODE"
[ -n "$GPU_ID" ] && echo "  GPU:          $GPU_ID"
echo ""

# Check Docker
if ! command -v docker &>/dev/null; then
    echo "Error: Docker is not installed."
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check GPU (if GPU mode)
if [ "$MODE" = "gpu" ]; then
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        echo "Warning: NVIDIA Container Toolkit not detected."
        echo "GPU mode may not work. Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        echo ""
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo ""
        [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
    fi
fi

# Create directory
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Download docker-compose.yml
echo "Downloading docker-compose.yml..."
curl -sSL -o docker-compose.yml \
    "https://raw.githubusercontent.com/jasonkim8652/protein-design-mcp/main/docker-compose.yml"

# Create data directories
mkdir -p models data cache

# Pull image
echo ""
echo "Pulling Docker image (this may take a few minutes)..."
docker pull ghcr.io/jasonkim8652/protein-design-mcp:latest

# Download model weights
echo ""
echo "Downloading model weights (~5GB, first time only)..."
docker compose --profile setup run --rm download-models

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""

if [ "$MODE" = "cpu" ]; then
    echo "Start the server (CPU mode):"
    echo "  cd $INSTALL_DIR"
    echo "  docker compose --profile cpu up"
else
    echo "Start the server (GPU mode):"
    echo "  cd $INSTALL_DIR"
    echo "  docker compose up"
fi

echo ""
echo "Configure Claude Desktop:"
echo '  Add to claude_desktop_config.json:'
echo '  {'
echo '    "mcpServers": {'
echo '      "protein-design": {'
echo '        "command": "docker",'
echo '        "args": ["compose", "run", "--rm", "-T", "protein-design", "server"],'
echo "        \"cwd\": \"$INSTALL_DIR\""
echo '      }'
echo '    }'
echo '  }'
echo ""
