"""
Auto-setup for protein-design-mcp.

Detects your environment, pulls Docker if needed, and writes MCP client
config automatically. One command, zero manual JSON editing.

Usage:
    protein-design-mcp-setup                   # Auto-detect everything
    protein-design-mcp-setup --docker          # Force Docker mode
    protein-design-mcp-setup --local           # Force local (pip) mode
    protein-design-mcp-setup --modal URL       # Force Modal mode
    protein-design-mcp-setup --client claude-desktop  # Specific client
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# Docker image names
DOCKER_IMAGE_GPU = "ghcr.io/jasonkim8652/protein-design-mcp:latest"
DOCKER_IMAGE_LITE = "ghcr.io/jasonkim8652/protein-design-mcp:lite"

# Named Docker volume for model weights (persisted between runs)
DOCKER_VOLUME_MODELS = "protein-design-models"


# ── Environment detection ────────────────────────────────────────────────


def detect_docker() -> bool:
    """Check if Docker daemon is running."""
    if not shutil.which("docker"):
        return False
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def detect_gpu() -> bool:
    """Check if NVIDIA GPU is visible on the host."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def detect_docker_gpu() -> bool:
    """Check if Docker can access GPU (nvidia-container-toolkit)."""
    try:
        r = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "nvidia/cuda:11.8.0-base-ubuntu22.04",
                "nvidia-smi",
                "-L",
            ],
            capture_output=True,
            timeout=60,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# ── Config path resolution ───────────────────────────────────────────────


def _claude_desktop_config_path() -> Path | None:
    system = platform.system()
    if system == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
    if system == "Linux":
        # XDG default
        config_home = os.environ.get(
            "XDG_CONFIG_HOME", str(Path.home() / ".config")
        )
        return Path(config_home) / "Claude" / "claude_desktop_config.json"
    return None


def _claude_code_config_path() -> Path:
    return Path.home() / ".claude" / "mcp.json"


def detect_clients() -> list[tuple[str, Path]]:
    """Return list of (client_name, config_path) for installed MCP clients."""
    found = []
    desktop = _claude_desktop_config_path()
    if desktop and desktop.parent.exists():
        found.append(("claude-desktop", desktop))
    code = _claude_code_config_path()
    if code.parent.exists():
        found.append(("claude-code", code))
    return found


# ── Docker image management ──────────────────────────────────────────────


def pull_docker_image(image: str) -> bool:
    """Pull a Docker image, showing progress."""
    print(f"\n  Pulling {image} ...")
    r = subprocess.run(["docker", "pull", image], timeout=600)
    return r.returncode == 0


def ensure_docker_volume() -> None:
    """Create Docker named volume for model weights if it doesn't exist."""
    subprocess.run(
        ["docker", "volume", "create", DOCKER_VOLUME_MODELS],
        capture_output=True,
    )


# ── Config builders ──────────────────────────────────────────────────────


def build_docker_config(gpu: bool) -> dict:
    """MCP config that runs the server inside Docker."""
    image = DOCKER_IMAGE_GPU
    args = ["run", "-i", "--rm"]

    if gpu:
        args.extend(["--gpus", "all"])
        args.extend(["-e", "DEVICE=auto"])
    else:
        args.extend(["-e", "DEVICE=cpu"])

    args.extend(
        [
            "-e",
            "SKIP_MODEL_DOWNLOAD=true",
            "-v",
            f"{DOCKER_VOLUME_MODELS}:/models",
            image,
        ]
    )

    return {"command": "docker", "args": args}


def build_local_config() -> dict:
    """MCP config that runs the server directly via Python."""
    return {
        "command": sys.executable,
        "args": ["-m", "protein_design_mcp.server"],
    }


def build_modal_config(modal_url: str) -> dict:
    """MCP config that proxies tool calls to a Modal endpoint."""
    return {
        "command": sys.executable,
        "args": ["-m", "protein_design_mcp.modal_proxy"],
        "env": {"MODAL_URL": modal_url},
    }


# ── Config writer ────────────────────────────────────────────────────────


def write_mcp_config(
    server_config: dict, config_path: Path, server_name: str = "protein-design"
) -> None:
    """Merge protein-design server into an existing MCP config file."""
    existing: dict = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"][server_name] = server_config

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(existing, indent=2) + "\n")


# ── Main CLI ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="protein-design-mcp-setup",
        description="Auto-setup protein-design-mcp for your MCP client",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--docker",
        action="store_true",
        help="Force Docker mode",
    )
    mode_group.add_argument(
        "--local",
        action="store_true",
        help="Force local mode (use current Python)",
    )
    mode_group.add_argument(
        "--modal",
        type=str,
        metavar="URL",
        help="Force Modal mode with the given endpoint URL",
    )
    parser.add_argument(
        "--client",
        choices=["claude-desktop", "claude-code"],
        default=None,
        help="MCP client to configure (default: auto-detect)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts",
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  protein-design-mcp setup")
    print("=" * 55)

    # ── 1. Detect environment ────────────────────────────────────────
    has_docker = detect_docker()
    has_gpu = detect_gpu()
    clients = detect_clients()

    print(f"\n  Docker available : {'yes' if has_docker else 'no'}")
    print(f"  NVIDIA GPU       : {'yes' if has_gpu else 'no'}")
    print(f"  MCP clients found: {', '.join(c for c, _ in clients) or 'none'}")

    # ── 2. Choose mode ───────────────────────────────────────────────
    if args.modal:
        mode = "modal"
    elif args.local:
        mode = "local"
    elif args.docker:
        mode = "docker"
    else:
        # Auto: Docker if available, else local
        mode = "docker" if has_docker else "local"

    print(f"  Mode             : {mode}")

    # ── 3. Choose client ─────────────────────────────────────────────
    if args.client:
        client_name = args.client
        if client_name == "claude-desktop":
            config_path = _claude_desktop_config_path()
            if not config_path:
                print("\nError: Cannot find Claude Desktop config path on this OS.")
                sys.exit(1)
        else:
            config_path = _claude_code_config_path()
    elif clients:
        # Prefer Claude Code (developers), fall back to Desktop
        client_name, config_path = clients[-1]
    else:
        # Default to Claude Code
        client_name = "claude-code"
        config_path = _claude_code_config_path()

    print(f"  Client           : {client_name}")
    print(f"  Config path      : {config_path}")

    # ── 4. Execute mode-specific setup ───────────────────────────────
    if mode == "docker":
        if not has_docker:
            print("\nError: Docker is not running.")
            print("  Install: https://docs.docker.com/get-docker/")
            sys.exit(1)

        # Check Docker GPU if host has GPU
        docker_gpu = False
        if has_gpu:
            print("\n  Checking Docker GPU support...")
            docker_gpu = detect_docker_gpu()
            if docker_gpu:
                print("  Docker GPU: yes (all 11 tools available)")
            else:
                print("  Docker GPU: no (9 tools, CPU mode)")
                print("  Install nvidia-container-toolkit for GPU support:")
                print("    https://docs.nvidia.com/datacenter/cloud-native/"
                      "container-toolkit/install-guide.html")

        # Pull image
        if not pull_docker_image(DOCKER_IMAGE_GPU):
            print("\nError: Failed to pull Docker image.")
            sys.exit(1)

        # Create named volume for weights
        ensure_docker_volume()

        server_config = build_docker_config(gpu=docker_gpu)
        tools_count = "11 tools (GPU)" if docker_gpu else "9 tools (CPU)"

    elif mode == "modal":
        server_config = build_modal_config(args.modal)
        tools_count = "11 tools (Modal GPU)"

    else:  # local
        server_config = build_local_config()
        tools_count = "11 tools" if has_gpu else "9 tools (CPU)"

    # ── 5. Show config and confirm ───────────────────────────────────
    print(f"\n  MCP config to write:")
    print(f"  {json.dumps({'protein-design': server_config}, indent=4)}")

    if not args.yes:
        try:
            answer = input(f"\n  Write to {config_path}? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            sys.exit(0)
        if answer in ("n", "no"):
            print("  Aborted.")
            sys.exit(0)

    # ── 6. Write config ──────────────────────────────────────────────
    write_mcp_config(server_config, config_path)

    # ── 7. Done ──────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  Setup complete! ({tools_count})")
    print(f"{'=' * 55}")
    print(f"\n  Config written to: {config_path}")

    if mode == "docker":
        print(f"  Model weights volume: {DOCKER_VOLUME_MODELS}")
        print("  Weights download lazily on first tool call.")

    if client_name == "claude-desktop":
        print("\n  Restart Claude Desktop to load the server.")
    else:
        print("\n  Start a new Claude Code session to use the tools.")

    print(
        "\n  Test: ask your LLM to"
        ' "predict the structure of MKWVTFISLLFLFSSAYS"'
    )


if __name__ == "__main__":
    main()
