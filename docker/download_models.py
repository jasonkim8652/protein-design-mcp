#!/usr/bin/env python3
"""
Download model weights for RFdiffusion, ProteinMPNN, and ESMFold.

This script is called by the Docker entrypoint to ensure all required
model weights are available before starting the MCP server.
"""

import os
import sys
import hashlib
import urllib.request
from pathlib import Path


# Model weight URLs and checksums
RFDIFFUSION_WEIGHTS = {
    "Complex_base_ckpt.pt": {
        "url": "http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt",
        "size": "1.5GB",
    },
    "Base_ckpt.pt": {
        "url": "http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/Base_ckpt.pt",
        "size": "1.5GB",
    },
}

PROTEINMPNN_WEIGHTS = {
    "v_48_020.pt": {
        "url": "https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_020.pt",
        "size": "150MB",
    },
}


def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Download a file with progress indication."""
    print(f"Downloading {desc or dest_path.name}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 // total_size)
                downloaded = min(total_size, block_num * block_size)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, show_progress)
        print("\n  Done!")
        return True

    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def download_rfdiffusion_weights(models_dir: Path) -> bool:
    """Download RFdiffusion model weights."""
    weights_dir = models_dir / "RFdiffusion" / "models"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Downloading RFdiffusion Weights ===")

    # Only download Complex_base_ckpt.pt by default (for binder design)
    weight_file = "Complex_base_ckpt.pt"
    info = RFDIFFUSION_WEIGHTS[weight_file]
    dest = weights_dir / weight_file

    if dest.exists():
        print(f"  {weight_file} already exists, skipping...")
        return True

    return download_file(info["url"], dest, f"RFdiffusion {weight_file} ({info['size']})")


def download_proteinmpnn_weights(models_dir: Path) -> bool:
    """Download ProteinMPNN model weights."""
    weights_dir = models_dir / "ProteinMPNN" / "vanilla_model_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Downloading ProteinMPNN Weights ===")

    for weight_file, info in PROTEINMPNN_WEIGHTS.items():
        dest = weights_dir / weight_file

        if dest.exists():
            print(f"  {weight_file} already exists, skipping...")
            continue

        if not download_file(info["url"], dest, f"ProteinMPNN {weight_file} ({info['size']})"):
            return False

    return True


def download_esmfold_weights(models_dir: Path) -> bool:
    """
    Trigger ESMFold weight download via fair-esm.

    ESMFold weights are downloaded automatically by the esm library
    when the model is first loaded. We trigger this here to ensure
    weights are cached before the MCP server starts.
    """
    print("\n=== Downloading ESMFold Weights ===")

    # Set cache directory
    cache_dir = models_dir / "esm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_dir)

    # Check if already downloaded
    esmfold_cache = cache_dir / "hub" / "checkpoints"
    if esmfold_cache.exists() and any(esmfold_cache.glob("*.pt")):
        print("  ESMFold weights already cached, skipping...")
        return True

    print("  Loading ESMFold model to trigger weight download...")
    print("  This may take a few minutes (~2GB download)...")

    try:
        import torch
        import esm

        # This triggers the download
        model = esm.pretrained.esmfold_v1()
        del model  # Free memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("  ESMFold weights downloaded successfully!")
        return True

    except ImportError:
        print("  Warning: fair-esm not installed, skipping ESMFold download")
        print("  ESMFold will download weights on first use")
        return True

    except Exception as e:
        print(f"  Warning: Could not pre-download ESMFold weights: {e}")
        print("  ESMFold will download weights on first use")
        return True


def setup_symlinks(models_dir: Path) -> None:
    """Create symlinks for RFdiffusion and ProteinMPNN to find weights."""
    print("\n=== Setting up model paths ===")

    # RFdiffusion expects weights in its models/ directory
    rfd_path = Path(os.environ.get("RFDIFFUSION_PATH", "/opt/RFdiffusion"))
    rfd_models = rfd_path / "models"

    if not rfd_models.exists():
        rfd_models.mkdir(parents=True, exist_ok=True)

    # Link downloaded weights to RFdiffusion models directory
    src_weights = models_dir / "RFdiffusion" / "models"
    if src_weights.exists():
        for weight_file in src_weights.glob("*.pt"):
            dest = rfd_models / weight_file.name
            if not dest.exists():
                print(f"  Linking {weight_file.name} to {dest}")
                dest.symlink_to(weight_file)

    # ProteinMPNN weights
    mpnn_path = Path(os.environ.get("PROTEINMPNN_PATH", "/opt/ProteinMPNN"))
    mpnn_weights = mpnn_path / "vanilla_model_weights"

    if not mpnn_weights.exists():
        mpnn_weights.mkdir(parents=True, exist_ok=True)

    src_mpnn = models_dir / "ProteinMPNN" / "vanilla_model_weights"
    if src_mpnn.exists():
        for weight_file in src_mpnn.glob("*.pt"):
            dest = mpnn_weights / weight_file.name
            if not dest.exists():
                print(f"  Linking {weight_file.name} to {dest}")
                dest.symlink_to(weight_file)


def main():
    """Main entry point for model download."""
    models_dir = Path(os.environ.get("MODELS_DIR", "/models"))

    print("=" * 60)
    print("Protein Design MCP - Model Weight Downloader")
    print("=" * 60)
    print(f"Models directory: {models_dir}")

    # Download each set of weights
    success = True

    if not download_rfdiffusion_weights(models_dir):
        print("Warning: Failed to download RFdiffusion weights")
        success = False

    if not download_proteinmpnn_weights(models_dir):
        print("Warning: Failed to download ProteinMPNN weights")
        success = False

    if not download_esmfold_weights(models_dir):
        print("Warning: Failed to download ESMFold weights")
        success = False

    # Setup symlinks
    setup_symlinks(models_dir)

    print("\n" + "=" * 60)
    if success:
        print("All model weights downloaded successfully!")
    else:
        print("Some weights failed to download. Server may still work.")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
