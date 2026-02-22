"""
Modal deployment for Protein Design MCP Server.

Deploy to your own Modal account for serverless GPU access.
No GPU on your machine? No problem — Modal provisions A10G GPUs on demand.

Setup (one-time):
    pip install modal
    modal setup

Deploy:
    cd protein-design-mcp
    modal deploy deploy/modal_app.py

Test:
    modal run deploy/modal_app.py

Then connect the local MCP proxy:
    MODAL_URL=https://<your-workspace>--protein-design-tools.modal.run \
        python -m protein_design_mcp.modal_proxy

Cost: ~$1.10/hr (A10G), billed per-second. Containers auto-stop after 5 min idle.
"""

import os
from pathlib import Path

import modal

app = modal.App("protein-design-mcp")

# Persistent volume for model weights — survives container restarts
weights_volume = modal.Volume.from_name(
    "protein-design-weights", create_if_missing=True
)

# ---------------------------------------------------------------------------
# GPU Image (mirrors Dockerfile — all 11 tools available)
# ---------------------------------------------------------------------------
gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "wget", "build-essential")
    # PyTorch (CUDA 11.8)
    .pip_install(
        "torch==2.0.1+cu118",
        "torchvision==0.15.2+cu118",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    # --- RFdiffusion ---
    .run_commands(
        "git clone --depth 1 https://github.com/RosettaCommons/RFdiffusion.git /opt/RFdiffusion",
        "cd /opt/RFdiffusion/env/SE3Transformer && pip install --no-cache-dir -e .",
        "cd /opt/RFdiffusion && pip install --no-cache-dir -e .",
    )
    .pip_install("hydra-core", "omegaconf", "e3nn==0.5.1", "numpy<2")
    .pip_install(
        "dgl==1.1.3",
        find_links="https://data.dgl.ai/wheels/cu118/repo.html",
    )
    # --- ProteinMPNN ---
    .run_commands(
        "git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git /opt/ProteinMPNN",
    )
    # --- OpenFold v1.0.1 (ESMFold dependency) ---
    .run_commands(
        "git clone --depth 1 --branch v1.0.1 "
        "https://github.com/aqlaboratory/openfold.git /opt/openfold",
    )
    .pip_install("biopython", "dm-tree", "ml-collections", "modelcif", "einops")
    # --- ColabFold / AlphaFold2 ---
    .pip_install(
        "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold.git",
    )
    # --- OpenMM ---
    .pip_install("openmm")
    # --- ESM + MCP server deps ---
    .pip_install(
        "fair-esm>=2.0.0",
        "aiohttp",
        "tqdm",
        "aiofiles",
        "pyyaml",
        "mcp>=0.1.0",
    )
    # --- Install protein-design-mcp package ---
    # From local source (requires `pip install -e .` locally first):
    .add_local_python_source("protein_design_mcp")
    # When published to PyPI, replace the line above with:
    # .pip_install("protein-design-mcp")
    .env(
        {
            "RFDIFFUSION_PATH": "/opt/RFdiffusion",
            "PROTEINMPNN_PATH": "/opt/ProteinMPNN",
            "PYTHONPATH": "/opt/openfold",
            "PYTORCH_JIT_USE_NNC_NOT_NVFUSER": "1",
            "DS_BUILD_OPS": "0",
            "DEVICE": "cuda",
            "MODELS_DIR": "/models",
            "TORCH_HOME": "/models/esm",
            "COLABFOLD_BACKEND": "api",
        }
    )
)


# ---------------------------------------------------------------------------
# Helper: materialize inlined file contents into temp files
# ---------------------------------------------------------------------------
def _materialize_files(arguments: dict) -> dict:
    """Write inlined file contents (from proxy) to local temp files.

    The MCP proxy reads local PDB files and sends their content as
    ``_file_<argname>`` keys. This function writes them to temp files
    so the tool handlers can use regular file paths.
    """
    import tempfile

    result = dict(arguments)
    for key in list(result.keys()):
        if not key.startswith("_file_"):
            continue
        real_key = key[6:]  # "_file_target_pdb" -> "target_pdb"
        content = result.pop(key)
        # Write to a temp file
        suffix = ".pdb" if "pdb" in real_key else ".txt"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=f"{real_key}_")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        result[real_key] = path
    return result


# ---------------------------------------------------------------------------
# Web endpoint — single entry point for all tool calls
# ---------------------------------------------------------------------------
@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={"/models": weights_volume},
    timeout=1800,
    container_idle_timeout=300,
)
@modal.web_endpoint(method="POST", label="protein-design-tools", docs=True)
async def call_tool(item: dict) -> dict:
    """Execute a protein design tool on GPU.

    Request body: ``{"name": "predict_structure", "arguments": {...}}``
    """
    name = item.get("name")
    arguments = item.get("arguments", {})

    if not name:
        return {"error": "Missing 'name' in request body"}

    # Materialize any inlined file contents from the proxy
    arguments = _materialize_files(arguments)

    # Ensure model weight directories exist
    os.makedirs("/models/esm/hub/checkpoints", exist_ok=True)
    os.makedirs("/models/RFdiffusion/models", exist_ok=True)
    os.makedirs("/models/ProteinMPNN/vanilla_model_weights", exist_ok=True)

    # Import handlers (lazy — avoids import overhead on cold start)
    from protein_design_mcp.server import (
        handle_analyze_interface,
        handle_design_binder,
        handle_energy_minimize,
        handle_generate_backbone,
        handle_get_design_status,
        handle_optimize_sequence,
        handle_predict_complex,
        handle_predict_structure,
        handle_score_stability,
        handle_suggest_hotspots,
        handle_validate_design,
    )

    handlers = {
        "design_binder": handle_design_binder,
        "analyze_interface": handle_analyze_interface,
        "validate_design": handle_validate_design,
        "optimize_sequence": handle_optimize_sequence,
        "suggest_hotspots": handle_suggest_hotspots,
        "get_design_status": handle_get_design_status,
        "predict_complex": handle_predict_complex,
        "predict_structure": handle_predict_structure,
        "score_stability": handle_score_stability,
        "energy_minimize": handle_energy_minimize,
        "generate_backbone": handle_generate_backbone,
    }

    handler = handlers.get(name)
    if not handler:
        return {"error": f"Unknown tool: {name}", "available": list(handlers.keys())}

    try:
        result = await handler(arguments)
        # Persist any newly downloaded model weights to the volume
        weights_volume.commit()
        return result
    except Exception as e:
        return {"error": str(e), "tool": name}


# ---------------------------------------------------------------------------
# CLI test entrypoint: modal run deploy/modal_app.py
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def test():
    """Quick smoke test — score stability of a short peptide."""
    import json

    print("Testing protein-design-mcp on Modal...")
    result = call_tool.remote(
        {"name": "score_stability", "arguments": {"sequence": "MKTLYVGDAKEF" * 5}}
    )
    print(json.dumps(result, indent=2))
    print("\nDeployment is working! Your endpoint URL:")
    print("  https://<your-workspace>--protein-design-tools.modal.run")
