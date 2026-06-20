"""
Modal deployment for Pulse Point detection server.

Runs LocateAnything-3B on an A10G GPU.
Model weights are cached in a persistent volume so cold starts are fast after the first run.

Deploy:
  modal deploy server/modal_app.py

The deployed URL will be printed after deploy — add it to Vercel as VITE_SERVER_URL.
"""

import sys
from pathlib import Path

import modal

SERVER_DIR = Path(__file__).parent

app = modal.App("pulse-point-server")

# Persistent volume — HuggingFace weights download once, cached forever
hf_cache = modal.Volume.from_name("pulse-point-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        "fastapi==0.115.0",
        "uvicorn==0.30.6",
        "python-multipart==0.0.9",
        "torch==2.4.1",
        "torchvision==0.19.1",
        "Pillow==10.4.0",
        "transformers>=4.57.1",
        "accelerate>=0.34.0",
        "qwen_vl_utils>=0.0.8",
    )
    .add_local_dir(SERVER_DIR, remote_path="/app")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    scaledown_window=300,          # stay warm for 5 min after last request
    allow_concurrent_inputs=4,
    volumes={"/root/.cache/huggingface": hf_cache},
)
@modal.asgi_app()
def server():
    sys.path.insert(0, "/app")
    from main import app as fastapi_app
    return fastapi_app
