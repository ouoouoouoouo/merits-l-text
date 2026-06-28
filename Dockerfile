# MERITS-L text-stage Docker image — works on RunPod / any CUDA 12.1 host.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv \
        git wget curl ffmpeg libsndfile1 ca-certificates \
        && rm -rf /var/lib/apt/lists/* \
        && ln -sf /usr/bin/python3.10 /usr/bin/python \
        && ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /workspace

# Install PyTorch with CUDA 12.1 wheels first (so the rest can resolve correctly).
RUN pip install --upgrade pip \
    && pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# Project files are mounted at runtime; we don't COPY them into the image so
# `git push` from the host stays the source of truth.

CMD ["bash"]
