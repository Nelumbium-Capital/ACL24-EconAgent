#!/usr/bin/env bash

# Start NVIDIA Nemotron Docker container for local LLM processing
# Requires NVIDIA GPU drivers and Docker configured for GPU access

set -e

export NGC_API_KEY="${NGC_API_KEY:-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}"
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-~/.cache/nim}"

echo "Starting NVIDIA Nemotron Docker container..."
echo "API Key: ${NGC_API_KEY:0:10}..."
echo "Cache directory: $LOCAL_NIM_CACHE"

# Create cache directory
mkdir -p "$LOCAL_NIM_CACHE"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check for NVIDIA GPU support
if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi > /dev/null 2>&1; then
    echo "Warning: NVIDIA GPU support not detected. Container may run slowly on CPU."
    echo "Consider installing NVIDIA Docker runtime for better performance."
fi

# Start Nemotron container
echo "Pulling and starting NVIDIA Nemotron container..."
docker run -it --rm \
    --gpus all \
    --shm-size=16GB \
    -e NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8000:8000 \
    nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2:latest

echo "Nemotron container started successfully!"
echo "API endpoint available at: http://localhost:8000/v1/chat/completions"