#!/usr/bin/env bash

# Start Ollama service as fallback LLM
# Provides free local LLM when Nemotron is unavailable

set -e

echo "Starting Ollama service..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Start Ollama service
echo "Starting Ollama daemon..."
ollama serve &
OLLAMA_PID=$!

# Wait for service to be ready
echo "Waiting for Ollama to be ready..."
sleep 5

# Pull a lightweight model for economic reasoning
echo "Pulling lightweight model for economic reasoning..."
ollama pull llama2:7b-chat

echo "Ollama service started successfully!"
echo "API endpoint available at: http://localhost:11434/v1/chat/completions"
echo "Ollama PID: $OLLAMA_PID"

# Keep service running
wait $OLLAMA_PID