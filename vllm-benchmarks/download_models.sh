#!/bin/bash

set -e

echo "Installing huggingface_hub and hf_xet..."
pip install huggingface_hub==1.3.5 hf_xet

echo "Setting HF_HOME environment variable..."
export HF_HOME=/mnt/hf_cache

# List of HuggingFace models to download
MODELS=(
    "deepseek-ai/DeepSeek-R1"
    "deepseek-ai/DeepSeek-V3.2"
    "facebook/opt-125m"
    "google/gemma-3-27b-it"
    "google/gemma-3-4b-it"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "openai/gpt-oss-120b"
    "openai/gpt-oss-20b"
    "pytorch/gemma-3-12b-it-FP8"
    "pytorch/gemma-3-12b-it-INT4"
    "pytorch/gemma-3-27b-it-FP8"
    "pytorch/gemma-3-27b-it-INT4"
    "Qwen/Qwen3-30B-A3B"
    "Qwen/Qwen3-8B"
    "turboderp/Qwama-0.5B-Instruct"
)

echo "Starting model downloads..."
echo "Total models to download: ${#MODELS[@]}"

# Download each model
for model in "${MODELS[@]}"; do
    echo "========================================"
    echo "Downloading model: $model"
    echo "========================================"
    TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 hf download "$model"

    if [ $? -eq 0 ]; then
        echo "Successfully downloaded: $model"
    else
        echo "Failed to download: $model"
    fi
done

echo "========================================"
echo "All downloads completed!"
echo "========================================"
