# VLLM against nightly pytorch build (CUDA)
Dockerfile used to build vllm with nightly pytorch

## EC2 Instance Setup Recommendations

To run this Docker container in EC2, we recommend:

- **AMI**: Deep Learning OSS NVIDIA Driver GPU AMI
- **Instance Type**: Must be a GPU instance with SM_80 or higher architecture (tested on g5.24xlarge):
    - p4d/p4de: A100 - SM_80
    - g5: A10G  - SM_86
    - p5: H100  - SM_90
- **CPU**: Recommend 32+ vCPUs (tested on g5.24xlarge), otherwise build process will be very slow
- **Storage**: 512 GiB with mp3

## Build Instructions

1. SSH to your EC2 instance using your private key:
```bash
ssh -i ~/secrets/gpu-test-yang.pem ec2-user@${ec2_instance_ip}
```

2. Clone the VLLM repository:
```bash
mkdir test-vllm
cd test-vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

3. Copy the Dockerfile to your EC2 instance. You can either use `scp` or clone this repository. To use `scp` from your local machine:
```bash
scp -i ~/path/to/your/ec2.pem path/to/repo/pytorch-integration-testing/vllm-torch-nightly/Dockerfile.nightly ec2-user@${ec2_instance_ip}:/home/ec2-user/test-vllm/vllm
```

4. Build and install the Docker image:

Using default max_jobs (64) and nvcc_threads:
```bash
BUILDKIT=1 docker build -t test-vllm:vllm-base -f Dockerfile.nightly --target vllm-base --progress plain .
```

Alternatively, specify max_jobs and nvcc_threads based on your instance hardware:
```bash
BUILDKIT=1 docker build -t test-vllm:vllm-base -f Dockerfile.nightly --target vllm-base --build-arg max-jobs=${MAX_JOBS} --build-arg nvcc_threads=${NVCC_THREADS} --progress plain .
```

Note: Set max_jobs based on your instance's vCPU count (but lower to prevent crashes). Recommended nvcc_threads value is between 2-4. Monitor CPU and memory usage during the build process to avoid instance crashes. If you encounter issues, such as ec2 instance crashes, try to tune these values down.

## Running the VLLM Docker Container

### Prerequisites
- HuggingFace account with model access permissions

### Steps

1. Start the Docker container:
```bash
# Verify the image exists
docker images

# Start the container
docker run --gpus all -i test-vllm:vllm-base bash
```

2. Verify the installation:
```bash
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
python3 -c "import torchvision; print('TorchVision version:', torchvision.__version__)"
python3 -c "import torchaudio; print('TorchAudio version:', torchaudio.__version__)"
python3 -c "import vllm; print(vllm.__version__)"
```

3. Download a HuggingFace model:
```bash
huggingface-cli login
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0
```
The model will be downloaded to `~/.cache/huggingface/hub/`.

4. Start the VLLM server:
```bash
vllm serve ~/.cache/huggingface/hub/TinyLlama-1.1B-Chat-v1.0/snapshots/${MODEL_VERSION}
```

Upon successful start, you should see:
```
INFO:     Started server process [46]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

5. Test the service:

In a new terminal, access the Docker container:
```bash
# Get the container ID
docker ps

# Access the container
docker exec -it ${CONTAINER_ID} bash
```

Send a test request:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{ "prompt": "tell me a joke", "max_tokens": 20, "temperature": 0 }'
```

You should receive a response similar to:
```json
{
  "id": "cg",
  "object": "text_completion",
  "created": 1744908741,
  "model": "/root/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/",
  "choices": [{
    "index": 0,
    "text": "?\n\nJOKE: (laughs) \"What do you call a pig",
    "logprobs": null,
    "finish_reason": "length",
    "stop_reason": null,
    "prompt_logprobs": null
  }],
  "usage": {
    "prompt_tokens": 6,
    "total_tokens": 26,
    "completion_tokens": 20,
    "prompt_tokens_details": null
  }
}
```

# Additional packages
Notice if no FlashInfer installed, the test will faill back to PyTorch-native implementation of top-p & top-k sampling. This is slower than FlashInfer, but stable.
