# VLLM against nightly pytorch build (CUDA)
Dockerfile used to build vllm with nightly pytorch

## EC2 Instance Setup Recommendations

To run this Docker container in EC2, we recommend:
- **AMI**: Deep Learning OSS NVIDIA Driver GPU AMI
- **Instance Type**: Must be a GPU instance with SM_80 or higher architecture (tested on g6.16xlarge):
    - p4d/p4de: A100 - SM_80
    - g6.16xlarge: L4  - SM_89
    - p5: H100  - SM_90
- **CPU**: Recommend 32+ vCPUs (tested on g6.16xlarge), otherwise build process will be very slow
- **Storage**: 512 GiB with mp3 ( can set larger)

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

3. Copy the Dockerfile, nightly_torch_test.txt and vllm_test.sh to your EC2 instance. You can either use `scp` or clone this repository. To use `scp` from your local machine:
```bash
cd path/to/repo/pytorch-integration-testing/vllm-torch-nightly
scp -i Dockerfile.nightly_torch ec2-user@${ec2_instance_ip}:/home/ec2-user/test-vllm/vllm/docker && \
scp -i nightly_torch_test.txt ec2-user@${ec2_instance_ip}:/home/ec2-user/test-vllm/vllm/requirements && \
scp -i vllm_test.sh ec2-user@${ec2_instance_ip}:/home/ec2-user/test-vllm
```

5. Build and install the Docker image:

Using default max_jobs (64) and nvcc_threads:
```bash
BUILDKIT=1 docker build -t test-vllm:vllm-base -f docker/Dockerfile.nightly_torch --target vllm-base --progress plain .
```

Alternatively, specify max_jobs and nvcc_threads based on your instance hardware:
```bash
BUILDKIT=1 docker build -t test-vllm:vllm-base -f Dockerfile.nightly_torch --target vllm-base --build-arg max-jobs=${MAX_JOBS} --build-arg nvcc_threads=${NVCC_THREADS} --progress plain .
```

If you changing dockerfile frequently during the development, you can put it in test-vllm/,
```bash
  scp -i Dockerfile.nightly_torch ec2-user@${ec2_instance_ip}:/home/ec2-user/test-vllm
```
then
```bash
  cd vllm/
  BUILDKIT=1 docker build -t test-vllm:vllm-base -f ../Dockerfile.nightly_torch --target vllm-base --progress plain .
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
docker run --gpus all  -e HF_TOKEN=$HF_TOKEN -it test-vllm:vllm-base bash
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
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0
```
The model will be downloaded to `~/.cache/huggingface/hub/`.

4. Start the VLLM server:
```bash
export VLLM_USE_V1=1
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
  -d '{ "prompt": "what is you ?", "max_tokens": 50, "temperature": 0 }'
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


# Test the vllm with nightly pytorch build
## 1. Build the Test Docker Image
### ✅ Build with Latest Nightly (no version pinning)

```bash
docker build -t nightly_torch1 \
  --target test \
  --build-arg CUDA_VERSION='12.8.0' \
  --build-arg max_jobs=32 \
  --build-arg nvcc_threads=2 \
  --build-arg torch_cuda_arch_list='8.9' \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg vllm_fa_cmake_gpu_arches='89-real' \
  -f ../Dockerfile.nightly_torch .
```
## Build with Pinned Nightly Versions (recommended for consistent CI)
### Use this to target a specific PyTorch nightly build — e.g., dev20250411+cu128 for L4 (SM 8.9):

with pinned torch version (L4 with cuda ard 89):
```
 docker build -t nightly_torch1 \
  --target test \
  --build-arg CUDA_VERSION='12.8.0' \
  --build-arg max_jobs=32 \
  --build-arg nvcc_threads=2 \
  --build-arg torch_cuda_arch_list='8.9' \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg vllm_fa_cmake_gpu_arches='89-real' \
  --build-arg PINNED_TORCH_VERSION="torch==2.8.0.dev20250411+cu128 torchaudio==2.6.0.dev20250411+cu128 torchvision==0.22.0.dev20250411+cu128" \
  -f ../Dockerfile.nightly_torch .
```

## access to the container
```
docker run -it --gpus all nightly_torch1 bash
```

## check the version of pytorch, exformers, vllm and flashinfer
```
pip freeze | grep -E 'torch|xformers|vllm|flashinfer'
```

## run vllm unit tests

```bash
cd tests && pytest -v -s  entrypoints/llm/test_lazy_outlines.py
```

# Additional packages
Notice if no FlashInfer installed, the test will faill back to PyTorch-native implementation of top-p & top-k sampling. This is slower than FlashInfer, but stable.
