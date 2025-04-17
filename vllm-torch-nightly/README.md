# VLLM against nightly pytorch build (CUDA)
Dockerfile used to build vllm with nightly pytorch

## EC2 instance Set up recommendation
To run this docker in ec2, we recommand:
- AMI: Deep Learning Oss Nvidia Driver GPU AMI
- Instance type: must be GPU instance that has sm_80+ (tested on g5.24xlarget):
    - p4d / p4de	A100 (40 GB)   sm_80
    - p5	        H100 (80 GB)   sm_90
    - g5	        A10G	       sm_86
- CPU: recommend has vCpu more than 32 (tested on g5.24xlarget), otherwise it will be very slow
- Storage:  512 GiB with mp3


## Build
ssh to your ec2 instance with the pem file from ec2 setup
```
ssh -i ~/secrets/gpu-test-yang.pem ec2-user@${ec2_instance_ip}
``

clone vllm repo
```
mkdir test-vllm
cd test-vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
```
you can use scp to copy the docker file to your ec2 instance, or git clone the repo

to scp from your local machine to ec2 instance, this will override the Dockerfile.nightly in vllm if any:
```
scp -i ~/path/to/your/ec2.pem  path/to/repo/pytorch-integration-testing/vllm-torch-nightly/Dockerfile.nightly ec2-user@${ec2_instance_ip}:/home/ec2-user/test-vllm/vllm
```

Build the docker image with bst-wheel and install (this can take a while)

use default max_jobs(64) and nvcc_threads:

```
BUILDKIT=1 docker build -t test-vllm:vllm-base --build-arg  -f Dockerfile.nightly --target vllm-base  --progress plain .
```
you can set the max-jobs and nvcc_threads based on your ec2 instance hardware.
The max-jobs is recommand to set based on the number of vcpu in your instance but less, you can also set nvcc_threads but recommand value between 2-4,
otherwise, it will cause the ec2 instance crash during the build process. If your ec2 crashes, tune those parameters and monitoring the cpu usage and memory usage during the build process.

pass in max-jobs and nvcc_threads:
```
BUILDKIT=1 docker build -t test-vllm:vllm-base -f Dockerfile.nightly --target vllm-base  --build-arg max-jobs={$MAX_JOBS} --build-arg nvcc_threads={$NVCC_THREADS}  --progress plain .
```

build the docker image with pip install
```
BUILDKIT=1 docker build -t test-vllm:pip-build -f Dockerfile.nightly --target pip-build  --progress plain .
```


## Run vllm

### Prerequisite
you need huggingface account and have the model access.

### steps

#### start the docker container
confirm the docker image exists:
```
docker images
```

start the docker container
```
docker run --gpus all -i test-vllm:vllm-base bash
```

#### test the vllm setup
double check the torch version and vllm:
```
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
python3 -c "import torchvision; print('TorchVision version:', torchvision.__version__)"
python3 -c "import torchaudio; print('TorchAudio version:', torchaudio.__version__)"
python3 -c "import vllm; print(vllm.__version__)
```

download a hugging face model to test the service:
```
huggingface-cli login
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0
```
In the end, you will see a folder named TinyLlama-1.1B-Chat-v1.0

start the vllm server:
```
vllm serve ~/.cache/huggingface/hub/TinyLlama-1.1B-Chat-v1.0/snapshots/${MODEL_VERSION}
```

In another terminal, ssh to yor ec2 instance and access to the docker container:

grab the container id
```
docker ps
```
Access to the docker container
```
docker exec -it ${CONTAINER_ID} bash
```

test the vllm service with curl:
```
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "prompt": "tell me a joke", "max_tokens": 20, "temperature": 0 }'
```
Make sure the response is readable and make sense.
