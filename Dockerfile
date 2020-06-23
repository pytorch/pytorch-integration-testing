FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 as base
RUN apt-get update && apt-get install -y build-essential curl git
RUN curl -fsSL -o install_conda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN sh install_conda.sh -b -p /opt/conda
ENV PATH /opt/conda/bin:$PATH
RUN conda install -c pytorch-test \
        cudatoolkit=10.1 \
        pytorch \
        torchvision \
        torchaudio \
        torchtext

FROM base as fastai
RUN git clone --branch 1.0.60 https://github.com/fastai/fastai.git /fastai
WORKDIR /fastai
RUN tools/run-after-git-clone && pip install ".[dev]" && pip install pytest pytest-runner

FROM base as pyro
RUN git clone --branch 1.3.1 https://github.com/pyro-ppl/pyro.git /pyro
WORKDIR /pyro
RUN pip install ".[dev]"
RUN pip install ".[test]"

# Shamelessly stolen from https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile-circleci
FROM nvidia/cuda:10.1-cudnn7-devel as detectron2
# This dockerfile only aims to provide an environment for unittest on CircleCI

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build && \
  rm -rf /var/lib/apt/lists/*

RUN wget -q https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# install dependencies
RUN pip install tensorboard cython onnx pytest
# I have no idea why but this fails when packages are installed from conda
RUN pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN git clone --branch v0.1.3 https://github.com/facebookresearch/detectron2.git /detectron2
