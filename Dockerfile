FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 as base
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
        ca-certificates \
        curl \
        git \
        ninja-build \
        python3-dev
RUN curl -fL -q -O https://bootstrap.pypa.io/get-pip.py && \
        python3 get-pip.py && \
        rm get-pip.py
ARG PYTORCH_DOWNLOAD_LINK=https://download.pytorch.org/whl/test/cu101/torch_test.html
RUN pip install \
        torch \
        torchaudio \
        torchvision \
        torchtext \
        -f ${PYTORCH_DOWNLOAD_LINK}

FROM base as fastai
RUN git clone --branch 1.0.61 https://github.com/fastai/fastai.git /fastai
WORKDIR /fastai
RUN tools/run-after-git-clone && pip install ".[dev]" && pip install pytest pytest-runner

FROM base as pyro
RUN git clone --branch master https://github.com/pyro-ppl/pyro.git /pyro
WORKDIR /pyro
RUN pip install ".[dev]"
RUN pip install ".[test]"

FROM base as detectron2
RUN git clone --branch v0.1.3 https://github.com/facebookresearch/detectron2.git /detectron2
RUN apt-get update && apt-get install -y \
        python3-opencv
# install dependencies
RUN pip install tensorboard cython onnx pytest 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

FROM base as transformers
RUN git clone --branch v2.11.0 https://github.com/huggingface/transformers.git /transformers
WORKDIR /transformers
RUN pip install -e ".[testing]"

FROM base as fairseq
RUN git clone --branch master https://github.com/pytorch/fairseq.git /fairseq
WORKDIR /fairseq
RUN pip install pytest pyyaml
RUN pip install -e .
