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

FROM base as pyro
# Needed to build pillow from source
RUN apt-get install -y \
        libfreetype6-dev \
        libfribidi-dev \
        libharfbuzz-dev \
        libjpeg8-dev \
        liblcms2-dev \
        libopenjp2-7-dev \
        libtiff5-dev \
        libwebp-dev \
        libxcb1-dev \
        python3-tk \
        tcl8.6-dev \
        tk8.6-dev \
        zlib1g-dev
RUN git clone --branch master https://github.com/pyro-ppl/pyro.git /pyro
WORKDIR /pyro
RUN pip install ".[dev]"
RUN pip install ".[test]"

FROM base as detectron2
RUN git clone --branch v0.2.1 https://github.com/facebookresearch/detectron2.git /detectron2
RUN apt-get update && apt-get install -y \
        python3-opencv
# install dependencies
RUN pip install tensorboard cython onnx pytest 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

FROM base as transformers
RUN git clone --branch v3.3.1 https://github.com/huggingface/transformers.git /transformers
WORKDIR /transformers
RUN pip install -e ".[testing]"

FROM base as fairseq
RUN git clone --branch master https://github.com/pytorch/fairseq.git /fairseq
WORKDIR /fairseq
RUN pip install pytest pyyaml
RUN pip install -e .

FROM base as pytorch-lightning
RUN git clone --branch 1.0.2 https://github.com/PyTorchLightning/pytorch-lightning /pytorch-lightning
WORKDIR /pytorch-lightning
RUN pip install --requirement ./requirements/base.txt --upgrade && pip install cmake
ENV LC_ALL=C.UTF-8
RUN sh -c 'apt-get install -y libblas-dev liblapack-dev gfortran && HOROVOD_BUILD_ARCH_FLAGS="-mfma" pip install -e ".[dev]" --quiet --upgrade'
