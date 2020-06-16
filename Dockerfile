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
