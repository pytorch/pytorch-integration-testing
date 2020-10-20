ARG BASE
ARG BASE_IMAGE=ubuntu:14.04
ARG PYTORCH_REF=master

FROM ${BASE_IMAGE} as debian-base
RUN apt-get update && apt-get install -y build-essential ninja-build curl

FROM archlinux as arch-base
RUN pacman -Sy --noconfirm base-devel curl ninja

FROM centos:7 as redhat-base
RUN yum install -y centos-release-scl
RUN yum-config-manager --enable rhel-server-rhscl-7-rpms
RUN yum install -y \
            devtoolset-7 \
            devtoolset-7-gcc-c++ \
            curl \
            ninja-build
ENV PATH /opt/rh/devtoolset-7/root/usr/bin/:${PATH}

FROM alpine:3 as pytorch-clone
ARG PYTORCH_REF=master
RUN apk -U add --no-cache git
RUN git clone https://github.com/pytorch/pytorch.git /pytorch
RUN git -C /pytorch checkout ${PYTORCH_REF} && git -C /pytorch submodule update --init --recursive

FROM ${BASE} as dev
RUN curl -fsSL -o install_conda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN sh install_conda.sh -b -p /opt/conda
ENV PATH /opt/conda/bin:$PATH
COPY --from=pytorch-clone /pytorch /pytorch
RUN conda install -c pytorch-test \
        cpuonly \
        future \
        hypothesis \
        ninja \
        numpy \
        pillow \
        pytorch \
        pyyaml \
        psutil \
        protobuf \
        six
RUN conda install -c conda-forge unittest-xml-reporting
COPY entrypoint.sh /entrypoint.sh
WORKDIR /pytorch/test
ENTRYPOINT '/entrypoint.sh'
