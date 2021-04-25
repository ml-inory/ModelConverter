FROM nvidia/cuda:10.0-runtime-ubuntu18.04
ARG git_user
ARG git_pass

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

ENV NCCL_VERSION 2.6.4

RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list &&\
    sed -i "s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list &&\
    echo nameserver 114.114.114.114 >> /etc/resolv.conf &&\
    apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-$CUDA_PKG_VERSION \
    cuda-npp-$CUDA_PKG_VERSION \
    cuda-nvtx-$CUDA_PKG_VERSION \
    libnccl2=$NCCL_VERSION-1+cuda10.0 \
    git \
    && apt-mark hold libnccl2 &&\
    git clone https://github.com/ml-inory/ModelConverter &&\
    cd modelconverter &&\
    bash build.sh

