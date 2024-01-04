FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ARG SYSTEM=gpu

# Enroll NVIDIA GPG public key and install CUDA
RUN if [ "$SYSTEM" = "gpu" ]; then \
    apt-get update && \
    apt-get install -y gnupg ca-certificates wget && \
    # - Install Nvidia repo keys
    # - See: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#network-repo-installation-for-ubuntu
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
    cuda-command-line-tools-11-8 \
    cuda-cudart-dev-11-8 \
    cuda-nvcc-11-8 \
    cuda-cupti-11-8 \
    cuda-nvprune-11-8 \
    cuda-libraries-11-8 \
    cuda-nvrtc-11-8 \
    libcufft-11-8 \
    libcurand-11-8 \
    libcusolver-11-8 \
    libcusparse-11-8 \
    libcublas-11-8 \
    # - CuDNN: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#ubuntu-network-installation
    libcudnn8=8.6.0.163-1+cuda11.8 \
    libnvinfer-plugin8=8.6.1.6-1+cuda11.8 \
    libnvinfer8=8.6.1.6-1+cuda11.8; \
fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    # - Other packages
    build-essential \
    pkg-config \
    curl \
    wget \
    software-properties-common \
    unzip \
    git \
    # - Packages to build Python
    tar make gcc zlib1g-dev libffi-dev libssl-dev liblzma-dev libbz2-dev libsqlite3-dev \
    # - Packages for docTR
    libgl1-mesa-dev libsm6 libxext6 libxrender-dev libpangocairo-1.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
fi

# Install Python
ARG PYTHON_VERSION=3.10.13

RUN wget http://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -zxf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    mkdir /opt/python/ && \
    ./configure --prefix=/opt/python && \
    make && \
    make install && \
    cd .. && \
    rm Python-$PYTHON_VERSION.tgz && \
    rm -r Python-$PYTHON_VERSION

ENV PATH=/opt/python/bin:$PATH

# Install docTR
ARG FRAMEWORK=tf
ARG DOCTR_REPO='mindee/doctr'
ARG DOCTR_VERSION=main
RUN pip3 install -U pip setuptools wheel && \
    pip3 install "python-doctr[$FRAMEWORK]@git+https://github.com/$DOCTR_REPO.git@$DOCTR_VERSION"
