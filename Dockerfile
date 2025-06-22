FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1


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
    && rm -rf /var/lib/apt/lists/*

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
