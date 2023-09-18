# Use the TensorFlow GPU image as the base image. This image also works with CPU-only setups
FROM tensorflow/tensorflow@sha256:b4676741c491bff3d0f29c38c369281792c7d5c5bfa2b1aa93e5231a8d236323

ENV PYTHONUNBUFFERED=1 
ENV PYTHONDONTWRITEBYTECODE=1 
ENV DOCTR_CACHE_DIR=/app/.cache 

WORKDIR /app

COPY . .

# Install necessary dependencies for video processing and GUI operations
RUN apt-get update \
    && apt-get install --no-install-recommends ffmpeg libsm6 libxext6 -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* 

# Install the current application with TensorFlow extras and modify permissions
RUN pip install --upgrade pip setuptools wheel \
    && pip install -e .[tf] \
    && chmod -R a+w /app
