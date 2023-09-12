FROM nvcr.io/nvidia/tensorflow:22.10.1-tf2-py3

ENV PYTHONUNBUFFERED=1 
ENV PYTHONDONTWRITEBYTECODE=1 
ENV DOCTR_CACHE_DIR=/app/.cache 
ENV PATH=/app/venv/bin:$PATH

WORKDIR /app

COPY . .

RUN apt-get update \
    && apt-get install --no-install-recommends ffmpeg libsm6 libxext6 python3-venv -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* 

RUN python -m venv /app/venv \
    && pip install --upgrade pip setuptools wheel \
    && pip install -e .[tf] \
    && chmod -R a+w /app
