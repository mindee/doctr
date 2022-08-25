FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/app"

# copy requirements file
COPY requirements.txt  /app/requirements.txt

RUN apt-get update \
    && apt-get install --no-install-recommends ffmpeg libsm6 libxext6 -y \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt \
    && pip cache purge \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip

# copy project
COPY app /app/app
