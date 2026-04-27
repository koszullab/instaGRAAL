# syntax=docker/dockerfile:1
# This image requires nvidia-docker2 and must be run with --gpus all

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

LABEL Name=instagraal Version=0.2.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y --no-install-recommends \
  gcc \
  g++ \
  software-properties-common \
  libjpeg-dev \
  zlib1g-dev \
  hdf5-tools \
  curl \
  ca-certificates \
  && . /etc/os-release \
  && if [ "$VERSION_ID" = "22.04" ]; then \
    add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update; \
  fi \
  && apt-get install -y --no-install-recommends \
  python3.12 \
  python3.12-dev \
  python3.12-venv \
  && apt-get clean autoclean \
  && apt-get autoremove -y \
  && rm -rf /var/lib/apt/lists/* \
  && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 - --break-system-packages

COPY . /tmp/instaGRAAL

RUN python3.12 -m pip install --break-system-packages /tmp/instaGRAAL[dev] \
  && rm -rf /tmp/instaGRAAL

WORKDIR /work
