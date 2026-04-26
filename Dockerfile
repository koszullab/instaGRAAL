# syntax=docker/dockerfile:1
# This image requires nvidia-docker2 and must be run with --gpus all

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

LABEL Name=instagraal Version=0.1.6

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y --no-install-recommends \
  gcc \
  g++ \
  python3-full \
  python3-dev \
  python3-pip \
  libjpeg-dev \
  zlib1g-dev \
  hdf5-tools \
  curl \
  ca-certificates \
  && apt-get clean autoclean \
  && apt-get autoremove -y \
  && rm -rf /var/lib/apt/lists/*

COPY . /tmp/instaGRAAL

RUN pip install --break-system-packages /tmp/instaGRAAL[dev] \
  && rm -rf /tmp/instaGRAAL

WORKDIR /work
