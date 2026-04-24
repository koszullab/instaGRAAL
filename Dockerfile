# syntax=docker/dockerfile:1
# This image requires nvidia-docker2 and must be run with --gpus all

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

LABEL Name=instagraal Version=0.1.6

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y --no-install-recommends \
  gcc \
  g++ \
  python3 \
  python3-dev \
  libjpeg-dev \
  zlib1g-dev \
  hdf5-tools \
  curl \
  ca-certificates \
  && apt-get clean autoclean \
  && apt-get autoremove -y \
  && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install instagraal and all its dependencies via uv
COPY . /tmp/instaGRAAL
RUN uv pip install --system /tmp/instaGRAAL && rm -rf /tmp/instaGRAAL

WORKDIR /work
