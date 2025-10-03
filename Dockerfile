# LAPTOP - ARM64
FROM --platform=linux/arm64 ubuntu:22.04 AS cpu

# Python
RUN apt-get update && apt-get install -y \
    build-essential cmake git python3 python3-dev python3-pip \
    libopenblas-dev libblas-dev liblapack-dev ninja-build \
    libffi-dev libssl-dev && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pytorch
RUN pip3 install \
  torch==2.6.0 \
  torchvision==0.13.0 \
  --index-url https://download.pytorch.org/whl/cpu

# PyG
RUN pip3 install torch_geometric
RUN pip3 install \
  torch-scatter==2.1.0 \
  torch-sparse==0.6.17 \
  torch-cluster==1.6.0 \
  torch-spline-conv==1.2.1 \
  -f https://data.pyg.org/whl/torch-2.6.0+cpu.html

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt 
COPY ./src .
RUN chmod +x run.sh

CMD ["bash", "run.sh"]

# PC - AMD64 + 2080Ti
FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn8-runtime AS gpu



WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools \
    && pip install -r requirements.txt \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
COPY ./src .
RUN chmod +x run.sh

CMD ["bash", "run.sh"]