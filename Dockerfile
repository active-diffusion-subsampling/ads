## Dockerfile for GPU environment with Tensorflow, Pytorch, Jax and Keras 3
# base image
FROM ubuntu:22.04

# Do not prompt for input when installing packages
ARG DEBIAN_FRONTEND=noninteractive

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Set pip cache directory
ENV PIP_CACHE_DIR=/tmp/pip_cache

# Install python, pip, git, opencv dependencies, ffmpeg, imagemagick, and ssh keyscan github
RUN apt-get update && \
    apt-get install -y python3 python3-pip git python3-tk \
                       libsm6 libxext6 libxrender-dev libqt5gui5 \
                       ffmpeg imagemagick sudo && \
    python3 -m pip install pip -U && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    ln -s /usr/bin/python3 /usr/bin/python

# Add non-root users
ARG BASE_UID=1000
ARG NUM_USERS=51

# Create users in a loop
RUN for i in $(seq 0 $NUM_USERS); do \
        USER_UID=$((BASE_UID + i)); \
        USERNAME="devcontainer$i"; \
        groupadd --gid $USER_UID $USERNAME && \
        useradd --uid $USER_UID --gid $USER_UID -m --shell /bin/bash $USERNAME && \
        echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
        chmod 0440 /etc/sudoers.d/$USERNAME; \
    done

# install dependencies
RUN --mount=type=cache,target=$PIP_CACHE_DIR pip install --extra-index-url https://pypi.nvidia.com tensorflow[and-cuda]==2.15.0 && \
    pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html jax[cuda12_pip]==0.4.26 && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.2.2+cu121 torchvision && \
    pip install --upgrade keras==3.1.1 && \
    pip install --upgrade keras-cv && \
    pip install wandb albumentations torchmetrics ax-platform;

    
# set working directory and copy files
WORKDIR /ads
COPY . /ads

RUN pip install -r requirements.txt

# Add active sampling to python path
ENV PYTHONPATH="${PYTHONPATH}:/ads/"