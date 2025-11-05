FROM nvidia/cuda:11.4.3-base-ubuntu20.04

ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /project

# Install basic tools and Python
RUN apt-get update && apt-get install -y \
    python3-pip \
    apt-utils \
    vim \
    git \
    iputils-ping \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Alias python to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python packages
RUN pip install \
    pyyaml \
    wandb \
    matplotlib \
    numpy \
    pandas \
    rtpt \
    scipy \
    scikit-learn \
    seaborn \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html \
    tensorboard \
    tensorflow \
    absl-py \
    Pillow \
    ipython \
    tqdm \ 
    ipykernel \
    notebook jupyterlab
