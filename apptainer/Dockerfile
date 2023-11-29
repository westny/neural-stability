# Use the official Ubuntu 22.04 image as a base
FROM ubuntu:22.04

# Set environment variables
ENV PATH="/opt/conda/bin:${PATH}"
ENV PYTHONNOUSERSITE=1

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    nano \
    wget \
    curl \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda package manager
RUN wget -q -P /tmp https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate && \
    /bin/bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Initialize conda for shell interaction and update conda base environment
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda update -n base -c defaults conda -y && \
    conda install -y python=3.11

# Install PyTorch
RUN conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
RUN conda install -y \
    -c conda-forge \
    lightning \
    wandb \
    scikit-learn \
    tqdm \
    h5py \
    matplotlib \
    pandas \
    openpyxl

# Install using pip
RUN /opt/conda/bin/pip install tfrecord


# Clean up conda packages to reduce the container size
RUN conda clean -a --yes
