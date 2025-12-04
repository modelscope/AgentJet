# Installation Guide

This document provides a step-by-step guide to installing AgentScope Tuner.

```{tip}
AgentScope Tuner is under active development and iteration. We recommend installing from source to get the latest futures and bug fixes.
```

## Prerequisites

- Python 3.10 to 3.12
- CUDA 12.8 or higher


## Install from Source

### Step 1: Clone the Repository

First, clone the AgentScope Tuner repository from GitHub and navigate into the project directory:

```bash
git clone https://github.com/agentscope-ai/agentscope-tuner.git
cd agentscope-tuner
```


### Step 2: Install Dependencies

#### For Users New to LLM Training

If you are new to LLM training and environment setup, we suggest using Docker to simplify the installation process.

Before proceeding, ensure you have Docker installed on your system and that it is configured to use your GPU.

You can download our Docker Image from Docker Hub or build it locally.

##### Option 1: Pull the pre-built Docker Image

You can pull the pre-built Docker image using the following command:

```bash
docker pull ghcr.io/agentscope-ai/agentscope-tuner:latest
```

##### Option 2: Build the Docker Image Locally

We provide a Dockerfile that sets up the necessary environment for AgentScope Tuner.

You can build the Docker image using the following command:

```bash
# make sure you are in the root directory of agentscope-tuner
docker build -f scripts/docker/dockerfile -t agentscope-tuner:latest .
```

##### Running the Docker Container

Once the image is pulled or built, you can run a container with the following command. Make sure to replace `/path/to/your/checkpoint/and/data` with the actual path where your model checkpoints and data are stored.
This command mounts your current working directory (the root directory of agentscope-tuner) to `/workspace` and your data directory to `/data` inside the container.

```bash
docker run -it \
  --gpus all \
  --shm-size="64g" \
  --rm \
  -v $PWD:/workspace \
  -v /path/to/your/checkpoint/and/data:/data \
  agentscope-tuner:latest
```

#### For Users Familiar with LLM Training

We recommend using `uv` to manage your Python environment.

If you don't have `uv` installed, you can install it via pip:

```bash
pip install uv
uv venv --python=3.10.16
```

Then, activate the virtual environment and install the required dependencies:

```bash
source .venv/bin/activate
uv pip install -e.[dev]
uv pip install flash-attn --no-build-isolation --no-cache-dir
```

## Install from PyPI (Not Recommended)

You can also install AgentScope Tuner directly from PyPI using pip. This method is suitable for users who prefer a quick installation without needing the latest development features.

```bash
pip install agentscope-tuner
# install flash-attn after agentscope-tuner, because flash-attn relies on some packages installed by agentscope-tuner
pip install flash-attn --no-build-isolation --no-cache-dir
```
