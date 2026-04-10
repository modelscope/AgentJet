---
name: docker-install-agentjet-swarm-server
description: Install and run AgentJet Swarm Server via Docker with GPU support
license: Complete terms in LICENSE.txt
---

>
> when the user only need to run agentjet client, and do not have to run models locally (e.g. user in their laptop), ONLY install AgentJet basic requirements is enough (pip install -e .).
> see `install-agentjet-client` skill
>


# AgentJet Docker Installation Skill

This skill guides you through installing and running the AgentJet Swarm Server in a Docker container with GPU support.

## Prerequisites Checklist

Before proceeding, verify:
1. **GPU Available**: System has NVIDIA GPU(s)
2. **Docker Installed**: Docker is available
3. **NVIDIA Container Toolkit**: nvidia-docker2 or nvidia-container-toolkit is installed

---

## Step 1: Check GPU

```bash
nvidia-smi
```

If this fails, the system may not have NVIDIA drivers or GPU hardware.

---

## Step 2: Install Docker

```bash
sudo apt update
sudo apt install docker docker.io curl
```

---

## Step 3: Install NVIDIA Container Toolkit

```bash
# Install Docker with convenience script
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker daemon
sudo systemctl restart docker
```

---

## Step 4: Configure Docker Mirror (Optional - For Slow Image Pulls)

If pulling Docker images is too slow, configure a mirror registry:

### Option A: Configure via daemon.json

```bash
# Create or edit Docker daemon config
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<EOF
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me"
  ]
}
EOF

# Restart Docker
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### Option B: Pull via Mirror URL Directly

For `ghcr.io` images, use a mirror prefix:

```bash
# Original (may be slow)
docker pull ghcr.io/modelscope/agentjet:main

# Using mirror (faster in China)
docker pull ghcr.modelscope.cn/modelscope/agentjet:main

# Or use dockerhub mirror
docker pull docker.1ms.run/modelscope/agentjet:main
```

### Popular Mirror Registries

| Mirror | Region | Note |
|--------|--------|------|
| `docker.1ms.run` | China | General Docker Hub mirror |
| `docker.xuanyuan.me` | China | Alternative mirror |
| `ghcr.modelscope.cn` | China | GitHub Container Registry mirror |
| `registry.docker-cn.com` | China | Official Docker China mirror |

### Verify Mirror Configuration

```bash
docker info | grep -A 5 "Registry Mirrors"
```

---

## Step 5: Verify GPU Support in Docker

```bash
docker run --rm --gpus=all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Step 6: Prepare Model Weights

Download LLM model weights locally (e.g., `Qwen2.5-7B-Instruct`):
```bash
# Example using modelscope
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./Qwen2.5-7B-Instruct
```

---

## Step 7: Run AgentJet Swarm Server

```bash
# Create directories for logs and experiments
mkdir -p ./swarmlog ./swarmexp

# Run AgentJet Swarm Server
docker run --rm -it \
  -v /path/to/host/Qwen/Qwen2.5-7B-Instruct:/Qwen/Qwen2.5-7B-Instruct \
  -v ./swarmlog:/workspace/log \
  -v ./swarmexp:/workspace/saved_experiments \
  -p 10086:10086 \
  -e SWANLAB_API_KEY=$SWANLAB_API_KEY \
  --gpus=all \
  --shm-size=32GB \
  ghcr.io/modelscope/agentjet:main \
  bash -c "(ajet-swarm overwatch) & (NO_COLOR=1 LOGURU_COLORIZE=NO ajet-swarm start &>/workspace/log/swarm_server.log)"
```

### Flag Explanations

| Flag | Purpose |
|------|---------|
| `--rm` | Auto-remove container on exit |
| `-it` | Interactive TTY for TUI monitor |
| `-v <host>:<container>` | Mount model weights into container |
| `-p 10086:10086` | Expose API port for Swarm Clients |
| `--gpus=all` | Use all available GPUs |
| `--shm-size=32GB` | Shared memory for large model inference |

---

## Step 8: Verify Deployment

After launch, you should see the `ajet-swarm overwatch` TUI showing server state transitions:
```
OFFLINE -> BOOTING -> ROLLING -> WEIGHT_SYNCING -> ROLLING -> ...
```

The server enters **BOOTING** only after a Swarm Client sends a training configuration.

---

## Step 9: Connect Swarm Client (Optional)

From any machine that can reach the server:

```python
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from ajet.copilot.job import AgentJetJob

swarm_worker = SwarmClient("http://<server-ip>:10086")
swarm_worker.auto_sync_train_config_and_start_engine(
    AgentJetJob(
        algorithm="grpo",
        n_gpu=8,
        model="/Qwen/Qwen2.5-7B-Instruct",  # Container-side path
        batch_size=32,
        num_repeat=4,
    )
)
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| Server stays OFFLINE | No client connected | Run Swarm Client script |
| Model not found | Wrong container path | Verify `-v` mount matches `model` field |
| Cannot connect port 10086 | Firewall | Check firewall rules |
| Empty log file | Missing log directory | `mkdir -p ./swarmlog` |
| Image pull timeout | Slow registry access | Configure Docker mirror (Step 4) |
| Image pull fails | Wrong mirror URL | Try different mirror or use original URL |