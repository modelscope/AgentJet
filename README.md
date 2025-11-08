# ASTune

ASTune is an advanced reinforcement learning framework for training and evaluating agentic AI systems. Built on top of the BeyondAgent platform, it provides seamless switching between different tasks and background environments, supporting both vLLM and SGLang inference backends.

## Features

- **Flexible Inference Backends**: Seamlessly switch between vLLM (high-throughput asynchronous inference) and SGLang (structured generative language tasks)
- **GSPO Loss Mode**: Gradient-based Structured Policy Optimization for more efficient reinforcement learning
- **Multi-Environment Support**: Integration with diverse simulation platforms via `env_service`
- **MoE Model Support**: Scalable training with Mixture of Experts models
- **Advanced Logging**: Token-level logging system for fine-grained analysis
- **Dynamic Sampling**: Enhanced data efficiency through advanced sampling techniques
- **Real-time Monitoring**: Progress tracking with real-time rollout visualization

## Installation

You can choose between `Trinity training backbone` and `Verl training backbone`.

1. Trinity backbone

```bash
uv venv --python=3.10.16
source .venv/bin/activate
git clone https://github.com/binary-husky/Trinity-RFT external/trinity
uv pip install --upgrade pip setuptools packaging -i https://mirrors.aliyun.com/pypi/simple/
uv pip install -r std_req.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --prerelease=allow
uv pip install -e external/trinity -i https://mirrors.aliyun.com/pypi/simple/
```


2. VERL Backbone

```bash
# Create virtual environment
uv venv --python=3.10.16
source .venv/bin/activate
git clone https://github.com/binary-husky/verl.git external/verl

# Install dependencies
uv pip install --upgrade pip setuptools packaging -i https://mirrors.aliyun.com/pypi/simple/
uv pip install -r std_req.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --prerelease=allow
uv pip install -e external/verl -i https://mirrors.aliyun.com/pypi/simple/

# Install flash attention (must be installed last)
uv pip install --verbose flash-attn ring-flash-attn -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --no-build-isolation
```

## Quick Start

### Launching with Different Environments

```bash
# Launch with appworld environment
python launcher.py --with-appworld

# Launch with a specific configuration
python launcher.py --conf launcher/appworld_context_clip/appworld-context-clip.yaml --with-appworld

# Launch with webshop environment
python launcher.py --with-webshop
```

### Configuration

To switch between vLLM and SGLang modes, modify the `mode` parameter in your configuration file:

```yaml
actor_rollout_ref:
  rollout:
    mode: async  # Use 'async' for vLLM or specify another mode for SGLang
```

To enable GSPO loss mode:

```yaml
actor_rollout_ref.actor.policy_loss.loss_mode=gspo
```

## Advanced Features

### Multi-Environment Support

Define environment settings in your configuration file:

```yaml
env_service:
  env_url: http://localhost:8080
  env_type: webshop
```

### Dynamic Sampling

Enable oversampling to improve data efficiency:

```yaml
enable_oversample: True
submit_oversample_multiplier: 2
```

## Project Structure

```
├── agentopia/                 # Agentopia modules
├── beyondagent/               # BeyondAgent core components
├── external/                  # External dependencies
├── launcher/                  # Launcher configurations
│   ├── appworld_context_clip/
│   ├── appworld_linear_base/
│   └── ...
├── checkpoints/               # Model checkpoints
├── logs/                      # Log files
└── outputs/                   # Output files
```

## Contributing

Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.