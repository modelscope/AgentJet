# AgentScope Tune

AgentScope Tune, or **ASTune**, is an advanced agent training framework for tuning AgentScope workflow and agent(s).


## Installation

You can choose between `Trinity training backbone` and `Verl training backbone`. We recommend using `uv` to setup the dependencies and `conda` also works.

1. Trinity backbone (Option 1)

```bash
# Create virtual environment
uv venv --python=3.10.16
source .venv/bin/activate
git clone https://github.com/binary-husky/Trinity-RFT external/trinity

# Install dependencies
uv pip install --upgrade pip setuptools packaging -i https://mirrors.aliyun.com/pypi/simple/
uv pip install -r requirements_trinity.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --prerelease=allow
uv pip install -e external/trinity -i https://mirrors.aliyun.com/pypi/simple/ --no-deps

# Install flash attention (must be installed last)
uv pip install --verbose flash-attn ring-flash-attn -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --no-build-isolation
```


2. VERL Backbone (Option 2)

```bash
# Create virtual environment
uv venv --python=3.10.16
source .venv/bin/activate
git clone https://github.com/binary-husky/verl.git external/verl

# Install dependencies
uv pip install --upgrade pip setuptools packaging -i https://mirrors.aliyun.com/pypi/simple/
uv pip install -r requirements_verl.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --prerelease=allow
uv pip install -e external/verl -i https://mirrors.aliyun.com/pypi/simple/

# Install flash attention (must be installed last)
uv pip install --verbose flash-attn ring-flash-attn -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --no-build-isolation
```

## Setup

本节仅内部沟通使用


### Creating a AgentScope Workflow



1. 脱离trinity和verl，只与vllm（自动创建）连接，进行调试
```bash
clear && \
killer ray && killer python && \
python launcher.py --with-appworld --conf launcher/appworld_linear_base/git-appworld-qwen2-agentscope-bz32-tp4-linear.yaml --backbone='debug'
```

2. 使用trinity进行训练
```bash
clear && \
ray stop && \
killer ray && \
killer python  && \
python launcher.py --with-appworld --conf launcher/appworld_linear_base/git-appworld-qwen2-agentscope-bz32-tp4-linear.yaml --with-ray --backbone='trinity'
```


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