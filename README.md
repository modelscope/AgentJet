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
uv pip install -e external/verl -i https://mirrors.aliyun.com/pypi/simple/ --no-deps

# Install flash attention (must be installed last)
uv pip install --verbose flash-attn ring-flash-attn -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --no-build-isolation
```

注意：二者不能同时安装
```bash
# verl -> trinity
cd external/verl && uv pip uninstall . && cd ../..
# trinity -> verl
uv pip install -e external/verl -i https://mirrors.aliyun.com/pypi/simple/ --no-deps
```

## Get Started

本节仅内部沟通使用，后期重写。

我们提供一个多功能launcher用于调试和训练，您借助launcher选择任意训练框架启动训练。

1. 使用launcher进行调试（--backbone='debug'）：脱离trinity和verl，只与vllm（自动创建）连接，进行调试
```bash
clear && \
killer ray && killer python && \
killer vllm  && \
killer VLLM  && \
python launcher.py --with-appworld --conf launcher/appworld_linear_base/git-appworld-qwen2-agentscope-bz32-tp4-linear.yaml --backbone='debug' --with-logview
```
备注：当--backbone=debug时，程序不再使用ray，可以编写vscode的launch.json进行便捷的断点调试


2. 使用launcher进行训练：使用trinity进行训练
```bash
clear && \
ray stop && \
killer ray && \
killer python  && \
python launcher.py --with-appworld --conf launcher/appworld_linear_base/git-appworld-qwen2-agentscope-bz32-tp4-linear.yaml --with-ray --backbone='trinity'
```
备注：如果需要断点调试，请添加参数 `python launcher.py --db=TAG1|TAG2|TAG3 --conf=...`，并在代码中需要断点的地方标记一行特殊代码 ``


3. 使用launcher进行训练：使用verl进行训练
```bash
clear && \
ray stop && \
killer ray && \
killer python  && \
python launcher.py --with-appworld --conf launcher/appworld_linear_base/git-appworld-qwen2-agentscope-bz32-tp4-linear.yaml --with-ray --backbone='verl'
```
备注：如果需要断点调试，请添加参数 `python launcher.py --db=TAG1|TAG2|TAG3 --conf=...`




# note

FlashInfer?

clear && killer VLLM  && killer ray && killer python  && python launcher.py --with-appworld --conf launcher/appworld_linear_base/git-appworld-qwen2-agentscope-bz32-tp4-linear.yaml --with-ray --backbone='verl'

clear && killer VLLM  && killer ray && killer python  && python launcher.py --with-appworld --conf launcher/appworld_linear_base/git-appworld-qwen2-agentscope-bz32-tp4-linear.yaml --with-ray --backbone='verl'
