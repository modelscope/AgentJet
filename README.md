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

项目提供一个多功能launcher用于调试和训练，借助launcher，只需要修改一个`--backbone`参数，就选择任意训练框架启动训练 or 调试。

1. 使用launcher进行全链路调试（--backbone='debug'）：脱离trinity和verl，只与vllm（自动创建）连接，进行调试
    ```bash
    # （训练math agent demo）建议开始前杀死所有ray、env_service进程
    clear && \
    python launcher.py --conf launcher/math_agent/git-math-agentscope.yaml --backbone='debug' --with-logview

    # （训练appworld demo）建议开始前杀死所有ray、env_service进程
    clear && \
    python launcher.py --with-appworld --conf launcher/appworld_linear_base/git-appworld-qwen2-agentscope-bz32-tp4-linear.yaml --backbone='debug' --with-logview
    ```
备注：当--backbone=debug时，程序不再使用ray，可以编写vscode的launch.json进行便捷的断点调试，launch.json的配置见文档最后


2. 使用launcher进行训练：使用trinity进行训练
    ```bash
    # 建议开始前杀死所有ray、vllm、env_service进程
    clear && \
    python launcher.py --with-appworld --conf launcher/appworld_linear_base/git-appworld-qwen2-agentscope-bz32-tp4-linear.yaml --with-ray --backbone='trinity'

    python launcher.py --conf launcher/math_agent/git-math-agentscope.yaml --with-ray --backbone='trinity'
    ```
备注：如果需要断点调试，请添加参数 `python launcher.py --db=TAG1|TAG2|TAG3 --conf=...`，并在代码中需要断点的地方标记一行特殊代码 ``


3. 使用launcher进行训练：使用verl进行训练
    ```bash
    # 建议开始前杀死所有ray、vllm、env_service进程
    clear && \
    python launcher.py --with-appworld --conf launcher/appworld_linear_base/git-appworld-qwen2-agentscope-bz32-tp4-linear.yaml --backbone='verl'

    python launcher.py --conf launcher/math_agent/git-math-agentscope.yaml --backbone='verl'
    ```
备注：如果需要断点调试，请添加参数 `python launcher.py --db=TAG1|TAG2|TAG3 --conf=...`




# note

FlashInfer?

clear && killer VLLM  && killer ray && killer python  && python launcher.py --with-appworld --conf launcher/appworld_linear_base/git-appworld-qwen2-agentscope-bz32-tp4-linear.yaml --with-ray --backbone='verl'

clear && killer VLLM  && killer ray && killer python  && python launcher.py --with-appworld --conf launcher/appworld_linear_base/git-appworld-qwen2-agentscope-bz32-tp4-linear.yaml --with-ray --backbone='verl'


- `launche.json` for vscode debugging
```json
{

    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Launch rollout",
            "type": "debugpy",
            "request": "launch",
            "program": "launcher.py",
            "console": "integratedTerminal",
            "args": [
                "--backbone",  "debug",
                "--conf", "xxxx/xxxx/xxxx.yaml"
            ],
            "env": {
            }
        },
    ]
}
```