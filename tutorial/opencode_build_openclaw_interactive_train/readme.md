# OpenClaw Interactive Training

这是一个带交互功能的 OpenClaw vLLM endpoint，支持用户在训练过程中通过对话动态调整训练参数和奖励标准。

## 功能特性

### 1. 用户意见检测与动态 Judge Prompt 更新

当用户在对话中表达对系统回答的意见时（例如："请幽默一点"、"你太傻了"），系统会：
- 使用 qwen-max 检测用户是否表达了意见
- 自动更新内存中的 judge prompt，将用户偏好纳入奖励计算

**示例对话：**
```
用户: 请幽默一点
系统: [检测到用户偏好] -> 更新 judge prompt 以增加对幽默性的评估权重
```

### 2. /agentjet 命令支持

用户可以通过 `/agentjet` 命令动态更新训练配置：

**支持的命令格式：**
```
/agentjet: 切换 '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen2___5-14B-Instruct' 模型
/agentjet: 更新 batch_size 为 64
/agentjet: 更新 n_gpu 为 4
```

命令会触发：
1. 解析用户命令（使用 qwen-max）
2. 更新 `ajet_job` 配置
3. 调用 `swarm_client.auto_sync_train_config_and_start_engine(ajet_job, force_restart=True)` 重启引擎

### 3. 奖励计算（已移除外向型奖励）

奖励公式：
```
final_reward = quality * (W_USER_FEEDBACK * user_feedback_score
                        + W_RELEVANCE     * relevance_score
                        + W_DIVERSITY     * diversity_score)
```

默认权重：
- `W_USER_FEEDBACK = 0.3` - 基于动态用户反馈的评分
- `W_RELEVANCE = 0.4` - 回答与问题的相关性
- `W_DIVERSITY = 0.3` - 回答的多样性（避免重复）

## 文件结构

```
tutorial/opencode_build_openclaw_interactive_train/
├── fake_vllm_endpoint.py          # 假 vLLM 端点，处理请求复制和奖励计算
├── on_compute_relative_reward.py  # 奖励计算逻辑，包含用户意见检测和命令解析
├── on_user_submit_new_requests.py # 用户请求记录
├── download_dataset.py            # 数据集下载工具
├── mock_user_request.py           # 测试用的模拟用户请求
└── readme.md                      # 本文档
```

## 部署步骤

### 1. 启动 Swarm Server

在 GPU 服务器上运行：
```bash
ajet-swarm start --swarm-port=10086
```

或同时启动监控：
```bash
(ajet-swarm start &> ajet-swarm-server.log) & (ajet-swarm overwatch)
```

### 2. 启动 Fake vLLM Endpoint

```bash
cd tutorial/opencode_build_openclaw_interactive_train
python fake_vllm_endpoint.py
```

服务将在 `http://localhost:8090` 启动。

### 3. 配置环境变量（可选）

```bash
export DASHSCOPE_API_KEY="your-api-key"        # 用于 judge model
export AJET_SWARM_URL="http://localhost:10086"  # Swarm 服务器地址
export NUM_REPEAT=4                             # GRPO 重复次数
export W_RELEVANCE=0.4                          # 相关性权重
export W_DIVERSITY=0.3                          # 多样性权重
export W_USER_FEEDBACK=0.3                      # 用户反馈权重
```

### 4. 使用 OpenAI 兼容 API 进行交互

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="any-key"  # 实际不检查
)

# 正常对话
response = client.chat.completions.create(
    model="any-model",
    messages=[{"role": "user", "content": "你好，介绍一下自己"}]
)

# 表达偏好（会更新 judge prompt）
response = client.chat.completions.create(
    model="any-model",
    messages=[{"role": "user", "content": "请幽默一点"}]
)

# 使用 /agentjet 命令切换模型
response = client.chat.completions.create(
    model="any-model",
    messages=[{"role": "user", "content": "/agentjet: 切换 '/path/to/new/model' 模型"}]
)
```

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI 兼容的聊天补全接口 |
| `/health` | GET | 健康检查 |
| `/requests` | GET | 获取最近的用户请求记录 |

## 工作原理

```
用户请求
    ↓
fake_vllm_endpoint 接收请求
    ↓
复制请求 N 次 (NUM_REPEAT)
    ↓
并行发送到 swarm server 的真实 vLLM
    ↓
收集所有响应
    ↓
on_compute_relative_reward:
  ├── 检测用户意见 → 更新 judge prompt
  ├── 检测 /agentjet 命令 → 更新配置并重启引擎
  └── 计算每个响应的奖励
    ↓
选择奖励最高的响应返回给用户
    ↓
将奖励提交到 swarm server 用于训练
```

## 注意事项

1. **qwen-max API**: 用户意见检测和命令解析需要 qwen-max API，确保 `DASHSCOPE_API_KEY` 已设置
2. **引擎重启**: 使用 `/agentjet` 命令切换模型会触发引擎重启，期间服务可能短暂不可用
3. **judge prompt 持久化**: 动态更新的 judge prompt 仅存储在内存中，服务重启后会重置
4. **权重调整**: 可通过环境变量调整各项奖励的权重
