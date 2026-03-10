# OpenClaw x OpenJudge 微调更懂用户的助手

## 架构概述


        OpenClaw 配置网页：设置>配置>Models>Model Providers>vllm:http://localhost:8090/v1
                            |
                            |
用户 --> OpenClaw界面 --> OpenClaw中枢 --> 假 vLLM --> 一个请求复制多份 --> 计算相对奖励 --> 提交奖励给AgentJet
                                              |              |             |                     |
                                              |              |             |                     |
                                              |              |             |         （bash: `ajet-swarm start`, port 10086）
                                              |              |             |
                                (bash: `python -m ajet.tuner_lib.experimental.oai_model_one2many`, port 8090)
                                                                 |                |
                                                                 |                |
                                                                 |                |
                                                                 |                |
                                                                 |                |
                                                                 |                |
                                                                 |                |
                                                                 |                |
                                                                 |                |
                                                              OpenJudge          OpenJudge
                                                            读取用户Query    读取所有Query的所有答案并计算奖励


## 启动方法

### 1. 灵俊上启动swarm server，并ssh打通到龙虾

```bash
ajet-swarm start
```

```bash
# 如果直接在灵俊上跑龙虾，这步省略
ssh -R 8090:localhost:8090 -p 22222 fuqingxu@111.36.208.22 -N   -o ServerAliveInterval=30   -o ServerAliveCountMax=3
```

### 2. 龙虾服务器启动OpenJudge & AgentJet请求一转多服务

```bash
python -m ajet.tuner_lib.experimental.oai_model_one2many
```

### 3. 启动龙虾，打开配置网页，然后

(3-1) 启动龙虾
(3-2) 龙虾配置网页：设置 > 配置 > Models > Model Providers > vllm:http://localhost:8090/v1


## 调试奖励：

修改 `ajet.tuner_lib.experimental.oai_model_one2many` 中的 `on_user_submit_new_requests` 和 `on_compute_relative_reward`
