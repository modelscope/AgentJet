# OpenClaw x OpenJudge 微调更懂用户的助手

## 架构概述

```txt
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
```

## 启动方法

### 1. 灵俊上启动swarm server，并ssh打通到龙虾

```bash
# agentjet 的 git checkout add-openclaw-training
ajet-swarm start        # terminal 1 (start engine)
ajet-swarm overwatch    # terminal 2 (watch status)
```

```bash
# 如果直接在灵俊上跑龙虾，这步省略
ssh -R 8090:localhost:8090 -p 22222 fuqingxu@111.36.208.22 -N   -o ServerAliveInterval=30   -o ServerAliveCountMax=3
```

### 2. 龙虾服务器启动OpenJudge & AgentJet请求一转多服务

```bash
# agentjet 的 git checkout add-openclaw-training
python -m ajet.tuner_lib.experimental.oai_model_one2many
```

### 3. 启动龙虾，打开配置网页，然后

(3-1) 启动龙虾
(3-2) 龙虾配置网页：设置 > 配置 > Models > Model Providers > vllm:http://localhost:8090/v1
![alt text](https://img.alicdn.com/imgextra/i2/O1CN01LK3R1W1Dy7bq8jLRR_!!6000000000284-2-tps-2450-1584.png)
(3-3) 尝试提交问题
![alt text](https://img.alicdn.com/imgextra/i1/O1CN013yqN5U1fpFApRMNzN_!!6000000004055-2-tps-3529-1594.png)
(3-4) 重复 (3-3) AgentJet会自动寻找合适的时机执行训练
![alt text](https://img.alicdn.com/imgextra/i3/O1CN01CBX7ug1TLDp2qPanE_!!6000000002365-2-tps-2756-1118.png)

## 调试奖励：

修改 `ajet.tuner_lib.experimental.oai_model_one2many` 中的 `on_user_submit_new_requests` 和 `on_compute_relative_reward`
