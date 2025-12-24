# Beast-Logger 使用指南

Beast-logger 是一个面向 LLM 系统构建的日志工具包，能够提供可靠的、细粒度到 **token-level** 的高分辨率 LLM 活动日志，这种粒度与完整性在其他项目中几乎前所未有。

下面介绍如何在 agentscope-tuner 中使用 beast-logger。

## 在 agentscope-tuner 中使用

1. 使用 agentscope-tuner 的 launcher 启动训练或调试。

2. 等待第一个 batch 完成。

3. 找到日志文件。默认情况下，它们会放在 `saved_experiments/${experiment_name}` 目录下。例如：
   `saved_experiments/benchmark_frozenlake_20251223_2305`

4. 在 VSCode 终端（或任何支持端口转发的软件）中运行 `beast_logger_go` 命令启动 Web 日志查看器。然后点击 `http://127.0.0.1:8181` 打开页面（VSCode 会自动将该端口从服务器转发到你的本地电脑）。

<div align="center">
<img width="480" alt="image" src="https://img.alicdn.com/imgextra/i4/O1CN01kfiOlZ1SRnsq7NZLP_!!6000000002244-2-tps-1414-968.png"/>
</div>

5. 填入日志文件所在目录的**绝对路径（ABSOLUTE path）**，然后点击 `submit`。（Beast-logger 会递归扫描该路径，并在可能的情况下自动选择包含文件数量最少的最深层目录。）

<div align="center">
<img width="480" alt="image" src="https://img.alicdn.com/imgextra/i3/O1CN01v6EZUi1wSf6BZrXWW_!!6000000006307-2-tps-1864-946.png"/>
</div>

6. 选择要展示的条目（entry）。

* 黄色 tokens：被排除在 loss 计算之外的 tokens
* 蓝色 tokens：参与 loss 计算的 tokens
* 将鼠标悬停在某个 token 上：会显示该 token 的 logprob 值

<div align="center">
<img width="480" alt="image" src="https://img.alicdn.com/imgextra/i2/O1CN018O2JSB1rWG8GDDQVD_!!6000000005638-2-tps-2222-1391.png"/>
</div>
