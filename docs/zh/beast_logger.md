# Beast-Logger 使用指南

Beast-logger 是一个面向 LLM 系统构建的日志工具包，能够提供可靠的、细粒度到 **token-level** 的高分辨率 LLM 活动日志，这种粒度与完整性在其他项目中几乎前所未有。

!!! success "核心优势"
    Beast-logger 可以记录每一个 token 的详细信息，包括 Token ID、Loss Mask 和 Logprobs，非常适合工作流开发和智能体诊断。

---

## 在 AgentJet 中使用

<div class="workflow-single">
<div class="workflow-header">使用流程</div>

<div class="workflow">
<ol class="workflow-steps">
<li><strong>启动训练</strong>

使用 AgentJet 的 launcher 启动训练或调试。</li>
<li><strong>等待数据生成</strong>

等待第一个 batch 完成，日志文件会自动生成。</li>
<li><strong>启动 Web 查看器</strong>

运行 `beast_logger_go` 命令打开可视化界面。</li>
<li><strong>浏览日志</strong>

选择日志目录并查看 token 级别的详细信息。</li>
</ol>
</div>
</div>

---

## 详细步骤

### Step 1：启动训练

使用 AgentJet 的 launcher 启动训练或调试。

### Step 2：找到日志文件

等待第一个 batch 完成后，日志文件会保存在 `saved_experiments/${experiment_name}` 目录下。

!!! example "日志目录示例"
    ```
    saved_experiments/benchmark_frozenlake_20251223_2305
    ```

### Step 3：启动 Web 查看器

在 VSCode 终端（或任何支持端口转发的软件）中运行：

```bash
beast_logger_go
```

然后点击 `http://127.0.0.1:8181` 打开页面。

!!! tip "端口转发"
    VSCode 会自动将该端口从服务器转发到您的本地电脑。

<div align="center">
<img width="480" alt="Beast Logger 启动界面" src="https://img.alicdn.com/imgextra/i4/O1CN01kfiOlZ1SRnsq7NZLP_!!6000000002244-2-tps-1414-968.png"/>
</div>

### Step 4：选择日志目录

填入日志文件所在目录的**绝对路径（ABSOLUTE path）**，然后点击 `submit`。

!!! warning "重要提示"
    Beast-logger 会递归扫描该路径，并在可能的情况下自动选择包含文件数量最少的最深层目录。

<div align="center">
<img width="480" alt="选择日志目录" src="https://img.alicdn.com/imgextra/i3/O1CN01v6EZUi1wSf6BZrXWW_!!6000000006307-2-tps-1864-946.png"/>
</div>

### Step 5：浏览日志

选择要展示的条目（entry），查看 token 级别的详细信息：

<div align="center">
<img width="600" alt="Token 级日志展示" src="https://img.alicdn.com/imgextra/i2/O1CN018O2JSB1rWG8GDDQVD_!!6000000005638-2-tps-2222-1391.png"/>
</div>

---

## Token 颜色说明

| 颜色 | 含义 |
|------|------|
| **黄色** | 被排除在 loss 计算之外的 tokens |
| **蓝色** | 参与 loss 计算的 tokens |

!!! tip "查看 Logprob"
    将鼠标悬停在某个 token 上，会显示该 token 的 logprob 值。

---

## 下一步

<div class="card-grid">
<a href="./visualization/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:chart-line.svg" class="card-icon card-icon-agent" alt=""><h3>训练可视化</h3></div><p class="card-desc">了解更多训练监控和可视化工具。</p></a>
<a href="./configuration/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:cog.svg" class="card-icon card-icon-tool" alt=""><h3>配置指南</h3></div><p class="card-desc">完整的配置选项参考。</p></a>
</div>
