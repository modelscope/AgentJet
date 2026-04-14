编写swarm模式训练任务 tutorial/opencode_build_aime

训练集 dapo-math-17k.parquet

测试集 aime-2024

奖励函数参考 cradle.md 和 /tmp/verl 仓库

下载数据集时使用proxychains

参考 tutorial/example_math_swarm

参考ajet/copilot/write-swarm-client



------------------------


编写一个agent_roll_test.py，其中用dashscope api临时取代swarm api，对reward进行测试

使用模型 qwen3 max

从环境变量读取 DASHSCOPE_API_KEY

base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

model: "qwen3-max"
