# Beast-Logger Usage

Beast-logger is a logging kit built for LLM systems,
providing reliable high-resolution token-level LLM activity
that is unprecedented in any other projects.

Here is how to use beast-logger in agentscope-tuner.

## Usage in agentscope-tuner

1. Start training or debugging with agentscope-tuner launcher.

2. Wait until the first batch is completed.

3. Locate log files. By default, they will be placed at `saved_experiments/${experiment_name}`. For example:
`saved_experiments/benchmark_frozenlake_20251223_2305`

4. Run `beast_logger_go` command in the VSCode terminal (or any other software with port-forwarding ability) to start the web log-viewer. Click `http://127.0.0.1:8181` to open it (VSCode will automatically forward this port from server -> your local computer)

<div align="center">
<img width="480" alt="image" src="https://img.alicdn.com/imgextra/i4/O1CN01kfiOlZ1SRnsq7NZLP_!!6000000002244-2-tps-1414-968.png"/>
</div>

5. Fill the **ABSOLUTE** path of the log files and click `submit`.

    > Hint: absolute path is recommended.
    >
    > However, you can also use relative path, if `beast_logger_go` command is launched at same working dir.

    > Warning: Beast-logger recursively scans this path,
    >
    > thus, where possible, selects the innermost directory containing the fewest files to read logs faster.

<div align="center">
<img width="480" alt="image" src="https://img.alicdn.com/imgextra/i3/O1CN01v6EZUi1wSf6BZrXWW_!!6000000006307-2-tps-1864-946.png"/>
</div>


6. Choose entry to display

* Yellow tokens: tokens that are excluded from loss computation.
* Blue tokens: tokens that participant loss computation.
* Hovor your mouse on one of the tokens: show the logprob value of that token.

<div align="center">
<img width="480" alt="image" src="https://img.alicdn.com/imgextra/i2/O1CN018O2JSB1rWG8GDDQVD_!!6000000005638-2-tps-2222-1391.png"/>
</div>
