ms = input("modelscope ? (Y/n)")

if ms == "Y" or ms == "y":

    from modelscope import snapshot_download
    from loguru import logger

    cache_dir = input("model path ( /mnt/data/model_cache/modelscope/hub/Qwen ): ").strip()
    if not cache_dir:
        cache_dir = "/mnt/data/model_cache/modelscope/hub/Qwen"
    res = snapshot_download(input("model name: ").strip(), cache_dir=cache_dir)
    logger.success(res)

else:

    import os
    import subprocess

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    repo_name = input("model name: ").strip()
    command = ["huggingface-cli", "download", "--resume-download", repo_name]
    process = subprocess.run(command, env=os.environ, check=True)
    if process.returncode == 0:
        print(f"成功下载 {repo_name}")
    else:
        print(f"下载 {repo_name} 失败")

# python -m scripts.download_model
# Qwen/Qwen3-0.6B
