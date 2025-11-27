ms = input("modelscope ? (Y/n)")

if ms == "Y" or ms == "y":
    from loguru import logger
    from modelscope import snapshot_download

    cache_dir = input("model path (./modelscope_cache): ").strip()
    if not cache_dir:
        cache_dir = "./modelscope_cache"
    res = snapshot_download(input("model name: ").strip(), cache_dir=cache_dir)
    logger.success(res)

else:
    import os
    import subprocess

    repo_name = input("model name: ").strip()
    command = ["huggingface-cli", "download", "--resume-download", repo_name]
    process = subprocess.run(command, env=os.environ, check=True)
    if process.returncode == 0:
        print(f"Download {repo_name} succeeded")
    else:
        print(f"Download {repo_name} failed")

# python -m scripts.download_model
# Qwen/Qwen3-0.6B
