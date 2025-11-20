import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 必须放在第一行
import argparse
import glob
import time
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description="download Hugging Face dataset")
parser.add_argument("--target", default="openai/gsm8k", type=str, help="要下载的数据集仓库名称")
parser.add_argument(
    "--path",
    default="/mnt/data_cpfs/qingxu.fu/dataset/openai/gsm8k",
    type=str,
    help="路径到下载的本地目录",
)
args = parser.parse_args()

snapshot_download(
    repo_id=args.target,
    repo_type="dataset",
    local_dir=args.path,
    resume_download=True,
)

time.sleep(2)

from beast_logger import print_list

downloaded = []
for item in glob.glob(os.path.join(args.path, "**", "*")):
    downloaded += [os.path.abspath(item)]
print_list(downloaded, header="downloaded files")


def display_dataset(dataset_name, dataset_iter, header):
    from beast_logger import print_listofdict

    data = []
    for sample in dataset_iter:
        s = dict(sample)
        data.append(s)
    print_listofdict(data[:5], header=header)


try:
    import datasets

    dataset_iter = datasets.load_dataset(args.path, name="main", split="train")
    display_dataset(args.target, dataset_iter, header="train")
    dataset_iter = datasets.load_dataset(args.path, name="main", split="test")
    display_dataset(args.target, dataset_iter, header="test")
except Exception as e:
    print(f"Error loading dataset {args.target}: {e}")


# python -m scripts.download_dataset --path='/root/data/gsm8k' --target='openai/gsm8k'
