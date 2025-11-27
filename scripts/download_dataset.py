import argparse
import glob
import os
import time

from beast_logger import print_list
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description="download Hugging Face dataset")
parser.add_argument("--target", default="openai/gsm8k", type=str, help="HuggingFace dataset name")
parser.add_argument(
    "--path",
    default="./dataset/openai/gsm8k",
    type=str,
    help="Path to the local directory where the dataset will be downloaded",
)
args = parser.parse_args()

snapshot_download(
    repo_id=args.target,
    repo_type="dataset",
    local_dir=args.path,
    resume_download=True,
)

time.sleep(2)


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


# python -m scripts.download_dataset --path='./dataset/openai/gsm8k' --target='openai/gsm8k'
