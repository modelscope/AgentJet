import argparse


parser = argparse.ArgumentParser(description="download Hugging Face dataset")
parser.add_argument("--target", default="openai/gsm8k", type=str, help="HuggingFace dataset name")
args = parser.parse_args()

def display_dataset(dataset_name, dataset_iter, header):
    from beast_logger import print_listofdict

    data = []
    for sample in dataset_iter:
        s = dict(sample)
        data.append(s)
    print_listofdict(data[:5], header=header)


try:
    import datasets

    dataset_iter = datasets.load_dataset(args.target, name="default", split="train")
    display_dataset(args.target, dataset_iter, header="train")
    dataset_iter = datasets.load_dataset(args.target, name="default", split="test")
    display_dataset(args.target, dataset_iter, header="test")
except Exception as e:
    print(f"Error loading dataset {args.target}: {e}")


# python -m scripts.download_dataset --path='./dataset/openai/gsm8k' --target='openai/gsm8k'
