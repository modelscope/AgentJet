import argparse
import glob
import os

from beast_logger import print_list
from datasets import DatasetDict, load_dataset

parser = argparse.ArgumentParser(description="download Hugging Face dataset")
parser.add_argument(
    "--target", default="Jiayi-Pan/Countdown-Tasks-3to4", type=str, help="HuggingFace dataset name"
)
parser.add_argument(
    "--path",
    default="./dataset",
    type=str,
    help="Path to the local directory where the dataset will be downloaded",
)
args = parser.parse_args()

# Don't download to local, just load directly from HuggingFace
print(f"Loading dataset from {args.target}...")


def display_dataset(dataset_name, dataset_iter, header):
    from beast_logger import print_listofdict

    data = []
    for sample in dataset_iter:
        s = dict(sample)
        data.append(s)
    print_listofdict(data[:5], header=header)


try:
    import datasets

    # Load the original dataset directly from HuggingFace (no local download)
    print("\nLoading original dataset from HuggingFace...")
    original_dataset = load_dataset(args.target, split="train")

    print(f"\nOriginal dataset size: {len(original_dataset)}")

    # Print dataset schema (column names and types)
    print("\n" + "=" * 80)
    print("Dataset Schema:")
    print("=" * 80)
    for col_name in original_dataset.column_names:
        col_type = original_dataset.features[col_name]
        print(f"Column: {col_name:20s} | Type: {col_type}")

    # Print examples for each column
    print("\n" + "=" * 80)
    print("Sample Data (First 3 Examples):")
    print("=" * 80)
    for i in range(min(3, len(original_dataset))):
        print(f"\nExample {i+1}:")
        for col_name in original_dataset.column_names:
            value = original_dataset[i][col_name]
            # Truncate long strings for display
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(f"  {col_name}: {value}")

    # Split dataset: 1024 examples for test, 10x (10240) for training
    test_size = 1024
    train_size = test_size * 10
    total_size = len(original_dataset)

    # Ensure we have enough data
    if total_size < test_size + train_size:
        print(
            f"\nWarning: Dataset size ({total_size}) is smaller than required ({test_size + train_size})"
        )
        print("Adjusting sizes proportionally...")
        test_size = min(test_size, total_size // 11)
        train_size = test_size * 10

    print("\n" + "=" * 80)
    print(f"Splitting dataset: {test_size} test samples, {train_size} train samples")
    print("=" * 80)

    # Create train/test split
    test_dataset = original_dataset.select(range(test_size))
    train_dataset = original_dataset.select(range(test_size, test_size + train_size))

    # Create a DatasetDict with train and test splits
    split_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Create output directory: dataset/Countdown-Tasks
    output_data_dir = os.path.join(args.path, "Countdown-Tasks")
    os.makedirs(output_data_dir, exist_ok=True)

    # Save as parquet files in the Countdown-Tasks directory
    train_parquet_path = os.path.join(output_data_dir, "train-00000-of-00001.parquet")
    test_parquet_path = os.path.join(output_data_dir, "test-00000-of-00001.parquet")

    print("\nSaving split datasets...")
    train_dataset.to_parquet(train_parquet_path)
    test_dataset.to_parquet(test_parquet_path)

    print(f"✓ Saved training set to: {train_parquet_path}")
    print(f"✓ Saved test set to: {test_parquet_path}")

    # Display statistics
    print("\n" + "=" * 80)
    print("Dataset Statistics:")
    print("=" * 80)
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Train/Test ratio: {len(train_dataset) / len(test_dataset):.1f}x")
    print(f"Total used: {len(train_dataset) + len(test_dataset)}")
    print(f"Original dataset size: {len(original_dataset)}")

    # Display sample data from train and test sets
    print("\n")
    display_dataset(args.target, train_dataset, header="train (first 5 samples)")
    print("\n")
    display_dataset(args.target, test_dataset, header="test (first 5 samples)")

    # Verify the split files can be loaded
    print("\n" + "=" * 80)
    print("Verifying split files...")
    print("=" * 80)
    train_loaded = datasets.load_dataset("parquet", data_files=train_parquet_path, split="train")
    test_loaded = datasets.load_dataset("parquet", data_files=test_parquet_path, split="train")
    print(f"✓ Train parquet loaded successfully: {len(train_loaded)} samples")
    print(f"✓ Test parquet loaded successfully: {len(test_loaded)} samples")

    # List saved files
    print("\n" + "=" * 80)
    print("Saved Files:")
    print("=" * 80)
    saved_files = []
    for item in glob.glob(os.path.join(output_data_dir, "*"), recursive=False):
        if os.path.isfile(item):
            saved_files.append(os.path.abspath(item))
    print_list(saved_files, header="saved files")

    # Final file structure
    print("\n" + "=" * 80)
    print("Final Directory Structure:")
    print("=" * 80)
    print(f"{args.path}/")
    print("└── Countdown-Tasks/")
    print(f"    ├── train-00000-of-00001.parquet ({len(train_dataset)} samples)")
    print(f"    └── test-00000-of-00001.parquet ({len(test_dataset)} samples)")

except Exception as e:
    print(f"Error loading dataset {args.target}: {e}")
    import traceback

    traceback.print_exc()
