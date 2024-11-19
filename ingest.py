import os
import argparse
from datasets import load_dataset

def download_dataset(output_dir: str):
    """
    Downloads the PUBHEALTH dataset and saves it in the specified directory.

    Args:
        output_dir (str): The directory where the dataset will be saved.
    """
    print(f"Downloading PUBHEALTH dataset...")
    dataset = load_dataset("ImperialCollegeLondon/health_fact")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the dataset in the specified directory
    for split in dataset.keys():
        file_path = os.path.join(output_dir, f"{split}.json")
        dataset[split].to_json(file_path)
        print(f"Saved {split} dataset to {file_path}")

    print("Dataset download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PUBHEALTH dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save the downloaded dataset (default: ./data)"
    )
    args = parser.parse_args()

    download_dataset(args.output_dir)