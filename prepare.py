import os
import argparse
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import re

def preprocess_data(data):
    """
    Preprocesses the raw dataset.
    - Removes rows with missing critical fields.
    - Cleans text by removing special characters or trimming length.
    """
    # Drop rows with missing critical fields
    cleaned_data = [
        record for record in data
        if record["claim"] and record["main_text"] and record["label"] is not None
    ]

    # Optionally clean text (example: remove special characters)
    for record in cleaned_data:
        record["claim"] = re.sub(r"[^a-zA-Z0-9\s.,?!]", "", record["claim"])
        record["main_text"] = re.sub(r"[^a-zA-Z0-9\s.,?!]", "", record["main_text"])

    return cleaned_data

def prepare_data(input_dir: str, output_dir: str, model_name: str):
    """
    Prepares the PUBHEALTH dataset for training.

    Args:
        input_dir (str): Directory containing the raw dataset files.
        output_dir (str): Directory to save the processed dataset.
        model_name (str): Hugging Face model name for tokenizer.
    """
    print(f"Loading raw data from {input_dir}...")
    splits = ["train", "test", "validation"]
    dataset_splits = {}

    for split in splits:
        file_path = os.path.join(input_dir, f"{split}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Expected file {file_path} not found.")
        
        # Read JSONL file line-by-line
        data = []
        with open(file_path, "r") as f:
            for line in f:
                record = json.loads(line)
                data.append(json.loads(line))

            # Preprocess the data
            processed_data = preprocess_data(data)

            # Combine claim and main_text
            for record in processed_data:
                record["combined_text"] = (record["claim"]
                    + " Main text: " + record["main_text"] 
                    + " Explanation: " + record.get("explanation", "") 
                    + " Subject: " + record.get("subjects", "") 
                    + " Published on: " + record.get("date_published", "")
                )

                # data.append(record)
        
        # Convert the list of JSON objects to a Dataset
        dataset_splits[split] = Dataset.from_list(processed_data)

    dataset = DatasetDict(dataset_splits)

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["claim"], truncation=True, padding="max_length", max_length=1024)

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    print(f"Saving tokenized dataset to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    tokenized_datasets.save_to_disk(output_dir)
    print("Data preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PUBHEALTH dataset for training")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data",
        help="Directory containing the raw dataset (default: ./data)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_data",
        help="Directory to save the processed dataset (default: ./processed_data)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Hugging Face model name for tokenizer (default: distilbert-base-uncased)"
    )
    args = parser.parse_args()

    prepare_data(args.input_dir, args.output_dir, args.model_name)