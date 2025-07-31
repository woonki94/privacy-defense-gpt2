import os
from datasets import load_dataset, concatenate_datasets

def get_dataset(shuffle=True, seed=42):
    # Get the absolute path to the directory this script is in
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the dataset file
    dataset_path = os.path.normpath(os.path.join(current_dir, "..", "preprocess", "merged_dataset","merged_large.txt"))

    dataset = load_dataset("text", data_files={"train": dataset_path})["train"]

    # Remove empty or super-short samples before tokenization
    dataset = dataset.filter(lambda x: x["text"] and len(x["text"].strip()) > 10)

    # Optional shuffle
    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    return dataset


