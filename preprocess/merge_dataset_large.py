import os
import random

def load_and_strip(file_path, marker):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().strip().split(f"{marker}\n")
        return [s.strip() for s in raw if s.strip()]

# Get the base directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define input file paths (relative to this script)
enron_path = os.path.join(base_dir, "cleaned_dataset/cleaned_emails.txt")
pii_path = os.path.join(base_dir, "cleaned_dataset/cleaned_pii.txt")
# wiki_path = os.path.join(base_dir, "train_dataset/wikitext103_clean.txt")

# Load datasets and strip markers
enron_samples = load_and_strip(enron_path, "<|email|>")
pii_samples = load_and_strip(pii_path, "<|pii|>")
# wiki_samples = load_and_strip(wiki_path, "<|wikitext|>")

# Combine and shuffle
all_samples = enron_samples + pii_samples  # + wiki_samples
random.seed(42)
random.shuffle(all_samples)

# Define output path
output_path = os.path.join(base_dir, "merged_dataset/merged_large.txt")

# Write to output file
with open(output_path, "w", encoding="utf-8") as f:
    for sample in all_samples:
        f.write(sample.strip() + "\n\n")

