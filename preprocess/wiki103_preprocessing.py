from datasets import load_dataset
import re
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Step 1: Strip whitespace
def clean(example):
    text = example["text"].strip()
    return {"text": text}

# Step 2: Remove Wikitext tokenization artifacts
def remove_wikitext_artifacts(example):
    text = example["text"]
    text = text.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
    return {"text": text}

# Step 3: Remove non-ASCII characters (keep only English)
def remove_non_english(example):
    text = example["text"]
    clean_text = re.sub(r"[^\x00-\x7F]+", "", text)
    return {"text": clean_text}

# Combine all cleaning steps and write to file
def clean_and_save(dataset):
    dataset = dataset.map(clean)
    dataset = dataset.filter(lambda x: x["text"] != "" and not x["text"].startswith("="))

    dataset = dataset.map(remove_wikitext_artifacts)
    dataset = dataset.map(remove_non_english)
    output_path = os.path.join(BASE_DIR,"cleaned_dataset/cleaned_wikitext103.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in dataset["train"]:
            f.write(ex["text"] + "\n")

if __name__ == '__main__':
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    clean_and_save(dataset)
