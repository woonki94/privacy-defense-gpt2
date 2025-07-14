from datasets import load_dataset
import os
from datasets import load_dataset
import re
from cleantext import clean
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def cleantext(text):
    return clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        lower=True,
        no_line_breaks=True,
        no_urls=True,
        no_emails=False,
        no_punct=True,
        replace_with_punct="",
    )

# Cleaning function (lightweight)
def clean_text(example):
    text = example["text"].strip().lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = " ".join(text.split())        # Normalize whitespace

    example["text"] = text
    return example

if __name__ == '__main__':
    # Load and shuffle the dataset
    dataset = load_dataset("Skylion007/openwebtext", split="train", trust_remote_code=True)
    dataset = dataset.shuffle(seed=42)

    # Limit to approx 5GB worth (~1M documents)
    subset = dataset.select(range(1000000))

    # Clean
    subset = subset.map(clean_text, batched=False)

    # Write output
    output_path = os.path.join(BASE_DIR,"cleaned_dataset/cleaned_openwebtext_5gb.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        for example in subset:
            if example["text"]:
                f.write(example["text"] + "\n")
                count += 1
                if count % 100000 == 0:
                    print(f"Wrote {count} examples...")
                    f.flush()

    print(f" Done. Total lines written: {count}")