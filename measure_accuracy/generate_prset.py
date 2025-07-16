from datasets import load_dataset
import nltk
nltk.download('punkt_tab')
from tqdm import tqdm
import os
NUM_SAMPLES = 100

def split_first_sentence(text):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) >= 2:
        return sentences[0], ' '.join(sentences[1:])
    elif len(sentences) == 1:
        return sentences[0], ""
    else:
        return "", ""

def generate_ref_prompt(prom_file_name, ref_file_name):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Prepare lists for prompts and references
    prompts = []
    references = []

    for example in tqdm(dataset, desc="Processing WikiText-103"):
        if len(prompts) >= NUM_SAMPLES:
            break

        text = example["text"].strip()
        if text and not text.startswith(" = "):  # Skip headers and empty lines
            prompt, reference = split_first_sentence(text)
            if prompt and reference:
                prompts.append(prompt)
                references.append(reference)

    # Save to files
    with open(prom_file_name, "w", encoding="utf-8") as f:
        f.write("\n".join(prompts))
    with open(ref_file_name, "w", encoding="utf-8") as f:
        f.write("\n".join(references))



def generate_ref_prompt_from_file(data_file, prom_file_name, ref_file_name):
    prompts = []
    references = []

    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing pii.txt"):
        if len(prompts) >= NUM_SAMPLES:
            break

        text = line.strip()
        if text and not text.startswith(" = "):  # Optional: skip headers or special formatting
            prompt, reference = split_first_sentence(text)
            if prompt and reference:
                prompts.append(prompt)
                references.append(reference)

    with open(prom_file_name, "w", encoding="utf-8") as f:
        f.write("\n".join(prompts))
    with open(ref_file_name, "w", encoding="utf-8") as f:
        f.write("\n".join(references))


def ensure_files_exists(file_paths):
    for path in file_paths:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

        # Create the file if it doesn't exist
        if not os.path.exists(path):
            with open(path, 'w') as f:
                pass  # creates an empty file


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prom_file_name = os.path.join(base_dir,"ref","prompts.txt")
    ref_file_name = os.path.join(base_dir,"ref","references.txt")
    prom_pii = os.path.join(base_dir,"ref","prompts_pii.txt")
    ref_pii = os.path.join(base_dir,"ref","references_pii.txt")
    data_file = os.path.join(base_dir, '..', 'preprocess','merged_dataset', 'merged_large.txt')

    file_lists= [prom_file_name,ref_file_name,prom_pii,ref_pii]
    ensure_files_exists(file_lists)

    generate_ref_prompt(prom_file_name, ref_file_name)
    generate_ref_prompt_from_file(data_file, prom_pii, ref_pii)

