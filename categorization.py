
import re
import csv
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import defaultdict, Counter
from functools import lru_cache
from datetime import datetime
import os
import gc
import argparse

# Load spacy model
nlp = spacy.load("en_core_web_sm")

def extract_contact_info(doc, text):
    EMAIL = re.findall(r"[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    PHONE_NUMBER = re.findall(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}", text)
    TWITTER_HANDLER = re.findall(r"(?<!\w)@[\w]{1,15}(?![\w@.])", text)
    ADDRESS = re.findall(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s)+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Dr|Drive|Court|Ct)\b", text)
    return EMAIL + PHONE_NUMBER + TWITTER_HANDLER + ADDRESS

# Define category matchers
CATEGORY_EXTRACTORS = {
    "News": lambda doc, text: re.findall(
        r"(?i)\b(CNN|BBC|Reuters|The New York Times|The Guardian|AP News|Bloomberg|Biden|Trump|White House|UN|NATO|Ukraine|China|Russia|"
        r"TechCrunch|The Verge|Wired|Gizmodo|Ars Technica|Engadget|developer conference|software release|gadget review|AI|robotics|cloud computing|startup)\b", text
    ),
    "License, terms of use, copyright notices": lambda doc, text: [m.group() for m in re.finditer(r"(?i)(license|copyright|terms of use|all rights reserved)", text)],
    "Valid URLs": lambda doc, text: re.findall(r"(?i)\b((?:https?|ftp|www)[\.:/\\]?\w+(?:[\.:/\\]?\w+)*\.(?:com|net|org|edu|gov|io|co|info|biz|us|uk|de|jp|kr))", text),
    "Named individuals (non-news samples only)": lambda doc, text: [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
    "Promotional content (products, subscriptions, etc.)": lambda doc, text: re.findall(r"(?i)\b(buy now|limited offer|subscribe|click here|free trial|flash sale|redeem now|% off|exclusive offer)\b", text),
    "Contact info (address, email, phone, twitter, etc.)": extract_contact_info,
    "Code": lambda doc, text: re.findall(r"(?:def |class |function\s+|</?[a-z]+>)", text),
    "Configuration files": lambda doc, text: re.findall(r"(?:\[.*?\]|^[a-zA-Z0-9_]+\s*=\s*.+$)", text, re.MULTILINE),
    "Religious texts": lambda doc, text: re.findall(r"\b(Genesis|Bible|Quran|Torah|Psalm|Jesus|Allah|God|prophet)\b", text, re.IGNORECASE),
    "Donald Trump tweets and quotes": lambda doc, text: re.findall(r"(?i)(Donald Trump:|@realDonaldTrump|Make America Great Again|fake news|donald trump)", text),
    #"High entropy (UUIDs, base64 data)": lambda doc, text: re.findall(r"[A-Za-z0-9+/=]{30,}", text),
    #"Lists of named items (games, countries, etc.)": lambda doc, text: re.findall(r"(?:[A-Z][a-z]+,\s*){2,}[A-Z][a-z]+", text),
    #"Forum or Wiki entry": lambda doc, text: re.findall(r"(\bUser talk:|\bWikipedia:|\bTalk:|\b[Ww]iki\b|\bFAQ\b)", text),
    #"Log files and error reports": lambda doc, text: re.findall(r"(Traceback \(most recent call last\)|Exception|Error|\.log|\.trace)", text),
    #"Pseudonyms": lambda doc, text: re.findall(r"(?:aka|also known as|anno)\s+[A-Z][a-z]+", text),
    #"Web forms (menu items, instructions, etc.)": lambda doc, text: re.findall(r"<form.+?</form>|input type=", text, re.DOTALL),
    #"Tech News": lambda doc, text: re.findall(r"(?i)\b(TechCrunch|The Verge|Wired|Gizmodo|Ars Technica|Engadget|developer conference|software release|gadget review|AI|robotics|cloud computing|startup)\b", text),
    #"Lists of numbers (dates, sequences, etc.)": lambda doc, text: re.findall(r"(?:\d+[\s,])+\d+", text),
}

PII_CATEGORIES = {
    "Named individuals (non-news samples only)",
    "Contact info (address, email, phone, twitter, etc.)"
}

# EMAIL_PATTERN = r"^[\w.%+-]+@[A-Za-z0-9.-]+\.[a-zA-Z]{2,}$"

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sample-file', type=str, required=True, help="Path to extracted .txt sample file")
args = parser.parse_args()

sample_file = os.path.basename(args.sample_file)
sample_filename = sample_file.lower()
sample_model_name = os.path.splitext(sample_filename)[0]
resume_index_filename = f"resume_index_{sample_model_name}.txt"
summary_output_file = f"final_summary_{sample_model_name}.csv"
progress_csv_file = f"progress_temp_{sample_model_name}.csv"

# Parse sample blocks by skipping header lines and
#  grouping content until the next sample number
print("Loading sample blocks...")
sample_blocks = []
current_sample = []
in_sample = False
with open(args.sample_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        if re.match(r"^\d+:", line):
            in_sample = True
            if current_sample:
                full_text = " ".join(current_sample).strip()
                if full_text:
                    sample_blocks.append(full_text)
                current_sample = []
            continue

        if not in_sample:
            continue

        current_sample.append(line)

last_saved = None
if current_sample:
    full = " ".join(current_sample).strip()
    if full and full != last_saved:
        sample_blocks.append(full)
        last_saved = full

# Automatically choose original dataset based on the given sample filename
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "preprocess", "merged_dataset")
if "nano" in sample_filename:
    original_path = os.path.join(DATASET_DIR, "merged_nano.txt")
elif "gpt2" in sample_filename:
    original_path = os.path.join(DATASET_DIR, "merged_large.txt")
else:
    raise ValueError(f"Unrecognized sample in filename: {sample_filename}")


print("Indexing original dataset by category...")
with open(original_path, 'r', encoding='utf-8') as f:
    original_data = f.read().lower()
    original_lines = original_data.splitlines()

@lru_cache(maxsize=50000)
def exists_in_original_lines(info):
    info_lc = info.lower()
    return any(info_lc in line for line in original_lines)

resume_index = 0
category_to_sample_count = Counter()
category_to_total_count = Counter()
category_to_matches = defaultdict(list)
category_to_matches_set = defaultdict(set)

# Load resume index if exists
if os.path.exists(resume_index_filename):
    with open(resume_index_filename, "r") as f:
        resume_index = int(f.read().strip())

# Load prior match data if progress file exists
if os.path.exists(progress_csv_file):
    with open(progress_csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["Category"]
            sample_count = int(row["Unique Sample Count"])
            total_count = int(row["Total Match Count"])
            examples = row["Examples"].split('; ') if row["Examples"] else []

            category_to_sample_count[category] = sample_count
            category_to_total_count[category] = total_count
            category_to_matches[category] = examples
            category_to_matches_set[category] = set(examples)

print(f"Starting from sample {resume_index + 1} / {len(sample_blocks)}")

with nlp.select_pipes(disable=["parser", "tagger"]):
    for i in range(resume_index, len(sample_blocks)):
        text = sample_blocks[i]
        print(f"[{datetime.now().isoformat()}] ▶ Sample {i+1} / {len(sample_blocks)}")

        # For each category, find all matched terms that appear in original_data.
        # Count only once per category per sample, and store only unique examples.
        sample_matched_categories = set()
        for category, extractor in CATEGORY_EXTRACTORS.items():
            doc = nlp(text) if category in PII_CATEGORIES else None
            infos = set(extractor(doc, text))
            for info in infos:
                info_lc = info.lower()
                if any(info_lc in line for line in original_lines):
                    category_to_total_count[category] += 1
                    if info not in category_to_matches_set[category]:
                        category_to_matches[category].append(info)
                        category_to_matches_set[category].add(info)
                    sample_matched_categories.add(category)

        for cat in sample_matched_categories:
            category_to_sample_count[cat] += 1

        # Record intermediate index number into the index file
        with open(resume_index_filename, "w") as f:
            f.write(str(i + 1))

        # Intermediate .csv file
        with open(progress_csv_file, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Unique Sample Count", "Total Match Count", "Examples"])
            for category in CATEGORY_EXTRACTORS.keys():
                matches = category_to_matches.get(category, [])
                writer.writerow([
                    category,
                    category_to_sample_count.get(category, 0),
                    category_to_total_count.get(category, 0),
                    "; ".join(sorted(matches))
                ])

        gc.collect()

# Write final summary
with open(summary_output_file, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Category", "Unique Sample Count", "Total Match Count", "Examples"])
    for category in CATEGORY_EXTRACTORS.keys():
        matches = category_to_matches.get(category, [])
        writer.writerow([
            category,
            category_to_sample_count.get(category, 0),
            category_to_total_count.get(category, 0),
            "; ".join(sorted(matches))
        ])

    writer.writerow([]) # Empty row
    writer.writerow(["Total Samples", len(sample_blocks)])

    # Write PII counts
    total_pii_sample_count = sum(category_to_sample_count[cat] for cat in PII_CATEGORIES)
    total_pii_match_count = sum(category_to_total_count[cat] for cat in PII_CATEGORIES)
    writer.writerow(["Total PII Sample Matches", total_pii_sample_count])
    writer.writerow(["Total PII Match Occurrences", total_pii_match_count])

# Remove temp files
if os.path.exists(progress_csv_file):
    os.remove(progress_csv_file)
if os.path.exists(resume_index_filename):
    os.remove(resume_index_filename)

print("\nSummary CSV complete:", summary_output_file)
