import re
import pandas as pd
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from cleantext import clean

# Ensure necessary resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Clean wikitext formatting
def clean_wikitext_markup(text):
    text = re.sub(r'=\s?.*?\s?=', '', text)  # remove section headers like = Title =
    text = re.sub(r'\[\[|\]\]', '', text)    # remove [[ ]] link markup
    text = re.sub(r"''+", '', text)          # remove ''italic'' and '''bold'''
    text = text.replace('<unk>', '')         # remove <unk> tokens
    return text.strip()

# Use clean-text + stopword removal
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = clean_wikitext_markup(text)
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        lower=True,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_punct=True,
    )
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

def main():
    # Load Wikitext dataset
    print("Loading Wikitext dataset...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # We'll just process the first N samples to keep it fast — adjust as needed
    N = 1000
    records = []
    for i in range(N):
        raw = ds[i]['text']
        cleaned = clean_text(raw)
        records.append({
            'raw_text': raw,
            'cleaned_text': cleaned
        })

    # Create DataFrame
    df = pd.DataFrame(records)
    print(df.head())

    # Print one sample like in your Enron example
    email_row = df.iloc[0]
    for column, value in email_row.items():
        print(f"{column}:\n{value}\n")

if __name__ == "__main__":
    main()
