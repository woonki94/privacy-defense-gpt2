import os
import pandas as pd
from email.parser import Parser
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from cleantext import clean
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
nltk.download('punkt_tab', force = True)
nltk.download('stopwords')

def load_emails(base_path):
    emails = []
    for user_dir in os.listdir(base_path):
        user_path = os.path.join(base_path, user_dir)
        if os.path.isdir(user_path):
            for folder in os.listdir(user_path):
                folder_path = os.path.join(user_path, folder)
                if os.path.isdir(folder_path):
                    for email_file in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, email_file)
                        if os.path.isfile(file_path):
                            with open(file_path, "r", encoding="latin-1") as file:
                                content = file.read()
                                emails.append(content)
    return emails



def parse_email(raw_email):
    parser = Parser()
    email = parser.parsestr(raw_email)
    return {
        "from": email.get("From"),
        "to": email.get("To"),
        "subject": email.get("Subject"),
        "date": email.get("Date"),
        "cc": email.get("Cc"),
        "bcc": email.get("Bcc"),
        "body": email.get_payload()
    }

def remove_disclaimer(text):
    # Simple pattern to catch common disclaimer phrases
    disclaimer_patterns = [
        r"(?i)this message contains confidential information.*",
        r"(?i)if you are not the intended recipient.*",
        r"(?i)please notify the sender immediately.*",
        r"(?i)delete this e[-]?mail from your system.*",
    ]
    for pattern in disclaimer_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    return text


def remove_reply_chain(text):
    cleaned = re.sub(
        r'-{5}Original Message-{5}.*?\n\s*\n',
        '',
        text,
        flags=re.DOTALL
    )
    return cleaned.strip()

def remove_symbols(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

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

def clean_body(text):
    if not isinstance(text, str):
        return ""
    text = remove_disclaimer(text)
    text = remove_reply_chain(text)
    text = cleantext(text)
    text = remove_symbols(text)
    words = word_tokenize(text)

    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)


if __name__ == '__main__':
    email_dir = os.path.join(BASE_DIR,"dataset", "maildir")
    email_list = load_emails(email_dir)
    parsed_emails = [parse_email(e) for e in email_list]
    df = pd.DataFrame(parsed_emails)

    df["body"] = df["body"].apply(clean_body)
    df["subject"] = df["subject"].apply(clean_body)
    df["date"] = df["date"].apply(clean_body)


    output_path = os.path.join(BASE_DIR,"cleaned_dataset/cleaned_emails.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        for index, row in df.iterrows():
            f.write(f"Email #{index + 1}\n")
            for column, value in row.items():
                f.write(f"{column}: {value}\n")

    



