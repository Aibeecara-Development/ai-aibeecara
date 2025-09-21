import re

def clean_text(text: str) -> str:
    # Replace newlines and slashes with spaces
    text = text.replace("\n", " ").replace("/", " ")

    # Regex: keep alphanumeric, spaces, and essential punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\.,\?!;:'\"()\-\u2026]", "", text)

    # Normalize spaces (remove double spaces)
    text = re.sub(r"\s+", " ", text).strip()

    return text
