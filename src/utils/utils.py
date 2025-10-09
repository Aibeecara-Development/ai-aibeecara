import re
import base64
import torch
import io

def clean_text(text: str) -> str:
    # Replace newlines and slashes with spaces
    text = text.replace("\n", " ").replace("/", " ")

    # Regex: keep alphanumeric, spaces, and essential punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\.,\?!;:'\"()\-\u2026]", "", text)

    # Normalize spaces (remove double spaces)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def serialize_waveform(waveform: torch.Tensor) -> str:
    """Convert torch tensor to base64 string."""
    buf = io.BytesIO()
    torch.save(waveform, buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def deserialize_waveform(waveform_str: str) -> torch.Tensor:
    """Convert base64 string back to torch tensor."""
    buf = io.BytesIO(base64.b64decode(waveform_str.encode("utf-8")))
    return torch.load(buf)