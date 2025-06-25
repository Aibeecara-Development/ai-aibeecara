import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from jiwer import wer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from gector import GECToR, predict, load_verb_dict
from google import genai
from google.genai import types
import time

# Load the .env file
load_dotenv()
gemini_key = os.getenv('GEMINI_KEY')

# Set the API key
client = genai.Client(api_key=gemini_key)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
ds = load_dataset("agentlans/grammar-correction", split="validation")
sample_size = 20
inputs = ds["input"][:sample_size]
targets = ds["output"][:sample_size]

# Load GECToR model and tokenizer
model_id = 'gotutiyan/gector-roberta-large-5k'
original_resize = PreTrainedModel.resize_token_embeddings

def patched_resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of=None, mean_resizing=False):
    return original_resize(self, new_num_tokens, pad_to_multiple_of=pad_to_multiple_of, mean_resizing=False)

PreTrainedModel.resize_token_embeddings = patched_resize_token_embeddings
model = GECToR.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
encode, decode = load_verb_dict('data/verb-form-vocab.txt')
corrected = predict(
    model, tokenizer, inputs,
    encode, decode,
    keep_confidence=0.0,
    min_error_prob=0.0,
    n_iteration=5,
    batch_size=2,
)

# Compute WER for GECToR
gector_wer = wer(targets, corrected)
print(f"GECToR WER: {gector_wer:.4f}")

instructions = """
You are a helpful grammar correction assistant. Given a sentence, correct any grammatical errors and return the corrected version.
Only provide the revised text without any additional text.
"""

client = genai.Client(api_key=gemini_key)

# gemini-2 predictions
gemini_preds = []
for i, input_text in enumerate(tqdm(inputs)):
    if i > 0 and i % 10 == 0:
        print("Sleeping for 60 seconds to respect rate limit...")
        time.sleep(60)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=input_text,
        config=types.GenerateContentConfig(
            system_instruction=instructions
        )
    )
    gemini_preds.append(response.text)

# Compute WER for gemini baseline
gemini_wer = wer(targets, gemini_preds)
print(f"gemini-2 (Simulated) WER: {gemini_wer:.4f}")
