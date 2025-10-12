from jiwer import wer
from happytransformer import HappyTextToText, TTSettings
from lexicalrichness import LexicalRichness
from transformers import pipeline
import pyphen
import ast
import os
from dotenv import load_dotenv
from google import genai
from src.chat_model.chatbot import grammar_correction
import requests

BASE_URL = "https://farrel-dr-aibeecara-models-2.hf.space"

# TODO: These models can be deployed on future APIs
# happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
# cefr_classifier = pipeline("text-classification", model="AbdulSami/bert-base-cased-cefr")

load_dotenv()
gemini_key = os.getenv("GEMINI_KEY")
client = genai.Client(api_key=gemini_key)

def evaluate_transcription(transcription):
    """Evaluate the transcription for grammar issues."""
    # args = TTSettings(num_beams=5, min_length=1)

    # Add the prefix "grammar: " before each input
    data = {"text": transcription}
    result = requests.post(f"{BASE_URL}/grammar-correct", json=data).json()['text']

    wer_score = wer(result, transcription)

    corrected_text = result

    transcription_score = 1 - wer_score

    grammar_explanation, tense_used = grammar_correction(client, transcription)

    return transcription_score, corrected_text, grammar_explanation, tense_used

def evaluate_vocabulary(transcription):
    lex = LexicalRichness(transcription)
    mtld_score = lex.mtld(threshold=0.72) * 0.01  # Scale to a 0-1 range
    if mtld_score + 0.3 > 1:
        mtld_score = 1.0
    else:
        mtld_score += 0.3
    print(f"Vocabulary score: {mtld_score:.2f}")
    return mtld_score

def evaluate_vocabulary_cefr(transcription: str) -> str:
    """Evaluate the vocabulary of the transcription based on CEFR levels."""
    data = {"text": transcription}
    result = requests.post(f"{BASE_URL}/cefr-vocab", json=data).json()['text']  # returns "A1"..."C2"
    parsed = ast.literal_eval(result)
    label = parsed[0]["label"]
    print(f"CEFR vocabulary score: {label}")
    return label

def evaluate_pause(deepgram_response):
    data = deepgram_response["results"]["channels"][0]["alternatives"][0]
    words = data["words"]

    pause_threshold = 1.0
    long_pauses = 0
    pause_between_words = []

    corrected_transcript = []

    for i in range(len(words) - 1):
        current_word = words[i]["punctuated_word"]
        current_end = words[i]["end"]
        next_start = words[i + 1]["start"]

        corrected_transcript.append(current_word)

        pause_duration = next_start - current_end
        if pause_duration > pause_threshold:
            long_pauses += 1
            pause_between_words.append({
                "start_word": words[i]["word"],
                "end_word": words[i + 1]["word"],
                "duration": pause_duration
            })
            corrected_transcript.append("...")  # insert pause marker

    # Add the last word
    corrected_transcript.append(words[-1]["punctuated_word"])

    # Rebuild transcript
    new_transcript = " ".join(corrected_transcript)

    score = 1.0 if long_pauses == 0 else 1 - (long_pauses / len(words))

    return {
        "score": score,
        "pause_between_words": pause_between_words,
        "pause_transcript": new_transcript
    }

def evaluate_stutter(deepgram_response):
    words_list = deepgram_response["results"]["channels"][0]["alternatives"][0]["words"]
    words = [w["word"].lower() for w in words_list]
    count = 0
    i = 0
    stuttered_phrases = []

    while i < len(words):
        if words[i] in ["um", "uh", "mhmm", "mm-mm", "uh-uh", "uh-huh", "nuh-uh"]:
            count += 1
            stuttered_phrases.append(words[i])
        i += 1

    print(f"Number of stutters: {count}")
    print("stuttered phrases:", stuttered_phrases)

    score = 1.0
    if count == 0:
        return score, stuttered_phrases
    else:
        return 1 - (count / len(words)), stuttered_phrases


def calculate_speech_rates(deepgram_response) -> dict:
    dic = pyphen.Pyphen(lang='en')
    words_data = deepgram_response["results"]["channels"][0]["alternatives"][0]["words"]
    duration = deepgram_response["metadata"]["duration"]

    # Extract words
    words = [w["word"] for w in words_data if w.get("word")]
    total_words = len(words)

    # Count syllables
    total_syllables = 0
    for word in words:
        hyphenated = dic.inserted(word)
        if hyphenated:
            total_syllables += len(hyphenated.split("-"))
        else:
            total_syllables += 1  # fallback

    minutes = duration / 60 if duration > 0 else 1

    wpm = total_words / minutes
    spm = total_syllables / minutes

    return {
        "wpm": int(wpm),
        "spm": int(spm),
    }

if __name__ == "__main__":
    evaluate_vocabulary_cefr("It was a real nice day today. Can I have you’re coat? We should contact they’re friend.")
