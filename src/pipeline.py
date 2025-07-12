from audio_processing.transcriber import process_audio, transcribe_deepgram, transcribe_whisper
from chat_model.grammar_corrector import correct_transcript
from chat_model.chatbot import generate_chatbot
import os
from dotenv import load_dotenv
from pronunciation_model.pronunciation_model import (regular_score_pronunciation, evaluate_pronunciation,
                                                     transcribe_phonemes, generate_text_reference)
from google import genai
from jiwer import wer

load_dotenv()
gemini_key = os.getenv('GEMINI_KEY')

# Set the API key
client = genai.Client(api_key=gemini_key)

def process_audio_files(audio_directory, reference_directory):
    wer_scores = []

    for file_name in os.listdir(audio_directory):
        if file_name.endswith((".mp3", ".wav", ".m4a", ".flac")):
            input_audio = os.path.abspath(os.path.join(audio_directory, file_name))
            base_name = os.path.splitext(file_name)[0]
            ref_path = os.path.join(reference_directory, base_name + ".txt")

            if not os.path.exists(ref_path):
                print(f"❌ Skipping {file_name} — no reference transcript found.")
                continue

            print(f"\nProcessing: {input_audio}")

            # Transcribe audio
            response = transcribe_deepgram(input_audio)

            # Extract transcript from response
            hypothesis = response.to_dict()["results"]["channels"][0]["alternatives"][0]["transcript"]

            if hypothesis:
                with open(ref_path, "r", encoding="utf-8") as f:
                    reference = f.read().strip()

                error = wer(reference, hypothesis)
                wer_scores.append(error)

                # Correct grammar
                corrected_transcript = correct_transcript(hypothesis, client)

                print(f"Reference:  {reference}")
                print(f"Hypothesis: {hypothesis}")
                print(f"Corrected: {corrected_transcript}")
            else:
                print("Transcript failed.")

    if wer_scores:
        avg_wer = sum(wer_scores) / len(wer_scores)
        print(f"\n✅ Average WER across {len(wer_scores)} samples: {avg_wer:.3f}")
    else:
        print("\n❌ No transcriptions evaluated.")



if __name__ == "__main__":
    audio_dir = os.path.join("data", "audio")
    for file_name in os.listdir(audio_dir):
        if file_name.endswith((".mp3", ".wav", ".m4a", ".flac")):
            input_audio = os.path.abspath(os.path.join(audio_dir, file_name))
            print(f"Transcribing: {input_audio}")
            ground_text = generate_text_reference()
            print(f"Ground truth text: {ground_text}")
            phoneme = evaluate_pronunciation(input_audio, ground_text)
            print(f"Scores for {file_name}: {phoneme}")
            # score = regular_score_pronunciation(phoneme, ground_phoneme)
            # print(f"Pronunciation score for {file_name}: {score}")
    # reference_dir = os.path.join("data", "transcript_ref")
    # process_audio_files(audio_dir, reference_dir)
    # generate_chatbot(client, "Travel")