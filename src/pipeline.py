from audio_processing.transcriber import process_audio, transcribe_deepgram, transcribe_whisper
from chat_model.grammar_corrector import correct_transcript
import os
from dotenv import load_dotenv
from google import genai
from jiwer import wer
from chat_model.scoring.score_model import evaluate_transcription
from pronunciation_model.pronunciation_model import evaluate_pronunciation, highlight_wrong_words

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

            try:
                # Transcribe audio
                response = transcribe_deepgram(input_audio)

                # Extract transcript
                hypothesis = response.to_dict()["results"]["channels"][0]["alternatives"][0]["transcript"]

                if not hypothesis:
                    raise ValueError("Empty hypothesis returned.")

                with open(ref_path, "r", encoding="utf-8") as f:
                    reference = f.read().strip()

                error = wer(reference, hypothesis)
                wer_scores.append(error)

                # Correct grammar
                corrected_transcript = correct_transcript(hypothesis, client)

                print(f"Reference:  {reference}")
                print(f"Hypothesis: {hypothesis}")
                print(f"Corrected: {corrected_transcript}")

            except Exception as e:
                print(f"❌ Error processing {file_name}: {e}")

    if wer_scores:
        avg_wer = sum(wer_scores) / len(wer_scores)
        print(f"\n✅ Average WER across {len(wer_scores)} samples: {avg_wer:.3f}")
    else:
        print("\n❌ No transcriptions evaluated.")



# if __name__ == "__main__":
#     incorrect_transcript = "It was a real nice day today. Can I have you’re coat? We should contact they’re friend."
#     transcription_score, corrected_transcript, grammar_explanation = evaluate_transcription(incorrect_transcript)
#     print(f"Original Transcript: {incorrect_transcript}")
#     print(f"Transcription Score: {transcription_score}")
#     print(f"Corrected Transcript: {corrected_transcript}")
#     print(f"Grammar Explanation: {grammar_explanation}")
#     audio_directory = "data/audio/Recording_14.wav"
#     example_text = """
#     and we want to highlight those and bring that to where we can have
#     a supportive system in place so nutrient recycling our nutrients back
#     on to the land to rejuvenate the soils that have been depleted by
#     plantation agriculture over a long period of time
#     """
#     hypothesis_phoneme, reference_phoneme, score = evaluate_pronunciation(audio_directory, example_text)
#     print(f"Pronunciation Score: {score}")
#     print(f"Hypothesis Phonemes: {hypothesis_phoneme}")
#     print(f"Reference Phonemes: {reference_phoneme}")
#
#     wrong_words = highlight_wrong_words(hypothesis_phoneme, reference_phoneme, example_text)
#
#     for w in wrong_words:
#         print(w)

