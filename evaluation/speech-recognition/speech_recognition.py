import subprocess
import os
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
from jiwer import wer
from dotenv import load_dotenv
import json
import whisper
from whisper import transcribe
from datasets import load_dataset

load_dotenv()
deepgram_key = os.getenv('DEEPGRAM_KEY')

def run_whisper(input_file):
    model = whisper.load_model("base")
    result = transcribe(model=model, audio=input_file)
    transcribed_text_whisper = result["text"]
    return transcribed_text_whisper



def run_deepgram(audio):
    try:
        deepgram = DeepgramClient(deepgram_key)

        with open(audio, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
        )

        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

        # Save response to output directory
        base_name = os.path.splitext(os.path.basename(audio))[0]
        output_path = os.path.join("output/deepgram", f"{base_name}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(response.to_dict(), f, indent=4)

        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Exception: {e}")

# for file_name in os.listdir("data"):
#     if file_name.endswith((".mp3", ".wav", ".m4a")):
#         input_audio = os.path.abspath(os.path.join("data", file_name))
#         print("===========================================================")
#         print(f"Processing: {input_audio}")
#         run_deepgram(input_audio)
#         # run_whisper(input_audio)

def run_deepgram_and_return_transcript(audio_path):
    try:
        deepgram = DeepgramClient(deepgram_key)
        with open(audio_path, "rb") as f:
            buffer_data = f.read()
        payload: FileSource = {"buffer": buffer_data}
        options = PrerecordedOptions(model="nova-3", smart_format=True)
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

        # Parse and return transcript
        transcript = response.to_dict()["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcript
    except Exception as e:
        print(f"Exception while processing {audio_path}: {e}")
        return None

# Track WERs
wer_scores_deepgram = []
wer_scores_whisper = []

# Loop through files
for i, file_name in enumerate(os.listdir("data")):
    if file_name.endswith((".mp3", ".wav", ".m4a", ".flac")):
        input_audio = os.path.abspath(os.path.join("data", file_name))
        base_name = os.path.splitext(file_name)[0]
        ref_path = os.path.join("data", "transcript_ref", base_name + ".txt")

        if not os.path.exists(ref_path):
            print(f"❌ Skipping {file_name} — no reference transcript found.")
            continue

        print("===========================================================")
        print(f"\n=== Sample {i + 1} ===")
        print(f"Processing: {input_audio}")

        hypothesis_deepgram = run_deepgram_and_return_transcript(input_audio)
        print("Done transcribing with Deepgram.")
        hypothesis_whisper = run_whisper(input_audio)
        print("Done transcribing with Whisper.")

        if hypothesis_deepgram is not None and hypothesis_whisper is not None:
            with open(ref_path, "r", encoding="utf-8") as f:
                reference = f.read().strip()

            error_deepgram = wer(reference, hypothesis_deepgram)
            wer_scores_deepgram.append(error_deepgram)

            print(f"\nSample {i + 1} - Deepgram:")

            print(f"Reference:  {reference}")
            print(f"Hypothesis: {hypothesis_deepgram}")
            print(f"WER: {error_deepgram:.3f}")

            error_whisper = wer(reference, hypothesis_whisper)
            wer_scores_whisper.append(error_whisper)
            print(f"\nSample {i + 1} - Whisper:")
            print(f"Reference:  {reference}")
            print(f"Hypothesis: {hypothesis_whisper}")
            print(f"WER: {error_whisper:.3f}")
        else:
            print("Transcript failed.")

# Average WER
# if wer_scores_deepgram and wer_scores_whisper:
#     avg_wer = sum(wer_scores) / len(wer_scores)
#     print(f"\n✅ Average WER across {len(wer_scores)} samples: {avg_wer:.3f}")
# else:
#     print("\n❌ No transcriptions evaluated.")




