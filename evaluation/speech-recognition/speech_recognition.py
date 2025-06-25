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
from datasets import load_dataset

load_dotenv()
deepgram_key = os.getenv('DEEPGRAM_KEY')

def run_whisper(input_file):
    command = [
        "whisper",
        input_file,
        "--model", "tiny",
        "--output_dir", os.path.abspath("output/whisper"),
        "--output_format", "srt",
        "--language", "en",
        "--word_timestamps", "True",
        "--max_line_width", "25",
        "--max_line_count", "2"
    ]

    try:
        subprocess.run(command, check=True)
        print(f"{input_file} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    except FileNotFoundError:
        print("Whisper CLI not found")

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
        return transcript.lower()
    except Exception as e:
        print(f"Exception while processing {audio_path}: {e}")
        return None

# Track WERs
wer_scores = []

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

        hypothesis = run_deepgram_and_return_transcript(input_audio)

        if hypothesis:
            with open(ref_path, "r", encoding="utf-8") as f:
                reference = f.read().strip().lower()

            error = wer(reference, hypothesis)
            wer_scores.append(error)

            print(f"Reference:  {reference}")
            print(f"Hypothesis: {hypothesis}")
            print(f"WER: {error:.3f}")
        else:
            print("Transcript failed.")

# Average WER
if wer_scores:
    avg_wer = sum(wer_scores) / len(wer_scores)
    print(f"\n✅ Average WER across {len(wer_scores)} samples: {avg_wer:.3f}")
else:
    print("\n❌ No transcriptions evaluated.")




