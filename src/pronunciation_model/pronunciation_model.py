import whisperx
from g2p_en import G2p
import subprocess
import os
import sys
from pydub import AudioSegment
from phonemizer import phonemize
from transformers import pipeline
import jiwer

# Load the model
pipe = pipeline(model="vitouphy/wav2vec2-xls-r-300m-timit-phoneme")

def convert_mp3_to_wav(mp3_path):
    sound = AudioSegment.from_file(mp3_path, format="mp3")
    wav_path = mp3_path.replace(".mp3", ".wav")
    sound.export(wav_path, format="wav")
    return wav_path

def pronunciation_to_phonemes(audio_file):
    """Convert pronunciation text to phonemes using the pipeline."""
    # Assuming pronunciation is a string of text
    phoneme_output = pipe(audio_file, chunk_length_s=10, stride_length_s=(4, 2))
    return phoneme_output['text']

def phonemize_text(text, language='en-us'):
    """Phonemize the input text into phonemes."""
    phonemes = phonemize(
        text,
        language=language,
        backend='espeak',
        strip=True,
        preserve_punctuation=True,
    )
    return phonemes

def count_pronunciation_score(hypothesis, reference):
    # Convert to lower case for case-insensitive comparison
    hypothesis = hypothesis.lower()
    reference = reference.lower()

    # Calculate the phoneme error rate
    per_score = jiwer.cer(reference, hypothesis)

    return 1 - per_score

# def evaluate_pronunciation(input_audio, reference_text):
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     repo_path = os.path.join(current_dir, "Goodness-of-Pronounciation-main")
#     main_py = os.path.join(repo_path, "main.py")
#
#     print("Using Python:", sys.executable)
#
#     converted_audio = convert_mp3_to_wav(input_audio)
#
#     result = subprocess.run(
#         [sys.executable, main_py, converted_audio, reference_text],
#         capture_output=True,
#         text=True,
#         cwd=repo_path
#     )
#
#     print("STDOUT:\n", result.stdout)
#     print("STDERR:\n", result.stderr)
#
#     return result.stdout
#
# def transcribe_phonemes(input_audio):
#     # input_audio = "input/audio.wav"
#     device = "cpu"
#     batch_size = 8
#     language = "en"
#     compute_type = "int8"
#
#     print("Loading WhisperX model...")
#
#     model = whisperx.load_model("medium", device, language=language, compute_type=compute_type)
#
#     print("Transcribing audio...")
#
#     audio = whisperx.load_audio(input_audio)
#     result = model.transcribe(audio, batch_size=batch_size)
#
#     print("Transcription complete. Aligning segments...")
#
#     model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
#     aligned_result = whisperx.align(result["segments"], model_a, metadata, input_audio, device)
#
#     # phoneme_list = []
#     # for segment in aligned_result["segments"]:
#     #     for phoneme in segment.get("phonemes", []):
#     #         phoneme_list.append(phoneme)
#     #
#     # return phoneme_list
#     return aligned_result
#
# def g2p_from_user_history(history_log):
#     g2p = G2p()
#     all_phonemes = []
#
#     for role, message in history_log:
#         if role == "user":
#             phoneme_list = g2p(message)
#             phoneme_list = [ph for ph in phoneme_list if ph.isalpha()]
#             all_phonemes.append({
#                 "text": message,
#                 "phonemes": phoneme_list
#             })
#
#     return all_phonemes
#
# def generate_text_reference():
#     # g2p = G2p()
#     example_text = """
#     and we want to highlight those and bring that to where we can have
#     a supportive system in place so nutrient recycling our nutrients back
#     on to the land to rejuvenate the soils that have been depleted by
#     plantation agriculture over a long period of time
#     """
#     phoneme_list = example_text
#     # phoneme_list = [ph for ph in phoneme_list if ph.isalpha()]
#     return phoneme_list
#
# def regular_score_pronunciation(aligned_phonemes, ref_phoneme_sequence):
#     matched = 0
#     total = len(ref_phoneme_sequence)
#
#     ref_idx = 0
#     align_idx = 0
#
#     while ref_idx < total and align_idx < len(aligned_phonemes):
#         ref_ph = ref_phoneme_sequence[ref_idx].upper()
#         align_ph = aligned_phonemes[align_idx]["text"].upper()
#
#         if ref_ph == align_ph:
#             matched += 1
#             ref_idx += 1
#             align_idx += 1
#         else:
#             # Either skip the aligned phoneme or assume mispronunciation
#             align_idx += 1
#
#     return round((matched / total) * 100, 2) if total > 0 else 0.0


if __name__ == "__main__":
    # Example usage
    input_audio = "data/2025_07_08_14_09_18.mp3"
    hypothesis_phoneme = pronunciation_to_phonemes(input_audio)
    print("Audio pronunciation to phonemes:")
    print(hypothesis_phoneme)
    reference_text = "and we want to highlight those and bring that to where we can have a supportive system in place so nutrient recycling our nutrients back on to the land to rejuvenate the soils that have been depleted by plantation agriculture over a long period of time"
    reference_phoneme = phonemize_text(reference_text)
    print("Phonemized reference text:")
    print(reference_phoneme)
    print("Counting phoneme error rate:")
    print(count_pronunciation_score(hypothesis_phoneme, reference_phoneme))
    # result = evaluate_pronunciation(input_audio, reference_text)
    # print(result)