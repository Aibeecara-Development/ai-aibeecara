from pydub import AudioSegment
from phonemizer import phonemize
from nltk.tokenize import SyllableTokenizer
from nltk.tokenize import word_tokenize
from transformers import pipeline
import random
import jiwer
import textdistance

# Load the model
pipe = pipeline(model="vitouphy/wav2vec2-xls-r-300m-timit-phoneme")

PHONEME_MAP = {
    "a": ["a", "ɑ", "æ", "ʌ", "ɒ"],              # open mouth vowels
    "fv": ["f", "v"],                            # teeth on lip
    "ie": ["i", "ɪ", "e", "ɛ", "j"],             # smile vowels + /j/
    "l": ["l"],                                  # tongue up
    "mb": ["m", "b", "p"],                       # closed lips
    "o": ["o", "ɔ", "u", "ʊ"],                   # round lips
    "th": ["θ", "ð"],                            # tongue between teeth
    "neutral": [".", ",", "?", "!"]              # punctuation
}

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

def phonemize_text(text, language='en-us', preserve_punctuation=True):
    """Phonemize the input text into phonemes."""
    phonemes = phonemize(
        text,
        language=language,
        backend='espeak',
        strip=True,
        preserve_punctuation=preserve_punctuation,
    )
    return phonemes

def get_word_phonemes(text: str) -> list[tuple[str, str]]:
    """
    Split transcript into words and get phonemes for each word.
    Returns list of (word, phoneme_string).
    """
    words = text.split()
    phonemes = phonemize_text(text, language="en-us", preserve_punctuation=False).split()
    return list(zip(words, phonemes))

def highlight_wrong_words(hypothesis_phonemes: str, reference_phonemes: str, reference_text: str, top_k: int = 3) -> list[dict]:
    """
    Highlight up to top_k words with the worst phoneme mismatch between
    hypothesis and reference.
    """

    # Split reference into word-level phonemes
    ref_words = get_word_phonemes(reference_text)  # [(word, phoneme_str), ...]

    # Now, split hypothesis phonemes proportionally across words (since it's one long string)
    hyp_phonemes = hypothesis_phonemes.strip()
    total_len = sum(len(p) for _, p in ref_words)

    hyp_word_phons = []
    idx = 0
    for word, ref_phon in ref_words:
        share = max(1, int(len(hyp_phonemes) * (len(ref_phon) / total_len)))
        hyp_word_phons.append((word, hyp_phonemes[idx:idx + share]))
        idx += share

    # Compare word by word
    results = []
    for (ref_word, ref_phon), (hyp_word, hyp_phon) in zip(ref_words, hyp_word_phons):
        dist = textdistance.levenshtein.normalized_distance(ref_phon, hyp_phon)
        results.append({
            "reference_word": ref_word,
            "hypothesis_word": hyp_word,
            "ref_phonemes": ref_phon,
            "hyp_phonemes": hyp_phon,
            "error_score": round(dist, 3)
        })

    # Sort and return worst offenders
    results = sorted(results, key=lambda x: x["error_score"], reverse=True)[:top_k]
    return results

def count_pronunciation_score(hypothesis, reference):
    # Convert to lower case for case-insensitive comparison
    hypothesis = hypothesis.lower()
    reference = reference.lower()

    # Calculate the phoneme error rate
    per_score = jiwer.cer(reference, hypothesis)

    actual_score = 1 - per_score

    if actual_score < 0.30:
        return 0.40
    elif actual_score >= 0.70:
        return 1.0
    else:
        return 0.40 + (actual_score - 0.30) * (0.60 / 0.40)

def evaluate_pronunciation(input_audio, reference_text):
    """Evaluate pronunciation by comparing audio to reference text."""
    # Transcribe the audio to phonemes
    hypothesis_phoneme = "".join(pronunciation_to_phonemes(input_audio).split())

    # Phonemize the reference text
    reference_phoneme = "".join(phonemize_text(reference_text, preserve_punctuation=False).split())

    # Count the pronunciation score
    score = count_pronunciation_score(hypothesis_phoneme, reference_phoneme)

    return hypothesis_phoneme, reference_phoneme, score

def categorize_phoneme(phoneme: str) -> str:
    """Map a phoneme (syllable string) to a mouth movement category."""
    # Look inside the string for any matching symbol
    for category, phon_list in PHONEME_MAP.items():
        for symbol in phon_list:
            if symbol in phoneme:  # substring match
                return category
    return "neutral"

def tokenize_syllables(text, speed: float = 1.0):
    """Tokenize the input text into syllables with adjustable speed multiplier."""
    ssp = SyllableTokenizer()
    words = word_tokenize(text)
    syllables_in_sentence = [ssp.tokenize(word) for word in words]
    result = []
    current_time = 0.476 * speed  # initial break scaled

    syllable_arr = []

    for group in syllables_in_sentence:
        for syllable in group:
            syllable_arr.append(syllable)

    phonemes = phonemize_text(syllable_arr)

    for syll in phonemes:
        # Handle punctuation directly
        if syll in [".", "?", "!"]:
            duration = 0.7
            category = "neutral"
        elif syll == ",":
            duration = 0.3
            category = "neutral"
        else:
            if phonemes:
                category = categorize_phoneme(syll)
            else:
                category = "neutral"
            duration = random.uniform(0.172, 0.240)

        duration *= speed

        result.append({
            "syllable": syll,
            "category": category,
            "start_time": round(current_time, 3),
            "duration": round(duration, 3),
            "end_time": round(current_time + duration, 3)
        })
        current_time += duration

    return result

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


# df = pd.read_csv("data/speech_emotions.csv")
#
# # Pick 15 random rows
# samples = df
#
# scores = []
# results = []
# count = 1
#
# for _, row in samples.iterrows():
#     set_id = row["set_id"]
#     reference_text = row["text"]
#
#     # Path to the folder containing wav files
#     folder_path = os.path.join("files", str(set_id))
#
#     # Get all wav files in the folder
#     wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
#
#     if not wav_files:
#         print(f"No .wav files found in folder: {folder_path}")
#         continue
#
#     # Pick a random wav file
#     wav_file = random.choice(wav_files)
#     wav_path = os.path.join(folder_path, wav_file)
#     print(f"Processing file {count}: {wav_path}")
#
#     # Hypothesis phonemes from audio
#     hypothesis_phoneme = pronunciation_to_phonemes(wav_path)
#     hypothesis_phoneme = "".join(hypothesis_phoneme.split())
#     print(f"Reference text {count}: {reference_text}")
#     print(f"Hypothesis phoneme {count}: {hypothesis_phoneme}")
#
#     # Reference phonemes from text
#     reference_phoneme = phonemize_text(reference_text, preserve_punctuation=False)
#     reference_phoneme = "".join(reference_phoneme.split())
#     print(f"Reference phoneme {count}: {reference_phoneme}")
#
#     # Pronunciation score
#     score = round(count_pronunciation_score(hypothesis_phoneme, reference_phoneme), 4)
#     scores.append(score)
#     print(f"Pronunciation score {count}: {score}\n")
#     print("-----------------------------------\n")
#
#     # Save results for Excel
#     results.append({
#         "wav_path": wav_path,
#         "reference_text": reference_text,
#         "hypothesis_phoneme": hypothesis_phoneme,
#         "reference_phoneme": reference_phoneme,
#         "pronunciation_score": score
#     })
#
#     count += 1
#
# # Mean score
# mean_score = round(np.mean(scores), 4) if scores else None
#
# print("Pronunciation scores:", scores)
# print("Mean pronunciation score:", mean_score)
#
# # Save to CSV
# results_df = pd.DataFrame(results)
# results_df.to_csv("data/pronunciation_results.csv", index=False)
# print("Results saved to data/pronunciation_results.csv")

# input_text = """
# In the heart of every forest, a hidden world thrives among the towering trees. Trees,
# those silent giants, are more than just passive observers of nature's drama; they are
# active participants in an intricate dance of life.
# """
#
# tokens = tokenize_syllables(input_text)
# sum = 0
# for entry in tokens:
#     print(entry)
#     sum += entry["duration"]
# print(f"Total duration: {sum} seconds")
