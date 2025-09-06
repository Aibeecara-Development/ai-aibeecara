from jiwer import wer
from happytransformer import HappyTextToText, TTSettings
from lexicalrichness import LexicalRichness
from transformers import pipeline
import pyphen

# TODO: These models can be deployed on future APIs
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
cefr_classifier = pipeline("text-classification", model="AbdulSami/bert-base-cased-cefr")

def evaluate_transcription(transcription):
    """Evaluate the transcription for grammar issues."""
    args = TTSettings(num_beams=5, min_length=1)

    # Add the prefix "grammar: " before each input
    result = happy_tt.generate_text(f"grammar: {transcription}.", args=args)

    print(f"trasncription: {transcription}")

    wer_score = wer(result.text, transcription)

    print(result.text)

    corrected_text = result.text

    transcription_score = 1 - wer_score

    print(transcription_score)

    return transcription_score, corrected_text

def evaluate_vocabulary(transcription):
    lex = LexicalRichness(transcription)
    mtld_score = lex.mtld(threshold=0.72) * 0.01  # Scale to a 0-1 range
    if mtld_score + 0.3 > 1:
        mtld_score = 1.0
    else:
        mtld_score += 0.3
    print(f"Vocabulary score: {mtld_score:.2f}")
    return mtld_score

def evaluate_vocabulary_cefr(transcription):
    """Evaluate the vocabulary of the transcription based on CEFR levels."""
    cefr_prediction = cefr_classifier(transcription)
    print(f"CEFR vocabulary score: {cefr_prediction}")
    return cefr_prediction

def evaluate_pause(deepgram_response):
    words = deepgram_response.to_dict()["results"]["channels"][0]["alternatives"][0]["words"]

    pause_threshold = 1.0
    long_pauses = 0
    pause_between_words = []

    for i in range(len(words) - 1):
        current_end = words[i]["end"]
        next_start = words[i + 1]["start"]

        pause_duration = next_start - current_end

        if pause_duration > pause_threshold:
            long_pauses += 1
            print(f"Pause of {pause_duration:.2f}s between '{words[i]['word']}' and '{words[i + 1]['word']}'")
            pause_between_words.append({"start_word": words[i]['word'], "end_word": words[i + 1]['word'], "duration": pause_duration})

    print(f"Number of pauses longer than 1 second: {long_pauses}")

    score = 1.0

    if long_pauses == 0:
        return score, pause_between_words
    else:
        return 1 - (long_pauses / len(words)), pause_between_words

def evaluate_stutter(deepgram_response):
    words_list = deepgram_response.to_dict()["results"]["channels"][0]["alternatives"][0]["words"]
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
    words_data = deepgram_response.to_dict()["results"]["channels"][0]["alternatives"][0]["words"]
    duration = deepgram_response.to_dict()["metadata"]["duration"]

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
    evaluate_transcription("It was a real nice day today. Can I have you’re coat? We should contact they’re friend.")



