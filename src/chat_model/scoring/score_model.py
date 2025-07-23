from jiwer import wer
from happytransformer import HappyTextToText, TTSettings

happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

def evaluate_transcription(transcription):
    """Evaluate the transcription for grammar issues."""
    args = TTSettings(num_beams=5, min_length=1)

    # Add the prefix "grammar: " before each input
    result = happy_tt.generate_text(f"grammar: {transcription}.", args=args)

    print(f"trasncription: {transcription}")

    wer_score = wer(result.text, transcription)

    print(result.text)

    transcription_score = 1 - wer_score

    print(transcription_score)

    return transcription_score


def evaluate_pause(deepgram_response):
    words = deepgram_response.to_dict()["results"]["channels"][0]["alternatives"][0]["words"]

    pause_threshold = 3.0
    long_pauses = 0

    for i in range(len(words) - 1):
        current_end = words[i]["end"]
        next_start = words[i + 1]["start"]

        pause_duration = next_start - current_end

        if pause_duration > pause_threshold:
            long_pauses += 1
            print(f"Pause of {pause_duration:.2f}s between '{words[i]['word']}' and '{words[i + 1]['word']}'")

    print(f"Number of pauses longer than 3 seconds: {long_pauses}")

    score = 1.0

    if long_pauses == 0:
        return score
    else:
        return 1 - (long_pauses / len(words))

def evaluate_repetition(deepgram_response):
    words_list = deepgram_response.to_dict()["results"]["channels"][0]["alternatives"][0]["words"]
    words = [w["word"].lower() for w in words_list]
    count = 0
    i = 0
    repeated_phrases = []

    while i < len(words):
        max_repeat_len = (len(words) - i) // 2
        found_repeat = False

        for size in range(max_repeat_len, 0, -1):
            first = words[i:i + size]
            second = words[i + size:i + 2 * size]
            if first == second:
                count += 1
                repeated_phrases.append(" ".join(first))
                i += size * 2  # skip the repeated pair
                found_repeat = True
                break

        if not found_repeat:
            i += 1

    print(f"Number of repetitions: {count}")
    print("Repeated phrases:", repeated_phrases)

    score = 1.0
    if count == 0:
        return score
    else:
        return 1 - (count / len(words))

if __name__ == "__main__":
    evaluate_transcription("It was a real nice day today. Can I have you’re coat? We should contact they’re friend.")



