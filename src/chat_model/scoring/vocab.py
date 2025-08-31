import spacy
import sqlite3
import lemminflect
import os
import json
from spacy.util import is_package
import nltk
from llama_cpp import Llama

def ensure_wordnet_downloaded():
    try:
        nltk.data.find("corpora/wordnet")
        nltk.data.find("corpora/omw-1.4")
        print("WordNet already downloaded ✅")
    except LookupError:
        print("Downloading WordNet resources...")
        nltk.download("wordnet")
        nltk.download("omw-1.4")

ensure_wordnet_downloaded()

def ensure_spacy_model(model_name: str = "en_core_web_sm"):
    try:
        # Try loading the model
        spacy.load(model_name)
        print(f"✅ Model '{model_name}' is already installed.")
    except OSError:
        # Model not found, check if it's in the environment
        if not is_package(model_name):
            print(f"⚠️ Model '{model_name}' not found. Downloading...")
            os.system(f"python -m spacy download {model_name}")
        else:
            print(f"⚠️ Model '{model_name}' is present but cannot be loaded.")

ensure_spacy_model("en_core_web_sm")

NLP = spacy.load("en_core_web_sm", exclude = ['parser', 'ner'])

DATABASE_FILENAME = '../data/word_cefr_minified.db'

conn = sqlite3.connect(DATABASE_FILENAME)
cursor = conn.cursor()

ABBREVIATION_MAPPING = {
    "'m": "am",
    "'s": "is",
    "'re": "are",
    "'ve": "have",
    "'d": "had",
    "n't": "not",
    "'ll": "will"
}

DIFFICULTY_MAPPING_REVERSE = {
    1: 'A1',
    2: 'A2',
    3: 'B1',
    4: 'B2',
    5: 'C1',
    6: 'C2'
}


def is_punctuation(word: str) -> bool:
    return not word and not any(char.isalpha() for char in word)


def custom_tokenize_text(text: str) -> list[tuple[str, str, str]]:
    text = text.replace("’", "'")
    tokens = []
    doc = NLP(text)
    for token in doc:
        word = token.text.lower().strip()
        word_pos = token.tag_
        proposed_lemma = token._.lemma().lower()

        abbreviation_form = ABBREVIATION_MAPPING.get(word)
        if abbreviation_form:
            word = abbreviation_form
            lemma = word
        elif proposed_lemma is None:
            lemma = word.lower()
        else:
            lemma = proposed_lemma

        tokens.append((word, lemma, word_pos))

    return tokens


def fetch_word_pos_level_tokens(word_pos_tokens_set: set[tuple[str, str]]) -> dict[tuple[str, str], float]:
    placeholders = ','.join(['(?, ?)' for _ in range(len(word_pos_tokens_set))])

    cursor.execute('''
        WITH word_pos_tags(word, pos_tag) AS (
            VALUES {}
        )
        SELECT
            word_pos_tags.word,
            word_pos_tags.pos_tag,
            COALESCE(
                AVG(CASE WHEN pt.tag = word_pos_tags.pos_tag THEN wp.level END),
                AVG(wp.level)
            ) AS avg_level
        FROM word_pos_tags
        JOIN words w ON word_pos_tags.word = w.word
        JOIN word_pos wp ON w.word_id = wp.word_id
        JOIN pos_tags pt ON wp.pos_tag_id = pt.tag_id
        GROUP BY word_pos_tags.word, word_pos_tags.pos_tag
    '''.format(placeholders), [item for sublist in word_pos_tokens_set for item in sublist])

    word_pos_level_tokens = cursor.fetchall()

    return {(word, pos_tag): float(avg_level) for word, pos_tag, avg_level in word_pos_level_tokens}


def get_word_pos_tokens_set(tokens: list[tuple[str, str, str]]) -> set[tuple[str, str]]:
    return {(token[0], token[2]) for token in tokens if not is_punctuation(token[1])}


def get_levels_tokens(tokens: list[tuple[str, str, str]]) -> list[tuple[str, str, str, float]]:
    word_pos_set = get_word_pos_tokens_set(tokens)
    word_pos_unique_level_tokens = fetch_word_pos_level_tokens(word_pos_set)

    word_pos_level_tokens = []
    for token in tokens:
        word, lemma, word_pos = token

        level = word_pos_unique_level_tokens.get((word, word_pos))
        if level is None:
            level = 0

        word_pos_level_tokens.append((word, lemma, word_pos, level))

    return word_pos_level_tokens


def get_word_level_count_statistic(level_tokens: list[tuple[str, str, str, float]]) -> list[int]:
    difficulty_levels_count = [0] * 6
    for token in level_tokens:
        level = round(token[3])
        if level:
            difficulty_levels_count[level - 1] += 1

    return difficulty_levels_count


def get_word_level_count_statistic_unique(level_tokens: list[tuple[str, str, str, float]]) -> list[int]:
    processed_word_pos_set = set()
    difficulty_levels_count = [0] * 6
    for token in level_tokens:
        level = round(token[3])
        to_check_tuple = (token[0], token[2])
        if level and not to_check_tuple in processed_word_pos_set:
            processed_word_pos_set.add(to_check_tuple)
            difficulty_levels_count[level - 1] += 1

    return difficulty_levels_count


def get_not_found_words(level_tokens: list[tuple[str, str, str, float]]) -> set[str]:
    not_found_words = set()
    for token in level_tokens:
        if not token[3] and token[0] and all(char.isalpha() for char in token[0]):
            not_found_words.add(token[0])

    return not_found_words


def filter_for_desired_level(level_tokens: list[tuple[str, str, str, float]],
                            min_level: float, max_level: float = 6) -> set[tuple[str, str, str, float]]:
    filtered_tokens = set()
    for token in level_tokens:
        level = token[3]
        if level >= min_level and level <= max_level:
            filtered_tokens.add(token)

    return filtered_tokens

input_text = """
In the heart of every forest, a hidden world thrives among the towering trees. Trees, 
those silent giants, are more than just passive observers of nature's drama; they are 
active participants in an intricate dance of life.

Did you know that trees communicate with each other? It's not through words or gestures 
like ours, but rather through a complex network of fungi that connect their roots 
underground. This network, often called the "wood wide web," allows trees to share 
nutrients, water, and even warnings about potential threats.

But trees are not just generous benefactors; they are also masters of adaptation. Take 
the mighty sequoias, for example, towering giants that have stood the test of time for 
thousands of years. These giants have evolved thick, fire-resistant bark to withstand 
the frequent wildfires of their native California.

And speaking of longevity, did you know that some trees have been around for centuries, 
witnessing history unfold? The ancient bristlecone pines of the American West, for 
instance, can live for over 5,000 years, making them some of the oldest living organisms 
on Earth.

So the next time you find yourself wandering through a forest, take a moment to appreciate 
the remarkable world of trees. They may seem like silent spectators, but their lives are 
full of fascinating stories waiting to be discovered.
"""

# tokens = custom_tokenize_text(input_text)
# level_tokens = get_levels_tokens(tokens)
#
# print("Text length:", len(input_text))
# print("Total tokens:", len(tokens))
#
# counter = 0
# print(f'{"WORD".ljust(26)}\t{"LEMMA".ljust(26)}\tPOS\tLEVEL\tCEFR')
# print('-' * 85)
# for token in level_tokens:
#     word, lemma, pos, level = token
#     cefr = DIFFICULTY_MAPPING_REVERSE.get(round(level))
#     if pos != '_SP':
#         print(f'{word.ljust(26)}\t{lemma.ljust(26)}\t{pos}\t{"{:.2f}".format(level)}\t{cefr}')
#
#         counter += 1
#         if counter >= 200:
#             break

def get_synonyms_with_levels(word: str, pos: str, get_levels_tokens_func) -> list[dict]:
    """Fetch synonyms of a word, assign CEFR levels, and provide WordNet example sentences."""
    from nltk.corpus import wordnet as wn

    # Map POS tag to WordNet POS
    pos_map = {
        "NOUN": wn.NOUN,
        "VERB": wn.VERB,
        "ADJ": wn.ADJ,
        "ADV": wn.ADV
    }
    wn_pos = pos_map.get(pos.upper(), wn.NOUN)

    synonyms = {}
    for synset in wn.synsets(word, pos=wn_pos):
        for lemma in synset.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() == word.lower():
                continue

            # Collect only examples that *contain the synonym*
            valid_examples = []
            for ex in synset.examples():
                # Replace original word with synonym if present
                if word.lower() in ex.lower():
                    new_ex = ex.lower().replace(word.lower(), synonym)
                    if synonym.lower() in new_ex:  # ensure synonym appears
                        valid_examples.append(new_ex)

            # Only keep synonym if there’s at least one valid example
            if valid_examples:
                synonyms[synonym] = valid_examples
                print(f"Found synonym: {synonym} for word: {word}")
                print(f"Examples: {valid_examples}")

    # If no synonyms found → return early
    if not synonyms:
        return []

    # Convert to token structure
    synonym_tokens = [(syn, syn, pos) for syn in synonyms.keys()]
    if not synonym_tokens:  # safeguard before DB
        return []

    # Get CEFR levels for synonyms
    level_tokens = get_levels_tokens_func(synonym_tokens)

    results = []
    for syn, lemma, pos_tag, level in level_tokens:
        examples = synonyms.get(syn, [])
        example_sentence = examples[0] if examples else f"No example available for '{syn}'."
        results.append({
            "synonym": syn,
            "pos": pos_tag,
            "level_score": round(level, 2),
            "cefr": DIFFICULTY_MAPPING_REVERSE.get(round(level), "NA"),
            "example_sentence": example_sentence
        })

    return results


def evaluate_cefr_stats(input_text: str) -> dict:
    """Evaluate CEFR statistics for the given input text."""
    tokens = custom_tokenize_text(input_text)
    level_tokens = get_levels_tokens(tokens)
    results = {
        "statistics": {},
        "tokens": []
    }

    difficulty_levels_count_unique = get_word_level_count_statistic_unique(level_tokens)
    for i in range(1, 7):
        results["statistics"][DIFFICULTY_MAPPING_REVERSE.get(i)] = difficulty_levels_count_unique[i - 1]

    # --- Token details ---
    synonym_counter = 0  # count how many words got synonyms
    for token in level_tokens:
        word, lemma, pos, level = token
        cefr = DIFFICULTY_MAPPING_REVERSE.get(round(level))

        if pos != '_SP':
            token_entry = {
                "word": word,
                "lemma": lemma,
                "pos": pos,
                "level_score": round(level, 2),
                "cefr": cefr
            }

            # only add synonyms if POS = JJ and limit to 5 words
            if (pos == "VBG" or pos == "VBN" or pos == "NNS" or pos == "NN") and synonym_counter < 5:
                token_entry["synonyms"] = get_synonyms_with_levels(word, pos, get_levels_tokens)
                synonym_counter += 1

            results["tokens"].append(token_entry)

    return results

# cefr_stats = evaluate_cefr_stats(input_text)
# print(json.dumps(cefr_stats, indent=4, ensure_ascii=False))