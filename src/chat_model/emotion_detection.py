import os
import requests

BASE_URL = os.getenv("EVALUATION_BASE_URL")

# emotion_classifier = pipeline(
#     task="text-classification",
#     model="SamLowe/roberta-base-go_emotions",
#     top_k=None
# )

confusion_category = [
    "confusion"
]

happy_category = [
    "admiration", "amusement", "approval", "caring", "desire",
    "excitement", "gratitude", "joy", "love", "neutral",
    "optimism", "pride", "relief", "surprise"
]

calm_category = [
    "curiosity"
]

sad_category = [
    "anger", "annoyance", "disappointment", "disapproval",
    "disgust", "embarrassment", "fear", "grief", "nervousness",
    "realization", "remorse", "sadness"
]

def detect_emotion(sentence: str):
    data = {"text": sentence}
    results = requests.post(f"{BASE_URL}/emotion-classifier", json=data).json()
    # print(results)
    emotion = results['text']
    if emotion in confusion_category:
        return "confusion"
    elif emotion in happy_category:
        return "happy"
    elif emotion in calm_category:
        return "calm"
    elif emotion in sad_category:
        return "sad"
    else:
        return "No emotion detected"

# if __name__ == "__main__":
#     sentences = [
#         "I'm sorry I don't understand. Can you repeat that again?"
#     ]
#     model_outputs = emotion_classifier(sentences)
#     sentence_emotions = []
#     for i, output in enumerate(model_outputs):
#         sentence_emotion = {
#             "sentence": sentences[i].replace('\n', ' ')
#         }
#         emotion = output[0]['label']
#         score = output[0]['score']
#         sentence_emotion['emotion'] = emotion
#         sentence_emotion['confidence_score'] = score
#         if emotion in confusion_category:
#             sentence_emotion['emotion_category'] = "confusion"
#         elif emotion in happy_category:
#             sentence_emotion['emotion_category'] = "happy"
#         elif emotion in calm_category:
#             sentence_emotion['emotion_category'] = "calm"
#         elif emotion in sad_category:
#             sentence_emotion['emotion_category'] = "sad"
#         else:
#             sentence_emotion['emotion_category'] = "No emotion detected"
#         sentence_emotions.append(sentence_emotion)
#     print(json.dumps(sentence_emotions, indent=2, ensure_ascii=False))

