from transformers import pipeline

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

confusion_category = [
    "confusion"
]

happy_category = [
    "admiration", "amusement", "approval", "caring", "desire",
    "excitement", "gratitude", "joy", "love", "neutral",
    "optimism", "pride", "relief", "surprise"
]

asking_questions_category = [
    "curiosity"
]

sad_category = [
    "anger", "annoyance", "disappointment", "disapproval",
    "disgust", "embarrassment", "fear", "grief", "nervousness",
    "realization", "remorse", "sadness"
]

def detect_emotion(text: str):
    sentence = [text]
    results = classifier(sentence)
    print(results)
    emotion = results[0][0]['label']
    if emotion in confusion_category:
        return "confusion"
    elif emotion in happy_category:
        return "happy"
    elif emotion in asking_questions_category:
        return "asking questions"
    elif emotion in sad_category:
        return "sad"
    else:
        return "No emotion detected"