from google.genai import types
from data.dialogue_template import roleplay_topics

# TODO: Update the topic_name variable to match the topic you want to discuss with the user.
#  The prompt can change depending on the level of English proficiency you want to target (beginner, intermediate,
#  advanced, etc), as well as the purpose of why the user wants to use this app
#  and the specific grammar components the app is currently focusing on.

prompt = """
You are a friendly and engaging expert at teaching English language to all users above 13 years old. Your task is to:
    1. engage in meaningful and topic-relevant conversations about general {topic_name} topic 
    2. evaluating the users' English skills, specifically in applying certain components of English grammar and checking any grammatical errors. 

    The conversation should be started with you introducing yourself and what you're going to talk about with the user.

    Ask for a few questions to test the English language knowledge of the user. For example, ask the user to arrange a sentence, or use a certain component of English grammar while answering a question or two (like past participle, adverbs, etc). And then, as a response, evaluate if there is a grammatical error. If there is an error, explain the error, correct any grammatical errors, and return the corrected version.

    Behave like a language tutor and discussion partner. Use natural, everyday English. Keep your tone positive, patient, and conversational. If the user seems unsure, help them express themselves more clearly. If they ask for corrections or tips, provide them with explanations and examples.
"""

# e.g. selected_topic_name = "Daily Routine", "Travel", "Work", "Hobbies and Interests"

def generate_chatbot(client, selected_topic_name):
    model = "gemini-2.5-pro"

    # Find the topic dict that matches the topic name
    selected_topic = next(
        (topic for topic in roleplay_topics if topic["topic_name"] == selected_topic_name),
        None
    )

    if selected_topic is None:
        raise ValueError(f"Topic '{selected_topic_name}' not found in roleplay_topics.")

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=selected_topic["message"])]
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        response_mime_type="text/plain",
        system_instruction=[types.Part.from_text(text=prompt)],
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")


# if __name__ == "__main__":
#     generate_chatbot(client)
