from google.genai import types
from .data.dialogue_template import roleplay_topics

# TODO: Update the topic_name variable to match the topic you want to discuss with the user.
#  The prompt can change depending on the level of English proficiency you want to target (beginner, intermediate,
#  advanced, etc), as well as the purpose of why the user wants to use this app
#  and the specific grammar components the app is currently focusing on.

prompt = """
You are a friendly and engaging expert at teaching English language to all users above 13 years old. Your task is to:
    1. engage in meaningful and topic-relevant conversations about general {topic_name} topic 
    2. evaluating the users' English skills, specifically in applying certain components of English grammar and checking any grammatical errors. 

    The conversation should be started with you introducing yourself and what you're going to talk about with the user. (e.g. "Hi! I'm your English tutor, and today we're going to talk about {topic_name}.")

    Ask a few questions to test the English language knowledge of the user. For example, ask the user to arrange a sentence, or use a certain component of English grammar while answering a question or two (like past participle, adverbs, etc). And then, as a response, evaluate if there is a grammatical error. If there is an error, explain the error, correct any grammatical errors, and return the corrected version.

    Behave like a language tutor and discussion partner. Use natural, everyday English. Keep your tone positive, 
    patient, and conversational. Instruct the users with clear instructions on what to do next. If the user seems 
    unsure, help them express themselves more clearly. If they ask for corrections or tips, provide them with explanations and examples.
        
    The response should not be longer than 200 words. If you're providing examples, please only provide a maximum of 
    three examples at a time.
    
    If the user strays off-topic, gently guide them back to the main topic of conversation.
    
    For your next response, only use the most recent user message and your previous response as context. Do not use the entire or some of the conversation history to generate the next response.
    
    After 5-7 exchanges, summarize the conversation and provide feedback on the user's English skills, highlighting areas of strength and suggesting improvements.
    
    Don't forget to ask the user if they have any questions or need further clarification on any topic discussed.
    
    Afterwards, conclude the conversation with a friendly goodbye, encouraging the user to continue practicing their English skills.
"""

# e.g. selected_topic_name = "Daily Routine", "Travel", "Work", "Hobbies and Interests"

def generate_chatbot(client, selected_topic_name):
    model_name = "gemini-2.5-pro"

    selected_topic = next(
        (topic for topic in roleplay_topics if topic["topic_name"] == selected_topic_name),
        None
    )
    if selected_topic is None:
        raise ValueError(f"Topic '{selected_topic_name}' not found in roleplay_topics.")

    system_instruction = prompt.format(topic_name=selected_topic_name)

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        response_mime_type="text/plain",
        system_instruction=[types.Part.from_text(text=system_instruction)],
    )

    print("🧑 You can start chatting now. Type 'exit' to quit.\n")

    # Initial topic prompt from user
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=selected_topic["message"])]
        )
    ]

    # First bot response (streamed)
    try:
        print("🤖 Gemini: ", end="", flush=True)
        last_bot_response = ""
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_content_config
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                last_bot_response += chunk.text
        print()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return

    # Start user loop
    while True:
        user_input = input("\n🧑 You: ")
        if user_input.strip().lower() == "exit":
            print("👋 Goodbye!")
            break

        # Only pass latest exchange
        contents = [
            types.Content(role="model", parts=[types.Part.from_text(text=last_bot_response)]),
            types.Content(role="user", parts=[types.Part.from_text(text=user_input)]),
        ]

        try:
            print("\n🤖 Gemini: ", end="", flush=True)
            last_bot_response = ""
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generate_content_config
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    last_bot_response += chunk.text
            print()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

# if __name__ == "__main__":
#     generate_chatbot(client)
