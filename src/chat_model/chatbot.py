from google.genai import types
from .data.dialogue_template import roleplay_topics
from .text_to_speech import transform_speech
from pydantic import BaseModel
from google import genai
from typing import Generator
# from ..pronunciation_model.pronunciation_model import g2p_from_user_history, transcribe_phonemes, score_pronunciation

# TODO: Update how the chat is gonna end.
#  The summarization and feedback should be provided after about 5-7 exchanges.
#  The user should be able to ask questions and get clarifications.

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

    Afterwards, conclude the conversation with a friendly goodbye, encouraging the user to continue practicing their English skills.
"""

class ChatInput(BaseModel):
    selected_topic_name: str
    user_input: str
    history_log: list[tuple[str, str]]
    exchange_count: int
    tts_model: str = "aura-2-thalia-en"

def chat_stream(client: genai.Client, input_data: dict) -> Generator[str, None, None]:
    selected_topic = next(
        (topic for topic in roleplay_topics if topic["topic_name"] == input_data["selected_topic_name"]),
        None
    )
    if not selected_topic:
        yield f"❌ Topic '{input_data['selected_topic_name']}' not found."
        return

    system_instruction = prompt.format(topic_name=input_data["selected_topic_name"])
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        response_mime_type="text/plain",
        system_instruction=[types.Part.from_text(text=system_instruction)],
    )

    exchange_count = input_data.get("exchange_count", 0)
    history_log = input_data.get("history_log", [])
    user_input = input_data.get("user_input", "")
    model = input_data.get("tts_model", "aura-2-thalia-en")

    if exchange_count == 0:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=selected_topic["message"])]
            )
        ]
    else:
        contents = []
        for role, msg in history_log[-2:]:
            role_ = "user" if role == "user" else "model"
            contents.append(types.Content(role=role_, parts=[types.Part.from_text(text=msg)]))
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))

    last_bot_response = ""

    try:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-pro", contents=contents, config=config
        ):
            if chunk.text:
                last_bot_response += chunk.text
                yield chunk.text
    except Exception as e:
        yield f"\n❌ Error: {e}"
        return

    # TTS output
    transform_speech(
        f"data/audio/output_{exchange_count+1}_{model}.wav",
        last_bot_response,
        model=model
    )

    # Summarize after 7 turns
    if exchange_count + 1 >= 7:
        yield "\n\n📚 Gemini summary:\n"
        summary_input = "\n".join(f"{r.capitalize()}: {msg}" for r, msg in history_log + [("user", user_input), ("bot", last_bot_response)])
        summary_prompt = f"""
You are an English tutor. The following is a conversation between you and a student. Based on the full conversation history below, summarize the session and give feedback on the user's English language skills. 
First, say thank you to the user for the conversation and summarize the main points discussed.
Highlight their strengths, point out areas for improvement, and suggest what they can focus on next.

Also ask if they have any questions about what was discussed, and end the session with a friendly goodbye encouraging them to keep practicing.

Conversation history:
{summary_input}
        """.strip()

        try:
            summary_contents = [
                types.Content(role="user", parts=[types.Part.from_text(text=summary_prompt)])
            ]
            for chunk in client.models.generate_content_stream(
                model="gemini-2.5-pro",
                contents=summary_contents,
                config=config
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n❌ Error during summary: {e}"



# e.g. selected_topic_name = "Daily Routine", "Travel", "Work", "Hobbies and Interests"

def generate_chatbot(client, selected_topic_name, model):
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

    # Initial user input (topic kickoff)
    initial_contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=selected_topic["message"])]
        )
    ]

    # First bot response
    try:
        print("🤖 Gemini: ", end="", flush=True)
        last_bot_response = ""
        for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=initial_contents,
                config=generate_content_config
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                last_bot_response += chunk.text
        print()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return

    # Conversation tracking
    exchange_count = 1
    history_log = [("user", selected_topic["message"]), ("bot", last_bot_response)]
    transform_speech(f"data/audio/output_{exchange_count}_{model}.wav", last_bot_response,
                     model=model)

    while True:
        user_input = input("\n🧑 You: ")
        if user_input.strip().lower() == "exit":
            print("👋 Goodbye!")
            break

        exchange_count += 1
        history_log.append(("user", user_input))

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

        transform_speech(f"data/audio/output_{exchange_count}_{model}.wav", last_bot_response,
                         model=model)

        history_log.append(("bot", last_bot_response))

        # Summarize after 7 exchanges (user+bot = 14 lines)
        if exchange_count >= 7:
            print("\n📚 Gemini is summarizing your progress so far...\n")

            # Build full conversation history as summary prompt
            summary_input = ""
            for role, message in history_log:
                summary_input += f"{role.capitalize()}: {message}\n"

            summary_prompt = f"""
You are an English tutor. The following is a conversation between you and a student. Based on the full conversation history below, summarize the session and give feedback on the user's English language skills. 
First, say thank you to the user for the conversation and summarize the main points discussed.
Highlight their strengths, point out areas for improvement, and suggest what they can focus on next.

Also ask if they have any questions about what was discussed, and end the session with a friendly goodbye encouraging them to keep practicing.

Conversation history:
{summary_input}
            """

            try:
                summary_contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=summary_prompt.strip())]
                    )
                ]

                print("🤖 Gemini: ", end="", flush=True)
                for chunk in client.models.generate_content_stream(
                        model=model_name,
                        contents=summary_contents,
                        config=generate_content_config
                ):
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                print("\n👋 Conversation ended.\n")
            except Exception as e:
                print(f"\n❌ Error during summary: {e}")

            break

# if __name__ == "__main__":
#     generate_chatbot(client)
