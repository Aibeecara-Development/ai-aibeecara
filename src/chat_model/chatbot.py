from google.genai import types
from .data.dialogue_template import roleplay_topics
from pydantic import BaseModel
from google import genai
from fastapi import WebSocket
# from src.chat_model.scoring.score_model import evaluate_transcription
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv
import time, random
# from ..pronunciation_model.pronunciation_model import g2p_from_user_history, transcribe_phonemes, score_pronunciation

# TODO:
#  - Update the chat_stream function for the API so that it yields score
#  of the evaluation transcription model like the generate_chatbot function.
#  - Integrate the build_contents and safe_generate into the chat_task function to minimize overloading error.
#  - Make it so that the pronunciation evaluation is done while the chat with the user is ongoing.

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

    Afterwards, if the user wants to conclude, conclude the conversation with a friendly goodbye, encouraging the 
    user to continue practicing their English skills.
"""

load_dotenv()
gemini_key = os.getenv("GEMINI_KEY")
client = genai.Client(api_key=gemini_key)

class ChatInput(BaseModel):
    selected_topic_name: str
    user_input: str
    history_log: list[tuple[str, str]]
    exchange_count: int
    tts_model: str = "aura-2-amalthea-en"

def build_contents(history_log):
    contents = []
    for role, msg in history_log:
        contents.append(
            types.Content(
                role="user" if role == "user" else "model",
                parts=[types.Part.from_text(text=msg)]
            )
        )
    return contents

def safe_generate(client, model_name, contents, config, retries=5):
    for attempt in range(retries):
        try:
            return client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config
            )
        except Exception as e:
            if "503" in str(e) and attempt < retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"\n⚠️ Model overloaded. Retrying in {wait_time:.1f}s...\n")
                time.sleep(wait_time)
            else:
                raise

async def chat_task(ws, chat_queue, client, tts_stream, input_data: ChatInput):
    """
    Processes messages from chat_queue,
    streams Gemini responses, and sends TTS audio to WebSocket in real-time.
    Mimics chat_api_sync logic for topic/history handling.
    """
    while True:
        user_text = await chat_queue.get()
        if user_text is None:
            break

        # --- Select topic ---
        selected_topic = next(
            (topic for topic in roleplay_topics if topic["topic_name"] == input_data["selected_topic_name"]),
            None
        )
        if not selected_topic:
            await ws.send_text(f"❌ Topic '{input_data['selected_topic_name']}' not found.")
            continue

        # --- System instruction setup ---
        system_instruction = prompt.format(topic_name=input_data["selected_topic_name"])
        system_instruction_content = types.Content(
            role="system",
            parts=[types.Part.from_text(text=system_instruction)]
        )
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            response_mime_type="text/plain",
            system_instruction=system_instruction_content,
        )

        # --- Conversation history ---
        exchange_count = input_data.get("exchange_count", 0)
        history_log = input_data.get("history_log", [])

        if exchange_count == 0:
            # First message: send topic's initial message
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=selected_topic["message"])]
                )
            ]
        else:
            # Include last exchanges + new user message
            contents = []
            for role, msg in history_log[-2:]:
                role_ = "user" if role == "user" else "model"
                contents.append(types.Content(role=role_, parts=[types.Part.from_text(text=msg)]))
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))

        # --- Stream Gemini output + TTS in real-time ---
        try:
            stream = client.models.generate_content_stream(
                model="gemini-2.5-pro",
                contents=contents,
                config=config
            )

            async for final_text in _stream_text_to_tts(ws, stream, tts_stream):
                pass  # Already handled inside helper

        except Exception as e:
            await ws.send_text(f"\n❌ Error: {e}")


async def _stream_text_to_tts(ws, stream, tts_stream):
    """
    Helper to iterate over Gemini chunks and send TTS audio in real-time.
    """
    for chunk in stream:
        if chunk.text:
            async for audio_chunk in tts_stream(chunk.text):
                await ws.send_bytes(audio_chunk)
            yield chunk.text

async def chat_stream_websocket(client: genai.Client, input_data: Dict, websocket: WebSocket):
    selected_topic = next(
        (topic for topic in roleplay_topics if topic["topic_name"] == input_data["selected_topic_name"]),
        None
    )
    if not selected_topic:
        await websocket.send_text(f"❌ Topic '{input_data['selected_topic_name']}' not found.")
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

    # First exchange: kickoff
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
                await websocket.send_text(chunk.text)
    except Exception as e:
        await websocket.send_text(f"\n❌ Error: {e}")
        return

    # Summarize if enough turns

async def summarize_conversation(
    client,
    history_log: List[Tuple[str, str]],
    user_input: str = "",
    model_name: str = "gemini-2.5-pro"
) -> str:
    """Summarize a conversation with Gemini without streaming."""

    summary_input = "\n".join(
        f"{r.capitalize()}: {msg}"
        for r, msg in history_log + ([("user", user_input)] if user_input else [])
    )

    summary_prompt = f"""
You are an English tutor. The following is a conversation between you and a student. Based on the full conversation history below, summarize the session and give feedback on the user's English language skills. 
First, say thank you to the user for the conversation and summarize the main points discussed.
Highlight their strengths, point out areas for improvement, and suggest what they can focus on next.

Also ask if they have any questions about what was discussed, and end the session with a friendly goodbye encouraging them to keep practicing.

Conversation history:
{summary_input}
    """.strip()

    summary_contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=summary_prompt)])
    ]

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        response_mime_type="text/plain"
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=summary_contents,
            config=config
        )
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        raise RuntimeError(f"Gemini summarization error: {e}")

async def chat_api_sync(client: genai.Client, input_data: dict) -> str:
    selected_topic = next(
        (topic for topic in roleplay_topics if topic["topic_name"] == input_data["selected_topic_name"]),
        None
    )
    if not selected_topic:
        return f"❌ Topic '{input_data['selected_topic_name']}' not found."

    system_instruction = prompt.format(topic_name=input_data["selected_topic_name"])
    system_instruction_content = types.Content(
        role="system",
        parts=[types.Part.from_text(text=system_instruction)]
    )
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        response_mime_type="text/plain",
        system_instruction=system_instruction_content,
    )

    exchange_count = input_data.get("exchange_count", 0)
    history_log = input_data.get("history_log", [])
    user_input = input_data.get("user_input", "")
    model = input_data.get("tts_model", "aura-2-amalthea-en")

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
        # Collect entire response into a string instead of yielding
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-pro", contents=contents, config=config
        ):
            if chunk.text:
                last_bot_response += chunk.text
    except Exception as e:
        return f"\n❌ Error: {e}"

    return last_bot_response

async def custom_topic_validation(client: genai.Client, selected_topic_name: str) -> str:

    validation_prompt = f"""
You are a classifier that determines if a given topic is a BROAD, conversation-worthy topic 
or a NARROW, object-specific topic.

BROAD: Topics that are large in scope, can be discussed in many contexts, and often involve ideas, fields, or domains.
NARROW: Topics that are specific physical objects or highly limited in scope.

Examples:
- "politics" → BROAD
- "geography" → BROAD
- "climate change" → BROAD
- "table" → NARROW
- "chair" → NARROW
- "HDMI cable" → NARROW
- "chess" → BROAD
- "basketball" → BROAD
- "toothbrush" → NARROW

Classify the following topic and respond with only BROAD or NARROW:

Topic: {selected_topic_name}
"""

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=validation_prompt
    )

    return response.text.strip()

async def hint_to_users(client: genai.Client, chatbot_message: str) -> str:
    hint_prompt = f"""
        You are helping learners practice English conversation.
        Given a chatbot’s last message, generate a hint for the learner that includes:
        
        Example you can say – a short, natural sentence they could reply with.
        
        Context – explain briefly why this response works and how it keeps the conversation going.
        
        Keep the format clean and consistent like this:
        
        Example you can say
        I’d love to try it with cheese and mushrooms.
        
        Context
        This works because it adds a fun twist to the conversation by suggesting flavors. It also invites the other person to share more about their food preferences.
        
        Another Example
        
        Example you can say
        That sounds exciting! Have you done it before?
        
        Context
        This works because it shows enthusiasm while also asking a follow-up question, encouraging a deeper conversation.
        
        Generate example you can say and context for the following chatbot message:
        {chatbot_message}
        """

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=hint_prompt
    )

    return response.text.strip()

def grammar_correction(client: genai.Client, incorrect_transcript: str) -> str:
    prompt = f"""
        You are a grammar correction assistant. 
        Follow the format strictly:
        
        Explanation:
            Explain why the grammar is wrong in no more than 50 words.
        
        Tense Used:
            Briefly describe the tense used in the corrected version in no more than 50 words.
        
        Here are examples:
        
        Text: "He go to school every day."
        Explanation:
            The verb "go" does not agree with the subject "he." It should be "goes."
        Tense Used:
            Present simple tense, used for regular or habitual actions.
        
        Text: "Yesterday, she is playing tennis with her friend."
        Explanation:
            "Is playing" is incorrect with "yesterday." It should be "was playing."
        Tense Used:
            Past continuous tense, used for ongoing actions in the past.
        
        Now do the same for this text:
        
        Text: "{incorrect_transcript}"
        Explanation:
    """

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    )

    return response.text.strip()

# e.g. selected_topic_name = "Daily Routine", "Travel", "Work", "Hobbies and Interests"

def generate_chatbot(client, selected_topic_name, model="gemini-2.5-pro"):

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
    # transform_speech(f"data/audio/output_{exchange_count}_{model}.wav", last_bot_response,
    #                  model=model)

    while True:
        print(f"Exchange count: {exchange_count}")
        user_input = input("\n🧑 You: ")
        if user_input.strip().lower() == "exit":
            print("👋 Goodbye!")
            break

        exchange_count += 1
        history_log.append(("user", user_input))
        contents = build_contents(history_log)

        contents = [
            types.Content(role="model", parts=[types.Part.from_text(text=last_bot_response)]),
            types.Content(role="user", parts=[types.Part.from_text(text=user_input)]),
        ]

        try:
            print("\n🤖 Gemini: ", end="", flush=True)
            last_bot_response = ""
            for chunk in safe_generate(client, model_name, contents, generate_content_config):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    last_bot_response += chunk.text
            print()
            history_log.append(("bot", last_bot_response))
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

        # transform_speech(f"data/audio/output_{exchange_count}_{model}.wav", last_bot_response,
        #                  model=model)

        history_log.append(("bot", last_bot_response))


        # Summarize after 7 exchanges (user+bot = 14 lines)
#         if exchange_count >= 3:
#             print("\n📚 Gemini is summarizing your progress so far...\n")
#
#             # Build full conversation history as summary prompt
#             summary_input = ""
#             for role, message in history_log:
#                 summary_input += f"{role.capitalize()}: {message}\n"
#
#             summary_prompt = f"""
# You are an English tutor. The following is a conversation between you and a student. Based on the full conversation history below, summarize the session and give feedback on the user's English language skills.
# First, say thank you to the user for the conversation and summarize the main points discussed.
# Highlight their strengths, point out areas for improvement, and suggest what they can focus on next.
#
# Also ask if they have any questions about what was discussed, and end the session with a friendly goodbye encouraging them to keep practicing.
#
# Conversation history:
# {summary_input}
#             """
#
#             try:
#                 summary_contents = [
#                     types.Content(
#                         role="user",
#                         parts=[types.Part.from_text(text=summary_prompt.strip())]
#                     )
#                 ]
#
#                 print("🤖 Gemini: ", end="", flush=True)
#                 for chunk in client.models.generate_content_stream(
#                         model=model_name,
#                         contents=summary_contents,
#                         config=generate_content_config
#                 ):
#                     if chunk.text:
#                         print(chunk.text, end="", flush=True)
#                 print("\n👋 Conversation ended.\n")
#
#                 user_message = ""
#                 user_message_array = []
#                 for role, message in history_log:
#                     if role == "user":
#                         user_message_array.append(message)
#
#                 for message in user_message_array[-(exchange_count - 1):]:
#                     user_message += message + " "
#
#                 # Evaluate transcription
#                 score = evaluate_transcription(user_message)
#
#                 print(f"📊 Your grammar score: {score * 100:.2f}%")
#
#             except Exception as e:
#                 print(f"\n❌ Error during summary: {e}")
#
#             break

# if __name__ == "__main__":
#     generate_chatbot(client, "School Subjects")
