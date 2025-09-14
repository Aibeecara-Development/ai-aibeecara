import os
from dotenv import load_dotenv
from google import genai
from .chat_model.chatbot import (chat_api_sync, chat_stream_websocket, summarize_conversation, custom_topic_validation,
                                 hint_to_users)
from .chat_model.emotion_detection import detect_emotion
from .audio_processing.transcriber import transcribe_audio_api
from .chat_model.text_to_speech import generate_tts_wav_api
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import requests
from deepgram import DeepgramClient
import asyncio
from deep_translator import GoogleTranslator
from .chat_model.scoring.score_model import (evaluate_pause, evaluate_repetition, evaluate_transcription)
from .chat_model.scoring.vocab import evaluate_cefr_stats
import json

load_dotenv()
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

gemini_key = os.getenv("GEMINI_KEY")
client = genai.Client(api_key=gemini_key)
load_dotenv()
deepgram_key = os.getenv('DEEPGRAM_KEY')
deepgram_client = DeepgramClient(deepgram_key)


class ChatInput(BaseModel):
    selected_topic_name: str = "General"
    user_input: str
    history_log: list[tuple[str, str]]
    exchange_count: int = 0
    tts_model: str = "aura-2-amalthea-en"


class TTSInput(BaseModel):
    text: str
    accent: str = "american"
    gender: str = "feminine"
    speed: float = 1.0


class AudioURLInput(BaseModel):
    audio_url: str


class EvaluationInput(BaseModel):
    history_log: list[tuple[str, str]]
    exchange_count: int


class ChatbotOutput(BaseModel):
    response: str


class CustomTopicInput(BaseModel):
    selected_topic_name: str


class ChatEmotion(BaseModel):
    model_output: str


class SpeakingInput(BaseModel):
    topic: str
    audio_url: str = ""
    history_log: list[tuple[str, str]] = []
    accent: str
    gender: str
    speed: float


def mock_stream_response(user_input):
    reply = f"{user_input}"
    for word in reply.split():
        yield word + " "
        time.sleep(0.2)


@app.post("/chat/")
async def chat_endpoint(chat_input: ChatInput):
    response_text = await chat_api_sync(client, chat_input.model_dump())
    return {"response": response_text}


@app.post("/chat/summary")
async def summary_endpoint(chat_input: ChatInput):
    """Generate a conversation summary as a REST API."""
    try:
        summary_text = await summarize_conversation(
            client=client,
            history_log=chat_input.history_log,
            user_input=chat_input.user_input
        )
        return {"summary": summary_text}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.websocket("/ws/chat/")
async def websocket_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            input_data = await websocket.receive_json()
            await chat_stream_websocket(client, input_data, websocket)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_text(f"❌ Error: {str(e)}")


@app.websocket("/ws/chat/summary")
async def websocket_summary_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            history_log = data.get("history_log", [])
            user_input = data.get("user_input", "")

            await websocket.send_text("📚 Gemini summary:\n")

            try:
                summary_text = await summarize_conversation(
                    client=client,
                    history_log=history_log,
                    user_input=user_input
                )
                await websocket.send_text(summary_text)

            except Exception as e:
                await websocket.send_text(f"❌ Error during summary: {str(e)}")

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")


@app.post("/transcribe/")
async def transcribe_endpoint(input_data: AudioURLInput):
    try:
        # Download audio file
        resp = requests.get(input_data.audio_url, timeout=30)
        resp.raise_for_status()
        audio_data = resp.content

        # Transcribe
        response, transcript = transcribe_audio_api(audio_data)

        # Evaluate pause
        pause_score, pause_details = evaluate_pause(response)

        # Evaluate repetition
        repetition_score, repeated_phrases = evaluate_repetition(response)

        return {
            "transcript": transcript,
            "pause_score": pause_score,
            "pause_details": pause_details,
            "repetition_score": repetition_score,
            "repeated_phrases": repeated_phrases
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/chat/tts/")
async def chat_tts(input: TTSInput):
    try:
        wav_path = generate_tts_wav_api(input.text, accent=input.accent, gender=input.gender, speed=input.speed)
        return FileResponse(wav_path, media_type="audio/wav", filename="response.wav")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/speaking/")
async def speaking_endpoint(input_data: SpeakingInput):
    """
    Combined endpoint for speaking practice:
    1. Optionally transcribe audio from URL
    2. Generate chat response based on topic and history
    3. Convert response to audio and return both transcripts and audio as multipart response
    """
    print(f"Received speaking request: {input_data}")
    start_time = time.perf_counter()
    timings = {}

    try:
        user_input = ""

        # Step 1: Transcribe audio if URL is provided
        if input_data.audio_url.strip():
            transcription_start = time.perf_counter()

            resp = requests.get(input_data.audio_url, timeout=30)
            resp.raise_for_status()
            audio_data = resp.content
            _, transcript = transcribe_audio_api(audio_data)
            user_input = transcript

            transcription_time = time.perf_counter() - transcription_start
            timings['transcription'] = transcription_time
            print(f"Audio transcription took: {transcription_time:.3f} seconds")
        else:
            timings['transcription'] = 0
            print("No audio to transcribe")

        # Step 2: Prepare chat input data
        prep_start = time.perf_counter()

        chat_input_data = {
            "selected_topic_name": input_data.topic,
            "user_input": user_input,
            "history_log": input_data.history_log,
            "exchange_count": len(input_data.history_log) // 2,
            "tts_model": "aura-2-amalthea-en"
        }

        prep_time = time.perf_counter() - prep_start
        timings['preparation'] = prep_time
        print(f"Data preparation took: {prep_time:.3f} seconds")

        # Step 3: Generate chat response
        chat_start = time.perf_counter()

        response_text = await chat_api_sync(client, chat_input_data)

        chat_time = time.perf_counter() - chat_start
        timings['chat_generation'] = chat_time
        print(f"Chat response generation took: {chat_time:.3f} seconds")

        # Step 4: Convert response to audio with specified accent, gender, and speed
        tts_start = time.perf_counter()

        wav_path = generate_tts_wav_api(
            response_text,
            accent=input_data.accent,
            gender=input_data.gender,
            speed=input_data.speed
        )

        tts_time = time.perf_counter() - tts_start
        timings['text_to_speech'] = tts_time
        print(f"Text-to-speech conversion took: {tts_time:.3f} seconds")

        # Step 5: Create multipart response with transcripts and audio
        multipart_start = time.perf_counter()

        # Prepare transcripts JSON
        transcripts = {
            "user_transcript": user_input,
            "bot_transcript": response_text
        }

        # Read audio file
        with open(wav_path, 'rb') as f:
            audio_data = f.read()

        # Create multipart response
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

        # Construct multipart body
        parts = []

        # Add transcripts part
        transcripts_json = json.dumps(transcripts)
        parts.append(
            f'------WebKitFormBoundary7MA4YWxkTrZu0gW\r\n'
            f'Content-Disposition: form-data; name="transcripts"\r\n'
            f'Content-Type: application/json\r\n'
            f'\r\n'
            f'{transcripts_json}\r\n'
        )

        # Add audio part header
        parts.append(
            f'------WebKitFormBoundary7MA4YWxkTrZu0gW\r\n'
            f'Content-Disposition: form-data; name="audio"; filename="speaking_response.wav"\r\n'
            f'Content-Type: audio/wav\r\n'
            f'\r\n'
        )

        # Combine text parts
        text_content = ''.join(parts).encode('utf-8')

        # Create final content with audio data and boundary end
        content = text_content + audio_data + b'\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--\r\n'

        multipart_time = time.perf_counter() - multipart_start
        timings['multipart_creation'] = multipart_time

        # Calculate total time
        total_time = time.perf_counter() - start_time
        timings['total'] = total_time

        # Log summary
        print(f"Total request time: {total_time:.3f} seconds")
        print(f"Timing breakdown: {timings}")

        return Response(
            content=content,
            media_type=f"multipart/form-data; boundary={boundary}",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}"
            }
        )

    except Exception as e:
        total_time = time.perf_counter() - start_time
        print(f"Request failed after {total_time:.3f} seconds: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


## Custom topic validation endpoint --> Either between BROAD or NARROW
@app.post("/chat/topic/")
async def chat_topic(input: CustomTopicInput):
    try:
        message = await custom_topic_validation(client, input.selected_topic_name)
        return {"validation": message}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/chat/emotion/")
async def chat_emotion(input: ChatInput):
    try:
        emotion = await asyncio.to_thread(detect_emotion(input.user_input))
        return {"emotion": emotion}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Retrieve the history log and exchange count from the POST request of /chat
@app.post("/evaluate/")
def evaluate_grammar(input: EvaluationInput):
    user_messages = " ".join(
        msg for role, msg in input.history_log[-(input.exchange_count * 2):] if role == "user"
    )
    grammar_score, corrected_transcript = evaluate_transcription(user_messages)
    vocabulary_score = evaluate_cefr_stats(user_messages)
    return {"original_message": user_messages,
            "corrected_transcript": corrected_transcript,
            "evaluation_score": grammar_score,
            "vocabulary_score": vocabulary_score}


# is it async or not?
@app.post("/translate/")
def translate_text(input: ChatInput):
    try:
        translated_text = GoogleTranslator(source='auto', target='id').translate(input.user_input)
        return {"translated_text": translated_text}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/hint/")
def hint_endpoint(input: ChatbotOutput):
    try:
        hint = hint_to_users(client, input.response)
        return {"hint": hint}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.websocket("/conversation_stream/")
async def conversation_stream(ws: WebSocket):
    await ws.accept()

    try:
        await ws.send_text("🔗 Connected to conversation stream")

        # Time the configuration setup
        config_start = time.perf_counter()

        config_data = await ws.receive_json()

        input_data = ChatInput(
            selected_topic_name=config_data.get("selected_topic_name", "General"),
            user_input=config_data.get("user_input", ""),
            history_log=config_data.get("history_log", []),
            exchange_count=config_data.get("exchange_count", 0),
            tts_model=config_data.get("tts_model", "aura-2-amalthea-en")
        )

        config_time = time.perf_counter() - config_start
        print(f"WebSocket configuration setup took: {config_time:.3f} seconds")

        await ws.send_text(f"Configuration received: Topic = {input_data.selected_topic_name}")
        await ws.send_text("Ready to receive audio.")

        message_counter = 0

        while True:
            try:
                message_counter += 1
                message_start_time = time.perf_counter()
                timings = {}

                print(f"📝 Processing message #{message_counter}")

                # Time audio reception
                audio_receive_start = time.perf_counter()
                audio_data = await ws.receive_bytes()
                audio_receive_time = time.perf_counter() - audio_receive_start
                timings['audio_receive'] = audio_receive_time

                await ws.send_text("Transcribing audio...")

                try:
                    # Time transcription
                    transcription_start = time.perf_counter()
                    response, transcript = transcribe_audio_api(audio_data)
                    transcription_time = time.perf_counter() - transcription_start
                    timings['transcription'] = transcription_time

                    if not transcript.strip():
                        total_message_time = time.perf_counter() - message_start_time
                        timings['total'] = total_message_time
                        print(f"❌ Message #{message_counter} - No speech detected (Total: {total_message_time:.3f}s)")
                        print(f"Timing breakdown: {timings}")
                        await ws.send_text("❌ No speech detected in audio")
                        continue

                    await ws.send_text(f"Transcribed: {transcript}")
                    print(f"Audio transcription took: {transcription_time:.3f} seconds")

                    # Time data preparation
                    prep_start = time.perf_counter()
                    input_data.user_input = transcript
                    input_data.exchange_count = max(1, input_data.exchange_count)
                    prep_time = time.perf_counter() - prep_start
                    timings['preparation'] = prep_time

                    await ws.send_text("Generating response...")

                    # Time chat response generation
                    chat_start = time.perf_counter()
                    chat_response = await chat_api_sync(client, input_data.model_dump())
                    chat_time = time.perf_counter() - chat_start
                    timings['chat_generation'] = chat_time

                    await ws.send_text(f"Bot: {chat_response}")
                    print(f"Chat response generation took: {chat_time:.3f} seconds")

                    # Time evaluation
                    eval_start = time.perf_counter()
                    evaluation_result = {
                        "transcript": transcript,
                        "response": chat_response
                    }
                    eval_time = time.perf_counter() - eval_start
                    timings['evaluation'] = eval_time

                    await ws.send_text(f"📈 Evaluation: {evaluation_result}")

                    # Time history update
                    history_start = time.perf_counter()
                    input_data.history_log.append(("user", transcript))
                    input_data.history_log.append(("assistant", chat_response))
                    input_data.exchange_count += 1
                    history_time = time.perf_counter() - history_start
                    timings['history_update'] = history_time

                    # Calculate total message processing time
                    total_message_time = time.perf_counter() - message_start_time
                    timings['total'] = total_message_time

                    # Log comprehensive timing summary
                    print(f"✅ Message #{message_counter} completed in {total_message_time:.3f} seconds")
                    print(f"Timing breakdown: {timings}")

                    # Send timing info to client (optional)
                    timing_summary = f"⏱️ Processing times - Total: {total_message_time:.3f}s, Transcription: {transcription_time:.3f}s, Chat: {chat_time:.3f}s"
                    await ws.send_text(timing_summary)

                except Exception as transcription_error:
                    total_message_time = time.perf_counter() - message_start_time
                    timings['total'] = total_message_time
                    print(
                        f"❌ Message #{message_counter} transcription failed after {total_message_time:.3f} seconds: {str(transcription_error)}")
                    print(f"Timing breakdown: {timings}")
                    await ws.send_text(f"❌ Transcription error: {str(transcription_error)}")
                    continue

            except Exception as e:
                total_message_time = time.perf_counter() - message_start_time
                if "receive_bytes" in str(e):
                    print(
                        f"⚠️ Message #{message_counter} - Client sent text instead of audio (after {total_message_time:.3f}s)")
                    await ws.send_text("⚠️  Please send audio data as binary, not text")
                else:
                    print(f"❌ Message #{message_counter} error after {total_message_time:.3f}s: {str(e)}")
                    await ws.send_text(f"❌ Error processing audio: {str(e)}")
                break

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
    except Exception as e:
        try:
            await ws.send_text(f"❌ Error: {str(e)}")
        except:
            pass
        print(f"❌ Error in conversation_stream: {str(e)}")
