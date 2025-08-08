import os
from dotenv import load_dotenv
from google import genai
from chat_model.chatbot import chat_api_sync, chat_stream_websocket, summarize_conversation
from audio_processing.transcriber import transcribe_audio_api
from chat_model.text_to_speech import generate_tts_wav_api
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel
import time
import requests
from chat_model.scoring.score_model import (evaluate_pause, evaluate_repetition, evaluate_transcription,
                                            evaluate_vocabulary, evaluate_vocabulary_cefr)

load_dotenv()
app = FastAPI()
gemini_key = os.getenv("GEMINI_KEY")
client = genai.Client(api_key=gemini_key)

class ChatInput(BaseModel):
    selected_topic_name: str = "General"
    user_input: str
    history_log: list[tuple[str, str]]
    exchange_count: int = 0
    tts_model: str = "aura-2-thalia-en"

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

# Retrieve the history log and exchange count from the POST request of /chat
@app.post("/evaluate/")
def evaluate_grammar(input: EvaluationInput):
    user_messages = " ".join(
        msg for role, msg in input.history_log[-(input.exchange_count * 2):] if role == "user"
    )
    grammar_score, corrected_transcript = evaluate_transcription(user_messages)
    vocabulary_score = evaluate_vocabulary_cefr(user_messages)
    return {"original message": user_messages,
            "corrected_transcript": corrected_transcript,
            "evaluation_score": grammar_score,
            "vocabulary_score": vocabulary_score}



