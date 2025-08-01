import os
from dotenv import load_dotenv
from google import genai
from chat_model.chatbot import chat_stream
from audio_processing.transcriber import transcribe_audio_api
from chat_model.text_to_speech import generate_tts_wav_api
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel
import time
from chat_model.scoring.score_model import evaluate_pause, evaluate_repetition

load_dotenv()
app = FastAPI()
gemini_key = os.getenv("GEMINI_KEY")
client = genai.Client(api_key=gemini_key)

class ChatInput(BaseModel):
    selected_topic_name: str
    user_input: str
    history_log: list[tuple[str, str]]
    exchange_count: int = 0
    tts_model: str = "aura-2-thalia-en"

class TTSInput(BaseModel):
    text: str
    accent: str = "american"
    gender: str = "feminine"
    speed: float = 1.0

def mock_stream_response(user_input):
    reply = f"{user_input}"
    for word in reply.split():
        yield word + " "
        time.sleep(0.2)

@app.post("/chat/")
def chat_endpoint(input: ChatInput):
    return StreamingResponse(
        chat_stream(client, input.model_dump()),
        media_type="text/plain"
    )

@app.post("/transcribe/")
async def transcribe_endpoint(file: UploadFile = File(...)):
    try:
        audio_data = await file.read()
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



