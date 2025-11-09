import os
from dotenv import load_dotenv
from google import genai
from .chat_model.chatbot import (chat_api_sync, chat_stream_websocket, summarize_conversation, custom_topic_validation,
                                chat_task, hint_to_users)
from .chat_model.emotion_detection import detect_emotion
from .audio_processing.transcriber import transcribe_audio_api, transcription_task
from .chat_model.text_to_speech import generate_tts_pcm_stream, tts_stream, generate_tts_wav
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel
import time
import requests
import io
from pydub import AudioSegment
from deepgram import DeepgramClient
import torchaudio
import asyncio
from typing import List, Optional, Any, Dict
from statistics import mean
from enum import Enum
from deep_translator import GoogleTranslator
from .pronunciation_model.pronunciation_model import evaluate_pronunciation
from .chat_model.scoring.score_model import (evaluate_pause, evaluate_stutter, evaluate_transcription,
                                            evaluate_vocabulary_cefr, calculate_speech_rates)
from .utils.utils import serialize_waveform, deserialize_waveform
from .chat_model.scoring.vocab import evaluate_cefr_stats

load_dotenv()
app = FastAPI()
gemini_key = os.getenv("GEMINI_KEY")
client = genai.Client(api_key=gemini_key)
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

class TTSWavInput(BaseModel):
    text: str
    accent: str = "american"   # e.g., american | british | australian
    gender: str = "feminine"   # e.g., feminine | masculine

class AudioURLInput(BaseModel):
    audio_url: str

class EvaluationInput(BaseModel):
    history_log: list[tuple[str, str]]
    exchange_count: int
    grammar_array: list[float] = []
    pronunciation_array: list[float] = []
    pause_array: list[float] = []
    stutter_array: list[float] = []

class ChatbotOutput(BaseModel):
    response: str

class CustomTopicInput(BaseModel):
    selected_topic_name: str

class ChatEmotion(BaseModel):
    model_output: str

class CorrectionItem(BaseModel):
    chat_bubble_id: int
    score: float
    transcript: str = ""

class CorrectionAspect(str, Enum):
    grammar = "grammar"
    fluency = "fluency"
    vocabulary = "vocabulary"
    pronunciation = "pronunciation"

class AspectScore(BaseModel):
    grammar_score: float
    vocabulary_score: str
    pronunciation_score: float
    fluency_score: float

class TranscriptionResult(BaseModel):
    response: Dict[str, Any]
    transcript: str
    waveform: str

class CorrectionScore(BaseModel):
    corrections: List[CorrectionItem]
    aspect_score: AspectScore
    aspect: CorrectionAspect
    audio_url: str
    chat_bubble_correction_id: int


def mock_stream_response(user_input):
    reply = f"{user_input}"
    for word in reply.split():
        yield word + " "
        time.sleep(0.2)

@app.post("/chat/")
async def chat_endpoint(chat_input: ChatInput):
    response_text = await chat_api_sync(client, chat_input.model_dump())

    return {
        "response": response_text,
    }

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
        # Download audio
        resp = requests.get(input_data.audio_url, timeout=30)
        resp.raise_for_status()
        audio_data = io.BytesIO(resp.content)

        # Convert to waveform
        audio = AudioSegment.from_file(audio_data, format="wav")
        audio_data = io.BytesIO(resp.content)
        waveform, sr = torchaudio.load(audio_data)

        # Resample and mono
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Transcribe via Deepgram (your function)
        response, transcript = transcribe_audio_api(input_data.audio_url)

        return {
            "response": response,
            "transcript": transcript,
            "waveform": serialize_waveform(waveform)
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/evaluate_chat_bubble/")
async def evaluate_endpoint(input_data: TranscriptionResult):
    try:
        response = input_data.response
        transcript = input_data.transcript
        waveform = deserialize_waveform(input_data.waveform)

        # Run evaluations
        start = time.time()
        pause_score_dict = evaluate_pause(response)
        print(f"evaluate_pause took {time.time() - start:.4f} seconds")

        start = time.time()
        stutter_score, stuttered_phrases = evaluate_stutter(response)
        print(f"evaluate_stutter took {time.time() - start:.4f} seconds")

        start = time.time()
        speech_rate_dict = calculate_speech_rates(response)
        print(f"calculate_speech_rates took {time.time() - start:.4f} seconds")

        start = time.time()
        pronunciation_score_dict = evaluate_pronunciation(waveform, transcript)
        print(f"evaluate_pronunciation took {time.time() - start:.4f} seconds")

        start = time.time()
        evaluate_transcription_score, corrected_text, grammar_explanation, tense_used = evaluate_transcription(
            transcript)
        print(f"evaluate_transcription took {time.time() - start:.4f} seconds")

        start = time.time()
        cefr_score_dict = evaluate_cefr_stats(transcript)
        print(f"evaluate_cefr_stats took {time.time() - start:.4f} seconds")

        fluency_score = stutter_score + pause_score_dict['score'] / 2.0
        if fluency_score <= 0.3:
            speed = "Slow"
        elif fluency_score <= 0.6:
            speed = "Hesitant"
        else:
            speed = "Fluent"

        return {
            "transcript": transcript,
            "corrected_transcript": corrected_text,
            "grammar_score": evaluate_transcription_score,
            "grammar_explanation": grammar_explanation,
            "tense_used": tense_used,
            "pause_score": pause_score_dict,
            "stutter_score": stutter_score,
            "stuttered_phrases": stuttered_phrases,
            "fluency_score": fluency_score,
            "fluency_speed": speed,
            "speech_rate": speech_rate_dict,
            "pronunciation_score": pronunciation_score_dict,
            "vocabulary_score": cefr_score_dict
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# class EvaluatePronunciationInput(BaseModel):
#     audio_url: str
#     transcript: str
#
# @app.post("/evaluate/pronunciation/")
# async def evaluate_pronunciation_endpoint(input_data: EvaluatePronunciationInput):
#     resp = requests.get(input_data.audio_url, timeout=30)
#     resp.raise_for_status()
#     audio_data = io.BytesIO(resp.content)
#     waveform, sr = torchaudio.load(audio_data)
#
#     # Resample and mono
#     if sr != 16000:
#         waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
#     if waveform.shape[0] > 1:
#         waveform = waveform.mean(dim=0, keepdim=True)
#
#     pronunciation_score_dict = evaluate_pronunciation(waveform, input_data.transcript)
#     return {"pronunciation_score": pronunciation_score_dict}

@app.post("/chat/tts/")
async def chat_tts(input: TTSInput):
    try:
        gen = generate_tts_pcm_stream(input.text, input.accent, input.gender, input.speed)
        # We return raw PCM; the frontend decodes/plays it via AudioWorklet.
        return StreamingResponse(gen, media_type="application/octet-stream")
    except Exception as e:
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
        sentence = input.user_input
        emotion = await asyncio.to_thread(detect_emotion(sentence))
        return {"emotion": emotion}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Retrieve the history log and exchange count from the POST request of /chat
@app.post("/evaluate/")
def evaluate_grammar(input: EvaluationInput):
    vocab_score_mapping = {
        "A1": 0.2,
        "A2": 0.4,
        "B1": 0.6,
        "B2": 0.8,
        "C1": 0.9,
        "C2": 1.0
    }
    user_messages = " ".join(
        msg for role, msg in input.history_log[-(input.exchange_count * 2):] if role == "user"
    )
    # grammar_score, corrected_transcript = evaluate_transcription(user_messages)
    # vocabulary_stats = evaluate_cefr_stats(user_messages)
    grammar_scores = input.grammar_array
    grammar_score = sum(grammar_scores) / len(grammar_scores)
    vocabulary_score = evaluate_vocabulary_cefr(user_messages)
    vocabulary_score = vocab_score_mapping.get(vocabulary_score, 0.0)
    pronunciation_scores = input.pronunciation_array
    pronunciation_score = sum(pronunciation_scores) / len(pronunciation_scores)
    pause_scores = input.pause_array
    pause_score = sum(pause_scores) / len(pause_scores)
    stutter_scores = input.stutter_array
    stutter_score = sum(stutter_scores) / len(stutter_scores)
    fluency_score = (pause_score + stutter_score) / 2.0
    total_score = (grammar_score + vocabulary_score + pronunciation_score + fluency_score) / 4.0
    return {"grammar_score": grammar_score,
            "vocabulary_score": vocabulary_score,
            "pronunciation_score": pronunciation_score,
            "total_score": total_score,}

# is it async or not?
@app.post("/translate/")
def translate_text(input: ChatInput):
    try:
        translated_text = GoogleTranslator(source='auto', target='id').translate(input.user_input)
        return {"translated_text": translated_text}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/hint/")
async def hint_endpoint(input: ChatbotOutput):
    try:
        hint = await hint_to_users(client, input.response)
        return {"hint": hint}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def update_score(input_data: CorrectionScore, new_score: float) -> tuple[float, Optional[float]]:
    for item in input_data.corrections:
        if item.chat_bubble_id == input_data.chat_bubble_correction_id:
            if new_score > item.score:
                item.score = new_score
                new_aspect_mean = mean([c.score for c in input_data.corrections])
                return item.score, new_aspect_mean
            return item.score, None
    raise ValueError(f"No correction found for chat_bubble_id={input_data.chat_bubble_correction_id}")

@app.post("/try_by_yourself/")
async def try_by_yourself(input_data: CorrectionScore):
    try:
        resp = requests.get(input_data.audio_url, timeout=30)
        resp.raise_for_status()
        audio_data = resp.content

        # Transcribe
        response, transcript = transcribe_audio_api(audio_data)

        speed = None

        if input_data.aspect == CorrectionAspect.pronunciation:
            pronunciation_result = evaluate_pronunciation(audio_data, transcript)
            pronunciation_score = pronunciation_result['score']
            new_score, new_aspect_mean = update_score(input_data, pronunciation_score)
            input_data.aspect_score.pronunciation_score = pronunciation_score
        elif input_data.aspect == CorrectionAspect.grammar:
            grammar_score, corrected_text, grammar_explanation = evaluate_transcription(transcript)
            new_score, new_aspect_mean = update_score(input_data, grammar_score)
            input_data.aspect_score.grammar_score = grammar_score
        elif input_data.aspect == CorrectionAspect.vocabulary:
            updated_transcripts = []
            for item in input_data.corrections:
                if item.chat_bubble_id == input_data.chat_bubble_correction_id:
                    # replace with the newly transcribed text
                    item.transcript = transcript
                # collect transcripts (fallback empty if not present yet)
                updated_transcripts.append(getattr(item, "transcript", ""))

            # Step 3: join all transcripts into one text
            combined_transcript = " ".join(t for t in updated_transcripts if t.strip())

            # Step 4: evaluate vocabulary level on the whole joined transcript
            vocabulary_score = evaluate_vocabulary_cefr(combined_transcript)  # str: "A1"..."C2"

            # Step 5: update aspect_score
            input_data.aspect_score.vocabulary_score = vocabulary_score

            new_score = vocabulary_score
            new_aspect_mean = vocabulary_score
        else:  # fluency
            pause_score, pause_details = evaluate_pause(response)
            stutter_score, stuttered_phrases = evaluate_stutter(response)
            fluency_score = (pause_score + stutter_score) / 2.0
            new_score, new_aspect_mean = update_score(input_data, fluency_score)
            input_data.aspect_score.fluency_score = fluency_score
            if fluency_score <= 0.3:
                speed = "Slow"
            elif fluency_score <= 0.6:
                speed = "Hesitant"
            else:
                speed = "Fluent"

        # Always recalc total score with updated aspect values
        new_total_score = mean([
            input_data.aspect_score.grammar_score,
            input_data.aspect_score.vocabulary_score,
            input_data.aspect_score.pronunciation_score,
            input_data.aspect_score.fluency_score
        ])

        return {
            "new_score": new_score,
            "new_aspect_mean": new_aspect_mean,
            "new_total_score": new_total_score,
            "new_speed": speed
        }

    except Exception as e:
        return {"error": str(e)}

# Don't forget to integrate the evaluate_cefr_stats, evaluate_transcription,
# evaluate_pronunciation, evaluate_pause, and evaluate_stutter functions into
# the chatbot API so that it can be evaluated every time the user
# makes an input. In the end, the evaluation results are averaged and returned to the user.
@app.websocket("/conversation_stream/")
async def conversation_stream(ws: WebSocket, input: ChatInput):
    await ws.accept()

    chat_queue = asyncio.Queue()

    await asyncio.gather(
        transcription_task(ws, chat_queue, deepgram_client),
        chat_task(ws, chat_queue, client, lambda text: tts_stream(text, deepgram_client), input_data=input)
    )

    await chat_queue.put(None)
