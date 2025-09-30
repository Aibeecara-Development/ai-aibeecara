import os
import json
from deepgram import DeepgramClient, PrerecordedOptions, FileSource, TextSource
from dotenv import load_dotenv
import whisper
import requests
from whisper import transcribe

load_dotenv()
deepgram_key = os.getenv('DEEPGRAM_KEY')
deepgram = DeepgramClient(deepgram_key)

async def transcription_task(ws, chat_queue, deepgram_client):
    async with deepgram_client.listen.websocket.v("1",
                                                  model="nova-3",
                                                  smart_format=True,
                                                  filler_words=True) as dg_ws:

        @dg_ws.on("transcript_received")
        async def handle_transcript(data):
            transcript_text = data["channel"]["alternatives"][0]["transcript"]

            # Only push when sentence ends
            if transcript_text.strip().endswith((".", "?", "!")):
                # --- run evaluations here ---
                cefr_level = evaluate_cefr_stats(transcript_text)
                grammar_score, corrected_text, grammar_explanation = evaluate_transcription(transcript_text)
                pause_score, pause_details = evaluate_pause(data)
                stutter_score, stuttered_phrases = evaluate_stutter(data)

                eval_result = {
                    "transcript": transcript_text,
                    "vocabulary": cefr_level,
                    "grammar": grammar_score,
                    "fluency": (pause_score + stutter_score) / 2.0,
                }

                # Put both transcript & eval results into queue
                await chat_queue.put(eval_result)

        try:
            while True:
                audio_chunk = await ws.receive_bytes()
                await dg_ws.send(audio_chunk)
        except:
            pass


def transcribe_audio_api(url: str):
    """Transcribes audio using Deepgram and returns the response and transcript text."""
    try:
        AUDIO_URL = {
            "url": url
        }
        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
        )
        response = deepgram.listen.rest.v("1").transcribe_url(AUDIO_URL, options)
        transcript = response.to_dict()["results"]["channels"][0]["alternatives"][0]["transcript"]
        return response.to_dict(), transcript
    except Exception as e:
        raise RuntimeError(f"Deepgram transcription error: {e}")

def transcribe_deepgram(audio_path):
    """Transcribe audio using the Deepgram API."""
    try:
        with open(audio_path, "rb") as audio_file:
            buffer_data = audio_file.read()

        payload: FileSource = {"buffer": buffer_data}
        options = PrerecordedOptions(model="nova-3", smart_format=True, filler_words=True)

        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

        return response
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return None

def transcribe_whisper(input_file):
    model = whisper.load_model("base")
    result = transcribe(model=model, audio=input_file)
    transcribed_text_whisper = result["text"]
    return transcribed_text_whisper

def process_audio(audio_directory):
    """Process all audio files in the specified directory and return their transcriptions."""
    transcriptions = {}
    for file_name in os.listdir(audio_directory):
        if file_name.endswith((".mp3", ".wav", ".m4a", ".flac")):
            audio_path = os.path.join(audio_directory, file_name)
            print(f"Transcribing: {audio_path}")
            transcription = transcribe_deepgram(audio_path)
            # transcription = transcribe_whisper(audio_path)
            if transcription:
                transcriptions[file_name] = transcription
    return transcriptions