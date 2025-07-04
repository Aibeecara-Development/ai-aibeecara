import os
import json
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from dotenv import load_dotenv
import whisper
from whisper import transcribe

load_dotenv()
deepgram_key = os.getenv('DEEPGRAM_KEY')

def transcribe_deepgram(audio_path):
    """Transcribe audio using the Deepgram API."""
    try:
        deepgram = DeepgramClient(deepgram_key)
        with open(audio_path, "rb") as audio_file:
            buffer_data = audio_file.read()

        payload: FileSource = {"buffer": buffer_data}
        options = PrerecordedOptions(model="nova-3", smart_format=True)

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