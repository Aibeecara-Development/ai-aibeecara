import os
import json
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from dotenv import load_dotenv

load_dotenv('../.env')
deepgram_key = os.getenv('DEEPGRAM_KEY')

def transcribe_audio(audio_path):
    """Transcribe audio using the Deepgram API."""
    try:
        deepgram = DeepgramClient(deepgram_key)
        with open(audio_path, "rb") as audio_file:
            buffer_data = audio_file.read()

        payload: FileSource = {"buffer": buffer_data}
        options = PrerecordedOptions(model="nova-3", smart_format=True)

        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

        # Extract transcript from response
        transcript = response.to_dict()["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcript.lower()
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return None

def process_audio(audio_directory):
    """Process all audio files in the specified directory and return their transcriptions."""
    transcriptions = {}
    for file_name in os.listdir(audio_directory):
        if file_name.endswith((".mp3", ".wav", ".m4a", ".flac")):
            audio_path = os.path.join(audio_directory, file_name)
            print(f"Transcribing: {audio_path}")
            transcription = transcribe_audio(audio_path)
            if transcription:
                transcriptions[file_name] = transcription
    return transcriptions