import time
from dotenv import load_dotenv
import wave
import os
import soundfile as sf
import tempfile
import librosa
from deepgram import (
    DeepgramClient,
    SpeakWebSocketEvents,
    SpeakWSOptions,
)

load_dotenv()
deepgram_key = os.getenv('DEEPGRAM_KEY')
deepgram: DeepgramClient = DeepgramClient(deepgram_key)

# AUDIO_FILE = "output.wav"
# TTS_TEXT = "Hello, this is a text to speech example using Deepgram. How are you doing today? I am fine thanks for asking."
def generate_tts_wav_api(text: str, accent: str = "american", gender: str = "feminine", speed: float = 1.0) -> str:
    raw_output_path = tempfile.mktemp(suffix="_raw.wav")
    final_output_path = tempfile.mktemp(suffix=".wav")

    dg_connection = deepgram.speak.websocket.v("1")

    wav_writer = wave.open(raw_output_path, "wb")
    wav_writer.setnchannels(1)
    wav_writer.setsampwidth(2)
    wav_writer.setframerate(16000)

    def on_binary_data(self, data, **kwargs):
        wav_writer.writeframesraw(data)

    dg_connection.on(SpeakWebSocketEvents.AudioData, on_binary_data)

    model = "aura-2-amalthea-en"
    if accent == "british" and gender == "feminine":
        model = "aura-2-draco-en"
    elif accent == "british" and gender == "masculine":
        model = "aura-2-pandora-en"
    elif accent == "american" and gender == "feminine":
        model = "aura-2-amalthea-en"
    elif accent == "american" and gender == "masculine":
        model = "aura-2-apollo-en"

    options = SpeakWSOptions(
        model=model,
        encoding="linear16",
        sample_rate=16000,
    )

    if not dg_connection.start(options):
        raise RuntimeError("Failed to start Deepgram connection")

    dg_connection.send_text(text)
    dg_connection.flush()
    time.sleep(7)  # crude wait — consider replacing with audio length-based logic
    dg_connection.finish()
    wav_writer.close()

    if speed == 1.0:
        return raw_output_path
    else:
        # Adjust speed using librosa
        y, sr = librosa.load(raw_output_path, sr=16000)
        y_fast = librosa.effects.time_stretch(y, rate=speed)
        sf.write(final_output_path, y_fast, sr)

        return final_output_path



def transform_speech(file_path, spoken_text, model="aura-2-amalthea-en" ):
    try:
        # Create a websocket connection to Deepgram
        dg_connection = deepgram.speak.websocket.v("1")

        wav_writer = wave.open(file_path, "wb")
        wav_writer.setnchannels(1)
        wav_writer.setsampwidth(2)  # 16-bit
        wav_writer.setframerate(16000)

        def on_binary_data(self, data, **kwargs):
            wav_writer.writeframesraw(data)

        dg_connection.on(SpeakWebSocketEvents.AudioData, on_binary_data)

        # Generate a generic WAV container header
        # since we don't support containerized audio, we need to generate a header
        header = wave.open(file_path, "wb")
        header.setnchannels(1)  # Mono audio
        header.setsampwidth(2)  # 16-bit audio
        header.setframerate(16000)  # Sample rate of 16000 Hz
        header.close()

        # connect to websocket
        options = SpeakWSOptions(
            model=model,
            encoding="linear16",
            sample_rate=16000,
        )

        print("\n\nPress Enter to stop...\n\n")
        if dg_connection.start(options) is False:
            print("Failed to start connection")
            return

        # send the text to Deepgram
        dg_connection.send_text(spoken_text)

        # if auto_flush_speak_delta is not used, you must flush the connection by calling flush()
        dg_connection.flush()

        # Indicate that we've finished
        time.sleep(7)
        print("\n\nPress Enter to stop...\n\n")
        input()

        # Close the connection
        dg_connection.finish()
        wav_writer.close()

        print("Finished")

    except ValueError as e:
        print(f"Invalid value encountered: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
