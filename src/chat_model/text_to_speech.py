import queue
import time
from dotenv import load_dotenv
import wave
import os
import threading
import io
import asyncio
from typing import AsyncGenerator
from deepgram import (
    DeepgramClient,
    SpeakWebSocketEvents,
    SpeakWSOptions,
)
from scipy.signal import resample_poly
import numpy as np


load_dotenv()
deepgram_key = os.getenv('DEEPGRAM_KEY')
deepgram: DeepgramClient = DeepgramClient(deepgram_key)

async def tts_stream(text: str, deepgram_client) -> AsyncGenerator[bytes, None]:
    """
    Streams audio chunks from Deepgram TTS.
    """
    queue = asyncio.Queue()
    done_event = asyncio.Event()
    dg_connection = deepgram_client.speak.websocket.v("1")

    @dg_connection.on(SpeakWebSocketEvents.AudioData)
    def on_audio(self, data, **kwargs):
        asyncio.create_task(queue.put(data))

    @dg_connection.on(SpeakWebSocketEvents.Close)
    def on_close(**kwargs):
        done_event.set()

    await dg_connection.start(SpeakWSOptions(
        model="aura-2-amalthea-en",
        encoding="linear16",
        sample_rate=16000,

    ))

    dg_connection.send_text(text)
    dg_connection.flush()

    while not done_event.is_set():
        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=1)
            yield chunk
        except asyncio.TimeoutError:
            continue

    dg_connection.finish()

# AUDIO_FILE = "output.wav"
# TTS_TEXT = "Hello, this is a text to speech example using Deepgram. How are you doing today? I am fine thanks for asking."
def generate_tts_wav_api(
    text: str,
    accent: str = "american",
    gender: str = "feminine",
    speed: float = 1.0
):
    def audio_stream():
        buffer = io.BytesIO()
        wav_writer = wave.open(buffer, "wb")
        wav_writer.setnchannels(1)
        wav_writer.setsampwidth(2)
        wav_writer.setframerate(16000)

        done_event = threading.Event()

        # Callback for each audio chunk from Deepgram
        def on_binary_data(self, data, **kwargs):
            audio = np.frombuffer(data, dtype=np.int16)

            if speed != 1.0:
                # speed > 1.0 => faster, speed < 1.0 => slower
                up = int(speed * 100)
                down = 100
                audio = resample_poly(audio, up, down)

            yield audio.tobytes()

        # Callback when Deepgram signals end of stream
        def on_close(**kwargs):
            done_event.set()

        # Pick Deepgram model
        model = "aura-2-amalthea-en"
        if accent == "british" and gender == "feminine":
            model = "aura-2-draco-en"
        elif accent == "british" and gender == "masculine":
            model = "aura-2-pandora-en"
        elif accent == "australian":
            model = "aura-2-hyperion-en"

        # Connect to Deepgram
        dg_connection = deepgram.speak.websocket.v("1")
        dg_connection.on(SpeakWebSocketEvents.AudioData, on_binary_data)
        dg_connection.on(SpeakWebSocketEvents.Close, on_close)

        options = SpeakWSOptions(
            model=model,
            encoding="linear16",
            sample_rate=16000,
        )

        if not dg_connection.start(options):
            raise RuntimeError("Failed to start Deepgram connection")

        # Send input text
        dg_connection.send_text(text)
        dg_connection.flush()

        # Wait for completion
        timeout = max(10.0, len(text) / 5)
        done_event.wait(timeout=timeout)

        dg_connection.finish()
        wav_writer.close()

        buffer.seek(0)
        yield buffer.read()  # send any remaining buffered audio

    return audio_stream()


def generate_tts_pcm_stream(text: str, accent: str = "american", gender: str = "feminine", speed: float = 1.0):
    """
    Yields raw 16-bit little-endian PCM (mono, 16 kHz) bytes as they arrive.
    """
    SAMPLE_RATE = 16000
    q: "queue.Queue[bytes | None]" = queue.Queue(maxsize=64)
    done_event = threading.Event()

    # Pick Deepgram voice/model
    model = "aura-2-amalthea-en"
    if accent == "british" and gender == "feminine":
        model = "aura-2-draco-en"
    elif accent == "british" and gender == "masculine":
        model = "aura-2-pandora-en"
    elif accent == "australian":
        model = "aura-2-hyperion-en"

    # Callback for Deepgram audio chunks
    def on_binary_data(_, data: bytes, **kwargs):
        # Incoming bytes are linear16
        audio = np.frombuffer(data, dtype=np.int16)

        # Apply speed change by resampling if requested
        if speed and abs(speed - 1.0) > 1e-6:
            up = max(1, int(round(speed * 100)))
            down = 100
            audio = resample_poly(audio, up, down).astype(np.int16)

        # Push bytes to queue (drop if the reader is too slow)
        try:
            q.put_nowait(audio.tobytes())
        except queue.Full:
            # Drop frames to keep latency bounded (no-seek stream)
            pass

    def on_close(**kwargs):
        done_event.set()
        try:
            q.put_nowait(None)
        except queue.Full:
            pass

    # Start the Deepgram streaming TTS
    dg_connection = deepgram.speak.websocket.v("1")
    dg_connection.on(SpeakWebSocketEvents.AudioData, on_binary_data)
    dg_connection.on(SpeakWebSocketEvents.Close, on_close)

    options = SpeakWSOptions(
        model=model,
        encoding="linear16",
        sample_rate=SAMPLE_RATE,
    )

    if not dg_connection.start(options):
        raise RuntimeError("Failed to start Deepgram connection")

    # Send text and flush
    dg_connection.send_text(text)
    dg_connection.flush()

    # The generator that FastAPI will iterate
    def iterator():
        # Optional: short leading silence to prime the player
        yield (np.zeros(int(0.05 * SAMPLE_RATE), dtype=np.int16)).tobytes()

        # Pull from queue until closed
        while True:
            chunk = q.get()
            if chunk is None and done_event.is_set():
                break
            if chunk:
                yield chunk

        dg_connection.finish()

    return iterator()


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

# filename = "output_api.mp3"
# SPEAK_TEXT = {"text": """In the heart of every forest, a hidden world thrives among the towering trees. Trees,
# those silent giants, are more than just passive observers of nature's drama; they are
# active participants in an intricate dance of life."""}
# options = SpeakOptions(model="aura-2-amalthea-en")
# response = deepgram.speak.rest.v("1").save(filename, SPEAK_TEXT, options)
# print(response.to_json(indent=4))