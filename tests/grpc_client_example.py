import wave
import grpc

from src import ai_service_pb2 as pb
from src import ai_service_pb2_grpc as pbg


def run_first_message():
    """
    First turn:
    - user_audio_url="" (no audio) => server will start the conversation
    - Save streamed bot audio to first_message.wav (16kHz, 16-bit PCM mono)
    """
    channel = grpc.insecure_channel("localhost:50051")
    stub = pbg.AiServiceStub(channel)
    req = pb.SpeakingRequest(
        topic="General",
        user_audio_url="",  # FIRST Speaking: empty means “ask bot to start”
        history_log=[pb.HistoryItem(role="user", text="Hi!")],  # optional
        accent="american",
        gender="feminine",
        speed=1.0,
    )

    wav = wave.open("first_message.wav", "wb")
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)

    try:
        for ev in stub.ProcessSpeaking(req):
            which = ev.WhichOneof("event")
            if which == "start":
                print("START request_id:", ev.start.request_id)
            elif which == "bot_text":
                # New proto: BotText.text
                print("BOT:", ev.bot_text.text)
            elif which == "bot_audio":
                # New proto: event name 'bot_audio' with message BotAudio{ bytes pcm16 }
                wav.writeframes(ev.bot_audio.pcm16)
            elif which == "done":
                print("DONE (first message)")
                break
            elif which == "error":
                print("ERROR:", ev.error.message)
                break
    finally:
        wav.close()


def run_follow_up():
    """
    Follow-up turn:
    - Provide user_audio_url so the server transcribes it, returns UserText, BotText, and BotAudio (streaming)
    """
    channel = grpc.insecure_channel("localhost:50051")
    stub = pbg.AiServiceStub(channel)
    req = pb.SpeakingRequest(
        topic="General",
        user_audio_url="https://storage.googleapis.com/aibeecara-dev-bucket/speaking/test-audio.wav",  # replace
        history_log=[
            pb.HistoryItem(role="user", text="I wake up early"),
            pb.HistoryItem(role="bot", text="Good job! What do you do after?"),
        ],
        accent="american",
        gender="feminine",
        speed=1.0,
    )

    wav = wave.open("follow_up.wav", "wb")
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)

    try:
        for ev in stub.ProcessSpeaking(req):
            which = ev.WhichOneof("event")
            if which == "start":
                print("START request_id:", ev.start.request_id)
            elif which == "user_text":
                # New proto: UserText.text
                print("TRANSCRIPT:", ev.user_text.text)
            elif which == "bot_text":
                print("BOT:", ev.bot_text.text)
            elif which == "bot_audio":
                wav.writeframes(ev.bot_audio.pcm16)
            elif which == "done":
                print("DONE (follow-up)")
                break
            elif which == "error":
                print("ERROR:", ev.error.message)
                break
    finally:
        wav.close()


if __name__ == "__main__":
    run_first_message()
    run_follow_up()
