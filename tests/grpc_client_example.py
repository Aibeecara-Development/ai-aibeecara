import wave
import grpc

from src import ai_service_pb2 as pb
from src import ai_service_pb2_grpc as pbg


def run_first_message():
    channel = grpc.insecure_channel("localhost:50051")
    stub = pbg.AiServiceStub(channel)
    req = pb.SpeakingRequest(
        topic="General",
        user_audio_url="",   # FIRST Speaking: empty means “ask bot to start”
        history_log=[pb.HistoryItem(role="user", text="Hi!")],  # optional
        accent="american",
        gender="feminine",
        speed=1.0,
    )

    wav = wave.open("first_message.wav", "wb")
    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)

    try:
        for ev in stub.ProcessSpeaking(req):
            which = ev.WhichOneof("event")
            if which == "bot":
                print("BOT:", ev.bot.bot_text)
            elif which == "tts":
                wav.writeframes(ev.tts.pcm16)
            elif which == "done":
                print("DONE (first message)")
                break
            elif which == "error":
                print("ERROR:", ev.error.message); break
    finally:
        wav.close()


def run_follow_up():
    channel = grpc.insecure_channel("localhost:50051")
    stub = pbg.AiServiceStub(channel)
    req = pb.SpeakingRequest(
        topic="General",
        user_audio_url="https://storage.googleapis.com/aibeecara-dev-bucket/speaking/test-audio.wav",  # replace
        history_log=[
            pb.HistoryItem(role="user", text="I wake up early"),
            pb.HistoryItem(role="bot",  text="Good job! What do you do after?"),
        ],
        accent="american",
        gender="feminine",
        speed=1.0,
    )

    wav = wave.open("follow_up.wav", "wb")
    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)

    try:
        for ev in stub.ProcessSpeaking(req):
            which = ev.WhichOneof("event")
            if which == "transcript":
                print("TRANSCRIPT:", ev.transcript.user_text)
            elif which == "scores":
                print("GRAMMAR:", ev.scores.grammar_score)
            elif which == "bot":
                print("BOT:", ev.bot.bot_text)
            elif which == "tts":
                wav.writeframes(ev.tts.pcm16)
            elif which == "done":
                print("DONE (follow-up)")
                break
            elif which == "error":
                print("ERROR:", ev.error.message); break
    finally:
        wav.close()


if __name__ == "__main__":
    run_first_message()
    run_follow_up()
