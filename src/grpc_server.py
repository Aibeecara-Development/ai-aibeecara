import asyncio
import os
import threading
import uuid
from typing import AsyncGenerator, List

from dotenv import load_dotenv
import grpc

from src import ai_service_pb2 as pb
from src import ai_service_pb2_grpc as pbg

from google import genai
from src.chat_model.chatbot import chat_api_sync
from src.audio_processing.transcriber import transcribe_audio_api
from src.chat_model.text_to_speech import generate_tts_pcm_stream
from src.utils.utils import clean_text


def _collapse_history_pairs(history_items: List[pb.HistoryItem]) -> List[tuple[str, str]]:
    """
    Convert role-based history `[("user", "..."), ("bot","..."), ...]`
    to pairs List[(user_text, bot_text)] for your existing chat_api_sync.
    If a bot appears without a preceding user, pair with "".
    If a trailing user has no bot, pair with "" (best-effort).
    """
    pairs: List[tuple[str, str]] = []
    pending_user: str | None = None
    for it in history_items:
        role = (it.role or "").lower().strip()
        text = it.text or ""
        if role == "user":
            if pending_user is not None:
                pairs.append((pending_user, ""))
            pending_user = text
        elif role == "bot":
            if pending_user is None:
                pairs.append(("", text))
            else:
                pairs.append((pending_user, text))
                pending_user = None
        else:
            # unknown role: ignore gracefully
            pass
    if pending_user is not None:
        pairs.append((pending_user, ""))
    return pairs


class AiServiceServicer(pbg.AiServiceServicer):
    def __init__(self) -> None:
        load_dotenv()
        gemini_key = os.getenv("GEMINI_KEY")
        if not gemini_key:
            raise RuntimeError("GEMINI_KEY missing in .env")
        self.genai_client = genai.Client(api_key=gemini_key)

    async def _to_thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def _async_tts_chunks(
        self, text: str, accent: str, gender: str, speed: float
    ) -> AsyncGenerator[bytes, None]:
        """
        Bridges a sync generator (generate_tts_pcm_stream) into an async generator of PCM16 bytes.
        """
        loop = asyncio.get_running_loop()
        q: "asyncio.Queue[bytes | None]" = asyncio.Queue()

        def producer():
            try:
                for chunk in generate_tts_pcm_stream(text, accent, gender, speed):
                    loop.call_soon_threadsafe(q.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    # ==== Streaming RPC: ProcessSpeaking (updated to new proto) ====
    async def ProcessSpeaking(
        self, request: pb.SpeakingRequest, context: grpc.aio.ServicerContext
    ) -> AsyncGenerator[pb.SpeakingEvent, None]:
        request_id = str(uuid.uuid4())

        # Start event
        yield pb.SpeakingEvent(start=pb.Start(request_id=request_id))

        topic = request.topic or "General"
        audio_url = (request.user_audio_url or "").strip()
        history_pairs = _collapse_history_pairs(list(request.history_log))
        accent = request.accent or "american"
        gender = request.gender or "feminine"
        speed = request.speed if request.speed else 1.0

        try:
            # ---- FIRST TURN (no audio): get initial bot message then TTS ----
            if audio_url == "":
                payload = {
                    "selected_topic_name": topic,
                    "user_input": "",              # no transcript
                    "history_log": history_pairs,  # pairs for your model
                    "exchange_count": len(history_pairs),
                    "tts_model": "aura-2-amalthea-en",
                    "initial": True,
                }
                bot_text = await chat_api_sync(self.genai_client, payload)

                # Bot response (new proto: BotText.text)
                yield pb.SpeakingEvent(bot_text=pb.BotText(text=bot_text))

                cleaned = clean_text(bot_text)
                async for pcm in self._async_tts_chunks(cleaned, accent, gender, speed):
                    # New proto: event 'bot_audio' with message BotAudio{ pcm16 }
                    yield pb.SpeakingEvent(bot_audio=pb.BotAudio(pcm16=pcm))

                yield pb.SpeakingEvent(done=pb.Done())
                return

            # ---- FOLLOW-UP TURN (has audio): transcribe -> bot -> TTS ----
            dg_response, transcript = await self._to_thread(transcribe_audio_api, audio_url)

            # Return the user's transcript (new proto: UserText.text)
            yield pb.SpeakingEvent(user_text=pb.UserText(text=transcript))

            payload = {
                "selected_topic_name": topic,
                "user_input": transcript,
                "history_log": history_pairs,
                "exchange_count": len(history_pairs),
                "tts_model": "aura-2-amalthea-en",
            }
            bot_text = await chat_api_sync(self.genai_client, payload)

            # Bot response
            yield pb.SpeakingEvent(bot_text=pb.BotText(text=bot_text))

            cleaned = clean_text(bot_text)
            async for pcm in self._async_tts_chunks(cleaned, accent, gender, speed):
                yield pb.SpeakingEvent(bot_audio=pb.BotAudio(pcm16=pcm))

            yield pb.SpeakingEvent(done=pb.Done())

        except Exception as e:
            yield pb.SpeakingEvent(error=pb.Error(message=str(e)))
            import traceback
            traceback.print_exc()


async def serve(host: str = "[::]:50051") -> None:
    server = grpc.aio.server(options=[
        ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ])
    pbg.add_AiServiceServicer_to_server(AiServiceServicer(), server)
    server.add_insecure_port(host)
    print(f"gRPC server listening on {host}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
