import asyncio
import os
import threading
import uuid
from typing import AsyncGenerator, List

from dotenv import load_dotenv
import grpc
from google import genai

from src import ai_service_pb2 as pb
from src import ai_service_pb2_grpc as pbg
from src.chat_model.chatbot import (
    chat_api_sync,
    custom_topic_validation,
    hint_to_users,
)
from src.audio_processing.transcriber import transcribe_audio_api
from src.chat_model.text_to_speech import (
    generate_tts_pcm_stream,
    generate_tts_wav,
)
from src.chat_model.scoring.score_model import evaluate_transcription
from src.chat_model.scoring.vocab import evaluate_cefr_stats

from deep_translator import GoogleTranslator


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

    # Speaking (server-stream)
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
                    "user_input": "",  # no transcript
                    "history_log": history_pairs,
                    "exchange_count": len(history_pairs),
                    "tts_model": "aura-2-amalthea-en",
                    "initial": True,
                }
                bot_text = await chat_api_sync(self.genai_client, payload)

                # Bot response (new proto: BotText.text)
                yield pb.SpeakingEvent(bot_text=pb.BotText(text=bot_text))

                async for pcm in self._async_tts_chunks(bot_text, accent, gender, speed):
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

            async for pcm in self._async_tts_chunks(bot_text, accent, gender, speed):
                yield pb.SpeakingEvent(bot_audio=pb.BotAudio(pcm16=pcm))

            yield pb.SpeakingEvent(done=pb.Done())

        except Exception as e:
            yield pb.SpeakingEvent(error=pb.Error(message=str(e)))
            import traceback
            traceback.print_exc()

    # Topic validation
    async def ValidateTopic(
        self, request: pb.TopicValidationRequest, context: grpc.aio.ServicerContext
    ) -> pb.TopicValidationResponse:
        """
        Validates whether a topic is BROAD or NARROW.
        """
        try:
            topic_name = (request.topic_name or "").strip()
            if not topic_name:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "topic_name is required")

            validation_result = await custom_topic_validation(self.genai_client, topic_name)
            return pb.TopicValidationResponse(validation=validation_result)

        except grpc.RpcError:
            raise  # keep original status
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, f"Topic validation failed: {str(e)}")

    # Translation
    async def TranslateText(
        self, request: pb.TranslateRequest, context: grpc.aio.ServicerContext
    ) -> pb.TranslateResponse:
        text = (request.text or "").strip()
        if not text:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "text is required")

        try:
            # deep_translator is sync; run off the event loop
            def _translate() -> str:
                return GoogleTranslator(source="en", target="id").translate(text)

            translated = await self._to_thread(_translate)
            return pb.TranslateResponse(translated_text=translated)

        except grpc.RpcError:
            raise
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, f"Translation failed: {str(e)}")

    # Hint
    async def GenerateHint(
        self, request: pb.HintRequest, context: grpc.aio.ServicerContext
    ) -> pb.HintResponse:
        response_text = (request.response_text or "").strip()
        if not response_text:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "response_text is required")

        try:
            hint = await hint_to_users(self.genai_client, response_text)
            return pb.HintResponse(hint=hint)

        except grpc.RpcError:
            raise
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, f"Hint generation failed: {str(e)}")

    async def EvaluateGrammar(
        self, request: pb.EvaluateTranscriptRequest, context: grpc.aio.ServicerContext
    ) -> pb.EvaluateGrammarResponse:
        """
        1) evaluate_transcription (score, corrected_text, explanation, tense)
        2) generate_tts_wav(corrected_text or original)
        If ANY step fails, abort the RPC (no partial success).
        """
        transcript = (request.transcript or "").strip()
        if not transcript:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "transcript is required")

        # If proto later adds accent/gender, use them; otherwise default.
        accent = request.tts_accent or "american"
        gender = request.tts_gender or "feminine"

        try:
            # Step 1: Grammar evaluation
            def _run_eval():
                return evaluate_transcription(transcript)

            (
                evaluate_transcription_score,
                corrected_text,
                grammar_explanation,
                tense_used,
            ) = await self._to_thread(_run_eval)

            # Step 2: TTS
            corrected_audio_bytes = await self._to_thread(
                generate_tts_wav, corrected_text, accent, gender
            )

            # Success only if both steps succeeded
            return pb.EvaluateGrammarResponse(
                score=evaluate_transcription_score,
                corrected_transcript=corrected_text,
                explanation=grammar_explanation,
                tense_used=tense_used,
                corrected_audio=corrected_audio_bytes,
            )

        except grpc.RpcError:
            # Preserve any upstream gRPC status
            raise
        except Exception as e:
            # ONE ERROR => ALL ERROR (abort the entire RPC)
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Evaluate grammar failed: {str(e)}"
            )

    async def EvaluateVocabulary(
        self, request: pb.EvaluateTranscriptRequest, context: grpc.aio.ServicerContext
    ) -> pb.EvaluateVocabularyResponse:
        """
        1) evaluate_cefr_stats (statistics, tokens with synonyms)
        2) generate_tts_wav for each example sentence
        If ANY step fails, abort the RPC (no partial success).
        """
        transcript = (request.transcript or "").strip()
        if not transcript:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "transcript is required")

        accent = request.tts_accent or "american"
        gender = request.tts_gender or "feminine"

        try:
            # Step 1: Vocabulary evaluation
            def _run_vocab_eval():
                return evaluate_cefr_stats(transcript)

            cefr_data = await self._to_thread(_run_vocab_eval)

            # Step 2: Build response with TTS for example sentences
            statistics = cefr_data.get("statistics", {})
            tokens_data = cefr_data.get("tokens", [])

            # Convert tokens to protobuf format
            vocab_tokens = []
            for token_data in tokens_data:
                # Generate TTS for original word's example sentence
                example_sentence = token_data.get("example_sentence", "")
                example_audio = await self._to_thread(
                    generate_tts_wav, example_sentence, accent, gender
                ) if example_sentence else b""

                # Build original entry
                original_entry = pb.VocabularyEntry(
                    word=token_data.get("word", ""),
                    cefr=token_data.get("cefr", "NA"),
                    pronunciation=token_data.get("pronunciation", ""),
                    definition=token_data.get("definition", ""),
                    example_sentence_transcript=example_sentence,
                    example_sentence_audio=example_audio,
                )

                # Process synonyms if they exist
                synonym_entries = []
                synonyms_list = token_data.get("synonyms", [])
                for syn_data in synonyms_list:
                    syn_example = syn_data.get("example_sentence", "")
                    syn_audio = await self._to_thread(
                        generate_tts_wav, syn_example, accent, gender
                    ) if syn_example else b""

                    synonym_entry = pb.VocabularyEntry(
                        word=syn_data.get("synonym", ""),
                        cefr=syn_data.get("cefr", "NA"),
                        pronunciation=syn_data.get("pronunciation", ""),
                        definition=syn_data.get("definition", ""),
                        example_sentence_transcript=syn_example,
                        example_sentence_audio=syn_audio,
                    )
                    synonym_entries.append(synonym_entry)

                vocab_token = pb.VocabularyToken(
                    original=original_entry,
                    synonyms=synonym_entries,
                )
                vocab_tokens.append(vocab_token)

            return pb.EvaluateVocabularyResponse(
                statistics=statistics,
                tokens=vocab_tokens,
            )

        except grpc.RpcError:
            raise
        except Exception as e:
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Evaluate vocabulary failed: {str(e)}"
            )

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
