import asyncio
import io
import os
import threading
import uuid
from typing import AsyncGenerator, List

import requests
import torchaudio
from dotenv import load_dotenv
import grpc
from google import genai

from protos import ai_service_pb2 as pb
from protos import ai_service_pb2_grpc as pbg
from protos import backend_service_pb2 as backend_pb
from protos import backend_service_pb2_grpc as backend_pbg
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
from src.chat_model.scoring.score_model import (
    evaluate_transcription,
    evaluate_pause,
    evaluate_stutter,
    calculate_speech_rates,
)
from src.chat_model.scoring.vocab import evaluate_cefr_stats
from src.pronunciation_model.pronunciation_model import evaluate_pronunciation

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

        # Backend service connection
        backend_url = os.getenv("BACKEND_SERVICE_URL", "localhost:50052")
        self.backend_channel = grpc.aio.insecure_channel(backend_url)
        self.backend_stub = backend_pbg.BackendServiceStub(self.backend_channel)

        # Limit concurrent TTS requests to avoid rate limiting
        self.tts_semaphore = asyncio.Semaphore(2)  # Max concurrent TTS requests

    async def _to_thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def _rate_limited_tts(self, text: str, accent: str, gender: str) -> bytes:
        """Generate TTS with rate limiting to avoid API 429 errors."""
        async with self.tts_semaphore:
            return await self._to_thread(generate_tts_wav, text, accent, gender)

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

    async def _evaluate_and_notify_backend(
        self,
        session_id: str,
        message_id: str,
        audio_url: str,
        transcript: str,
        accent: str,
        gender: str,
        deepgram_response: dict = None,
    ) -> None:
        """
        Run all four evaluations (grammar, vocabulary, pronunciation, fluency) and notify backend.
        This runs in the background and does not block the speaking stream.
        """
        try:
            # Run all evaluations in parallel
            grammar_task = self._evaluate_grammar(transcript, accent, gender)
            vocab_task = self._evaluate_vocabulary(transcript, accent, gender)
            pronunciation_task = self._evaluate_pronunciation(audio_url, transcript, accent, gender)
            fluency_task = None
            if deepgram_response:
                fluency_task = self._evaluate_fluency(deepgram_response)

            # Wait for all to complete
            tasks = [grammar_task, vocab_task, pronunciation_task]
            if fluency_task:
                tasks.append(fluency_task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            grammar_result = results[0]
            vocab_result = results[1]
            pronunciation_result = results[2]
            fluency_result = results[3] if len(results) > 3 else None

            # Notify backend for each successful evaluation
            # Grammar
            if isinstance(grammar_result, Exception):
                print(f"Grammar evaluation failed: {grammar_result}")
            else:
                try:
                    await self.backend_stub.NotifyGrammarEvaluation(
                        backend_pb.EvaluateGrammarResponse(
                            session_id=session_id,
                            message_id=message_id,
                            **grammar_result
                        )
                    )
                    print(f"✓ Grammar evaluation sent to backend for session {session_id}")
                except Exception as e:
                    print(f"Failed to notify backend about grammar: {e}")

            # Vocabulary
            if isinstance(vocab_result, Exception):
                print(f"Vocabulary evaluation failed: {vocab_result}")
            else:
                try:
                    await self.backend_stub.NotifyVocabularyEvaluation(
                        backend_pb.EvaluateVocabularyResponse(
                            session_id=session_id,
                            message_id=message_id,
                            **vocab_result
                        )
                    )
                    print(f"✓ Vocabulary evaluation sent to backend for session {session_id}")
                except Exception as e:
                    print(f"Failed to notify backend about vocabulary: {e}")

            # Pronunciation
            if isinstance(pronunciation_result, Exception):
                print(f"Pronunciation evaluation failed: {pronunciation_result}")
            else:
                try:
                    await self.backend_stub.NotifyPronunciationEvaluation(
                        backend_pb.EvaluatePronunciationResponse(
                            session_id=session_id,
                            message_id=message_id,
                            **pronunciation_result
                        )
                    )
                    print(f"✓ Pronunciation evaluation sent to backend for session {session_id}")
                except Exception as e:
                    print(f"Failed to notify backend about pronunciation: {e}")

            # Fluency
            if fluency_result is not None:
                if isinstance(fluency_result, Exception):
                    print(f"Fluency evaluation failed: {fluency_result}")
                else:
                    try:
                        await self.backend_stub.NotifyFluencyEvaluation(
                            backend_pb.EvaluateFluencyResponse(
                                session_id=session_id,
                                message_id=message_id,
                                score=int(fluency_result['fluency_score'] * 100),
                                words_per_minute=fluency_result['words_per_minute'],
                                syllables_per_minute=fluency_result['syllables_per_minute'],
                            )
                        )
                        print(f"✓ Fluency evaluation sent to backend for session {session_id}")
                    except Exception as e:
                        print(f"Failed to notify backend about fluency: {e}")

        except Exception as e:
            print(f"Background evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    async def _evaluate_grammar(self, transcript: str, accent: str, gender: str) -> dict:
        """Evaluate grammar and return results as dict."""
        def _run_eval():
            return evaluate_transcription(transcript)

        (
            evaluate_transcription_score,
            corrected_text,
            grammar_explanation,
            tense_used,
        ) = await self._to_thread(_run_eval)

        corrected_audio_bytes = await self._rate_limited_tts(corrected_text, accent, gender)

        return {
            "score": evaluate_transcription_score,
            "corrected_transcript": corrected_text,
            "explanation": grammar_explanation,
            "tense_used": tense_used,
            "corrected_audio": corrected_audio_bytes,
        }

    async def _evaluate_vocabulary(self, transcript: str, accent: str, gender: str) -> dict:
        """Evaluate vocabulary and return results as dict."""
        def _run_vocab_eval():
            return evaluate_cefr_stats(transcript)

        cefr_data = await self._to_thread(_run_vocab_eval)

        statistics = cefr_data.get("statistics", {})
        tokens_data = cefr_data.get("tokens", [])

        # Collect all TTS tasks
        tts_tasks = []
        tts_metadata = []

        for token_idx, token_data in enumerate(tokens_data):
            example_sentence = token_data.get("example_sentence", "")
            if example_sentence:
                tts_tasks.append(self._rate_limited_tts(example_sentence, accent, gender))
            else:
                tts_tasks.append(asyncio.sleep(0, result=b""))
            tts_metadata.append({"token_idx": token_idx, "is_synonym": False, "syn_idx": None})

            synonyms_list = token_data.get("synonyms", [])
            for syn_idx, syn_data in enumerate(synonyms_list):
                syn_example = syn_data.get("example_sentence", "")
                if syn_example:
                    tts_tasks.append(self._rate_limited_tts(syn_example, accent, gender))
                else:
                    tts_tasks.append(asyncio.sleep(0, result=b""))
                tts_metadata.append({"token_idx": token_idx, "is_synonym": True, "syn_idx": syn_idx})

        tts_results = await asyncio.gather(*tts_tasks)

        # Build result map
        result_map = {}
        for metadata, audio_bytes in zip(tts_metadata, tts_results):
            token_idx = metadata["token_idx"]
            if token_idx not in result_map:
                result_map[token_idx] = {"original_audio": None, "synonym_audios": {}}

            if metadata["is_synonym"]:
                result_map[token_idx]["synonym_audios"][metadata["syn_idx"]] = audio_bytes
            else:
                result_map[token_idx]["original_audio"] = audio_bytes

        # Convert to protobuf format
        vocab_tokens = []
        for token_idx, token_data in enumerate(tokens_data):
            example_sentence = token_data.get("example_sentence", "")
            example_audio = result_map[token_idx]["original_audio"]

            original_entry = backend_pb.VocabularyEntry(
                word=token_data.get("word", ""),
                cefr=token_data.get("cefr", "NA"),
                pronunciation=token_data.get("pronunciation", ""),
                definition=token_data.get("definition", ""),
                example_sentence_transcript=example_sentence,
                example_sentence_audio=example_audio,
            )

            synonym_entries = []
            synonyms_list = token_data.get("synonyms", [])
            for syn_idx, syn_data in enumerate(synonyms_list):
                syn_example = syn_data.get("example_sentence", "")
                syn_audio = result_map[token_idx]["synonym_audios"].get(syn_idx, b"")

                synonym_entry = backend_pb.VocabularyEntry(
                    word=syn_data.get("synonym", ""),
                    cefr=syn_data.get("cefr", "NA"),
                    pronunciation=syn_data.get("pronunciation", ""),
                    definition=syn_data.get("definition", ""),
                    example_sentence_transcript=syn_example,
                    example_sentence_audio=syn_audio,
                )
                synonym_entries.append(synonym_entry)

            vocab_token = backend_pb.VocabularyToken(
                original=original_entry,
                synonyms=synonym_entries,
            )
            vocab_tokens.append(vocab_token)

        return {"statistics": statistics, "tokens": vocab_tokens}

    async def _evaluate_pronunciation(self, audio_url: str, transcript: str, accent: str, gender: str) -> dict:
        """Evaluate pronunciation and return results as dict."""
        # Download audio
        def _download_audio():
            resp = requests.get(audio_url, timeout=30)
            resp.raise_for_status()
            return io.BytesIO(resp.content)

        audio_data = await self._to_thread(_download_audio)

        # Process audio
        def _process_audio():
            waveform, sr = torchaudio.load(audio_data)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform

        waveform = await self._to_thread(_process_audio)

        # Evaluate pronunciation
        def _evaluate_pronunciation():
            return evaluate_pronunciation(waveform, transcript)

        pronunciation_score_dict = await self._to_thread(_evaluate_pronunciation)

        overall_score = round(pronunciation_score_dict['score'])
        words_data = pronunciation_score_dict['words']

        # Filter imperfect words and collect TTS tasks
        imperfect_words = []
        tts_tasks = []

        for word_data in words_data:
            score = round(word_data.get('Pronunciation score', 0)) if word_data.get('Pronunciation score') is not None else 0

            if score < 100:
                imperfect_words.append(word_data)

                corrected_word = word_data.get('Real words', '')
                if corrected_word:
                    tts_tasks.append(self._rate_limited_tts(corrected_word, accent, gender))
                else:
                    tts_tasks.append(asyncio.sleep(0, result=b""))

        if tts_tasks:
            tts_results = await asyncio.gather(*tts_tasks)
        else:
            tts_results = []

        # Build response
        tokens = []
        for i, word_data in enumerate(imperfect_words):
            score = round(word_data.get('Pronunciation score', 0)) if word_data.get('Pronunciation score') is not None else 0
            word = word_data.get('Real words', '')
            wrong_transcript = word_data.get('Transcribed words', '')
            corrected_transcript = word_data.get('Real words', '')
            corrected_ipa = word_data.get('Ground truth phonemes', '')
            corrected_audio = tts_results[i] if i < len(tts_results) else b""

            token = backend_pb.PronunciationToken(
                score=score,
                word=word,
                wrong_transcript=wrong_transcript,
                corrected_transcript=corrected_transcript,
                corrected_ipa=corrected_ipa,
                corrected_audio=corrected_audio
            )
            tokens.append(token)

        return {"overall_score": overall_score, "tokens": tokens}

    async def _evaluate_fluency(self, deepgram_response: dict) -> dict:
        """Evaluate fluency and return results as dict."""
        def _run_fluency_eval():
            pause_score_dict = evaluate_pause(deepgram_response)
            stutter_score, stuttered_phrases = evaluate_stutter(deepgram_response)
            speech_rate_dict = calculate_speech_rates(deepgram_response)
            fluency_score = (stutter_score + pause_score_dict['score']) / 2.0

            return {
                "fluency_score": fluency_score,
                "words_per_minute": speech_rate_dict['wpm'],
                "syllables_per_minute": speech_rate_dict['spm'],
            }

        return await self._to_thread(_run_fluency_eval)

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
        session_id = request.session_id or ""
        message_id = request.message_id or ""

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

            # ---- FOLLOW-UP TURN (has audio): transcribe -> bot -> TTS -> evaluate ----
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

            # ---- TRIGGER BACKGROUND EVALUATION ----
            if session_id and message_id and transcript:
                # Fire and forget - run evaluation in background
                asyncio.create_task(
                    self._evaluate_and_notify_backend(
                        session_id, message_id, audio_url, transcript, accent, gender, dg_response
                    )
                )

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
