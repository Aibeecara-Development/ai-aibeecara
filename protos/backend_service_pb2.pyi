from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EvaluateGrammarResponse(_message.Message):
    __slots__ = ("session_id", "message_id", "score", "corrected_transcript", "explanation", "tense_used", "corrected_audio")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    CORRECTED_TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    EXPLANATION_FIELD_NUMBER: _ClassVar[int]
    TENSE_USED_FIELD_NUMBER: _ClassVar[int]
    CORRECTED_AUDIO_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    message_id: str
    score: float
    corrected_transcript: str
    explanation: str
    tense_used: str
    corrected_audio: bytes
    def __init__(self, session_id: _Optional[str] = ..., message_id: _Optional[str] = ..., score: _Optional[float] = ..., corrected_transcript: _Optional[str] = ..., explanation: _Optional[str] = ..., tense_used: _Optional[str] = ..., corrected_audio: _Optional[bytes] = ...) -> None: ...

class VocabularyEntry(_message.Message):
    __slots__ = ("word", "cefr", "pronunciation", "definition", "example_sentence_transcript", "example_sentence_audio")
    WORD_FIELD_NUMBER: _ClassVar[int]
    CEFR_FIELD_NUMBER: _ClassVar[int]
    PRONUNCIATION_FIELD_NUMBER: _ClassVar[int]
    DEFINITION_FIELD_NUMBER: _ClassVar[int]
    EXAMPLE_SENTENCE_TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    EXAMPLE_SENTENCE_AUDIO_FIELD_NUMBER: _ClassVar[int]
    word: str
    cefr: str
    pronunciation: str
    definition: str
    example_sentence_transcript: str
    example_sentence_audio: bytes
    def __init__(self, word: _Optional[str] = ..., cefr: _Optional[str] = ..., pronunciation: _Optional[str] = ..., definition: _Optional[str] = ..., example_sentence_transcript: _Optional[str] = ..., example_sentence_audio: _Optional[bytes] = ...) -> None: ...

class VocabularyToken(_message.Message):
    __slots__ = ("original", "synonyms")
    ORIGINAL_FIELD_NUMBER: _ClassVar[int]
    SYNONYMS_FIELD_NUMBER: _ClassVar[int]
    original: VocabularyEntry
    synonyms: _containers.RepeatedCompositeFieldContainer[VocabularyEntry]
    def __init__(self, original: _Optional[_Union[VocabularyEntry, _Mapping]] = ..., synonyms: _Optional[_Iterable[_Union[VocabularyEntry, _Mapping]]] = ...) -> None: ...

class EvaluateVocabularyResponse(_message.Message):
    __slots__ = ("session_id", "message_id", "statistics", "tokens")
    class StatisticsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    STATISTICS_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    message_id: str
    statistics: _containers.ScalarMap[str, int]
    tokens: _containers.RepeatedCompositeFieldContainer[VocabularyToken]
    def __init__(self, session_id: _Optional[str] = ..., message_id: _Optional[str] = ..., statistics: _Optional[_Mapping[str, int]] = ..., tokens: _Optional[_Iterable[_Union[VocabularyToken, _Mapping]]] = ...) -> None: ...

class PronunciationToken(_message.Message):
    __slots__ = ("score", "word", "wrong_transcript", "corrected_transcript", "corrected_ipa", "corrected_audio")
    SCORE_FIELD_NUMBER: _ClassVar[int]
    WORD_FIELD_NUMBER: _ClassVar[int]
    WRONG_TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    CORRECTED_TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    CORRECTED_IPA_FIELD_NUMBER: _ClassVar[int]
    CORRECTED_AUDIO_FIELD_NUMBER: _ClassVar[int]
    score: int
    word: str
    wrong_transcript: str
    corrected_transcript: str
    corrected_ipa: str
    corrected_audio: bytes
    def __init__(self, score: _Optional[int] = ..., word: _Optional[str] = ..., wrong_transcript: _Optional[str] = ..., corrected_transcript: _Optional[str] = ..., corrected_ipa: _Optional[str] = ..., corrected_audio: _Optional[bytes] = ...) -> None: ...

class EvaluatePronunciationResponse(_message.Message):
    __slots__ = ("session_id", "message_id", "overall_score", "tokens")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    OVERALL_SCORE_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    message_id: str
    overall_score: int
    tokens: _containers.RepeatedCompositeFieldContainer[PronunciationToken]
    def __init__(self, session_id: _Optional[str] = ..., message_id: _Optional[str] = ..., overall_score: _Optional[int] = ..., tokens: _Optional[_Iterable[_Union[PronunciationToken, _Mapping]]] = ...) -> None: ...

class EvaluateFluencyResponse(_message.Message):
    __slots__ = ("session_id", "message_id", "score", "words_per_minute", "syllables_per_minute")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    WORDS_PER_MINUTE_FIELD_NUMBER: _ClassVar[int]
    SYLLABLES_PER_MINUTE_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    message_id: str
    score: int
    words_per_minute: int
    syllables_per_minute: int
    def __init__(self, session_id: _Optional[str] = ..., message_id: _Optional[str] = ..., score: _Optional[int] = ..., words_per_minute: _Optional[int] = ..., syllables_per_minute: _Optional[int] = ...) -> None: ...
