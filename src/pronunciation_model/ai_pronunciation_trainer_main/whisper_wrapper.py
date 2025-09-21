import torch 
from transformers import pipeline
from src.pronunciation_model.ai_pronunciation_trainer_main.ModelInterfaces import IASRModel
from typing import Union
import numpy as np 

class WhisperASRModel(IASRModel):
    def __init__(self, model_name="openai/whisper-base"):
        self.asr = pipeline("automatic-speech-recognition", model=model_name, return_timestamps="segment")
        self._transcript = ""
        self._word_locations = []
        self.sample_rate = 16000

    def processAudio(self, audio:Union[np.ndarray, torch.Tensor]):
        # 'audio' can be a path to a file or a numpy array of audio samples.
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        result = self.asr(audio[0])
        self._transcript = result["text"]
        self._word_locations = []
        if "chunks" in result:
            self._word_locations = []
            for word_info in result["chunks"]:
                start_ts = word_info["timestamp"][0]
                end_ts = word_info["timestamp"][1]
                # Handle None values safely
                if start_ts is None:
                    start_ts = 0
                if end_ts is None:
                    end_ts = start_ts + 1
                self._word_locations.append({
                    "word": word_info["text"],
                    "start_ts": start_ts * self.sample_rate,
                    "end_ts": end_ts * self.sample_rate,
                    "tag": "processed"
                })

    def getTranscript(self) -> str:
        return self._transcript

    def getWordLocations(self) -> list:
        
        return self._word_locations
