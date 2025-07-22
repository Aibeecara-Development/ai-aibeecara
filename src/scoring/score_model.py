import requests
import json
import librosa
from librosa.feature import rms, mfcc
from librosa.sequence import dtw
import itertools
import numpy as np

def score_model(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    # Calculate short-term energy
    frame_length = 2048
    hop_length = 512
    energy = rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Normalize energy
    energy_db = librosa.amplitude_to_db(energy, ref=np.max)

    silence_threshold_db = -40
    silent_frames = energy_db < silence_threshold_db

    # Convert to time
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)

    silent_regions = []
    for k, g in itertools.groupby(enumerate(silent_frames), lambda x: x[1]):
        if k:
            group = list(g)
            start_time = times[group[0][0]]
            end_time = times[group[-1][0]]
            duration = end_time - start_time
            if duration > 1:  # only keep pauses > 1s
                silent_regions.append((start_time, end_time))

    # mfcc_result = mfcc(y=y, sr=sr, n_mfcc=13)
    # D, wp = dtw(mfcc_result[:, start1:end1], mfcc_result[:, start2:end2])
    # if D[-1, -1] < threshold:
    #     print("Likely repetition detected")





