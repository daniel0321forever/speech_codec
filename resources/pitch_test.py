import librosa
import numpy as np

audio_file = "pitch_test.wav"
y, sr = librosa.load(audio_file)
pitch, mag = librosa.piptrack()