import librosa
import numpy as np
import matplotlib.pyplot as plt

audio_files = ["pitch_test.wav", "../raw_data_sample/train/wav48_silence_trimmed/p225/p225_002_mic1.flac", "../raw_data_sample/train/wav48_silence_trimmed/p225/p225_010_mic1.flac", "../raw_data_sample/train/wav48_silence_trimmed/p225/p225_045_mic2.flac"]

x = []
y = []
for audio in audio_files:
    sig, sr = librosa.load(audio)
    pitch, mag = librosa.piptrack(y=sig)

    print(pitch.shape)
    pitch = pitch.T
    print(pitch.shape)
    
    for frame in pitch:
        for idx in range(len(frame)):
            if frame[idx] > 0:
                x.append(idx)
                y.append(frame[idx])

plt.scatter(x=x, y=y)
plt.savefig("quantize_test.png")