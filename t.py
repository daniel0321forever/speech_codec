import torch
from torch import nn, Tensor

import librosa
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load("80s Sine Synth.wav")
pitches, mags = librosa.piptrack(y)
print(pitches.shape)

for i in range(pitches.shape[1]):
    index = mags[:,i].argmax()
    pitch = pitches[index, i]

    print(index)
    print(pitch)
    
    note_name = librosa.hz_to_note(pitch)
    print(note_name)

plt.plot(np.tile(np.arange(pitches.shape[1]), [100, 1]).T, pitches[:100, :].T, '.')
plt.savefig("pitch.png")