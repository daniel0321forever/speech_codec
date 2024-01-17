import torchaudio.datasets
import torch

ds = torchaudio.datasets.VCTK_092(
    root=".",
    download=True,
    audio_ext=".wav",
)