import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = "J Dilla - Stop - Donuts Full Album [2026-03-29 173144].wav"
y, sr = librosa.load(audio_path, duration=12)

# Check RMS Energy
rms = librosa.feature.rms(y=y)[0]
print(f"RMS Energy max: {np.max(rms)}")

# Check Spectral Rolloff
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
print(f"Rolloff len: {len(rolloff)}")

# Check Chromagram
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
print(f"Chromagram shape: {chroma.shape}")
