import librosa
import numpy as np

audio_path = "J Dilla - Stop - Donuts Full Album [2026-03-29 173144].wav"
y, sr = librosa.load(audio_path, duration=12)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
t = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)

chroma = librosa.feature.chroma_stft(y=y, sr=sr)
key_idx = np.argmax(np.sum(chroma, axis=1))
key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
print(f"Tempo: {t:.0f} BPM")
print(f"Key: {key_names[key_idx]}")
