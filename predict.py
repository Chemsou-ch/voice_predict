import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from scipy.io.wavfile import write

labels = ['yes', 'no', 'stop', 'go']
model = tf.keras.models.load_model("model.keras")

fs = 16000
duration = 1 

print("Get ready...")
print("Speak now!")

audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()

write("temp.wav", fs, audio)

signal, sr = librosa.load("temp.wav", sr=16000)

# trim silence
signal, _ = librosa.effects.trim(signal, top_db=20)

# pad/trim
if len(signal) < 16000:
    signal = np.pad(signal, (0, 16000 - len(signal)))
else:
    signal = signal[:16000]

mel = librosa.feature.melspectrogram(
    y=signal,
    sr=sr,
    n_mels=64,
    n_fft=1024,
    hop_length=256
)

log_mel = librosa.power_to_db(mel, ref=np.max)
log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)

X = log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)

pred = model.predict(X, verbose=0)
idx = np.argmax(pred)
conf = pred[0][idx] * 100

if conf < 70:
    print("Uncertain prediction")
else:
    print(f"Detected: {labels[idx]} ({conf:.2f}%)")