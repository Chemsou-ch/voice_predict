import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping


labels = ['yes', 'no', 'stop', 'go']

X = []
y = []

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)

    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000 - len(audio)))
    else:
        audio = audio[:16000]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=64,
        n_fft=1024,
        hop_length=256
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)

    return log_mel

for label_index, label in enumerate(labels):
    folder = f"dataset/{label}"

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            feat = extract_features(path)
            X.append(feat)
            y.append(label_index)

X = np.array(X, dtype=np.float32)
y = np.array(y)

X = X[..., np.newaxis]

print("Dataset shape:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=X.shape[1:]),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Accuracy: {acc*100:.2f}%")

model.save("model.keras")
print('model trained')