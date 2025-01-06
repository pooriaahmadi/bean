import kagglehub
import os
import tensorflow_hub as hub
import librosa
import numpy as np


model = hub.load("https://tfhub.dev/google/yamnet/1")


def extract_features(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=16000)
    scores, embeddings, spectrogram = model(waveform)
    embeddings_np = embeddings.numpy()
    return embeddings_np


# Download latest version
path = kagglehub.dataset_download(
    "andradaolteanu/gtzan-dataset-music-genre-classification"
)
base_dir = path + "/Data/genres_original"
for subfolder in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder)

    if os.path.isdir(subfolder_path):  # Ensure it's a folder
        for file in os.listdir(subfolder_path):
            if file.endswith(".wav"):  # Check if it's a .wav file
                file_path = os.path.join(subfolder_path, file)
                print(file_path)
                try:
                    embedding = extract_features(file_path)
                    np.save(f"embeddings/{file}.npy", embedding)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
