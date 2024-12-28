import tensorflow_hub as hub
import librosa
import numpy as np


model = hub.load("https://tfhub.dev/google/yamnet/1")
waveform, sample_rate = librosa.load("docwhiler.wav", sr=16000)
scores, embeddings, spectrogram = model(waveform)
embeddings_np = embeddings.numpy()
np.save("embeddings.npy", embeddings_np)


# Print the shape of the embeddings
print("Embeddings shape:", embeddings_np.shape)  # [Time, 1024]
