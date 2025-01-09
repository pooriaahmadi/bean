import argparse

parser = argparse.ArgumentParser(
    prog="Embedder",
    description="This program will convert any given .wav file into embeddings",
)

parser.add_argument(
    "--musicpath", type=str, help="The path to the music file", required=True
)
parser.add_argument(
    "--outputpath", type=str, help="The output file path", default="embeddings.npy"
)
args = parser.parse_args()

if not ".wav" in args.musicpath:
    raise TypeError("musicpath should end in .wav")

import tensorflow_hub as hub
import librosa
import numpy as np

model = hub.load("https://tfhub.dev/google/yamnet/1")
waveform, sample_rate = librosa.load(args.musicpath, sr=16000)
scores, embeddings, spectrogram = model(waveform)
embeddings_np = embeddings.numpy()
np.save(
    args.outputpath if ".npy" in args.outputpath else args.outputpath + ".npy",
    embeddings_np,
)


# Print the shape of the embeddings
print("Embeddings shape:", embeddings_np.shape)  # [Time, 1024]
print("File saved in " + args.outputpath)
