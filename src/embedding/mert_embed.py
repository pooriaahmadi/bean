# from transformers import Wav2Vec2Processor
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
from matplotlib import cm as cm
import librosa
from scipy.stats import zscore
from sklearn import cluster as sk
from sklearn.decomposition import PCA
import sklearn

import argparse

parser = argparse.ArgumentParser(
    prog="Dolev's embedding and clustering script",
    description="This file will embed and cluster a given .wav file",
)


parser.add_argument(
    "--musicpath", type=str, help="The path to the music file", required=True
)
parser.add_argument(
    "--outputpath", type=str, help="The output file path", default="embeddings.npy"
)
parser.add_argument(
    "--timeframe",
    type=float,
    help="The intervals at which the embedding happens",
    default=0.3,
)
parser.add_argument(
    "--clusters",
    type=int,
    help="The number of clusters",
    default=3,
)

args = parser.parse_args()
"""
Alternate pipeline 
Implements embedding model MERT-v1-330M
and feature clustering methods PCA Dimension Reduction and Agglomerate Clustering

Outputs set of points to be passed on to visuals.
Not currently implemented in code, but can be connected by altering the files in /visuals to use embedsong() to generate list of points
embedsong() takes in argument "name" which should be route to .wav file, and returns list of cluster values. 
"""


def point_maker(embeddings, labels, cluster=0):
    temp = []
    points = []
    for step in np.array(embeddings):
        for count, x in enumerate(step):
            try:
                if labels[count] == cluster:
                    temp.append(x)
            except IndexError:
                pass
        points.append(np.average(temp))
    return points


def PCAPointmaker(embeddings):
    x = []
    y = []
    z = []
    c = []
    t = []
    f = []
    pca = PCA(4, svd_solver="randomized", n_oversamples=20)
    pointar = pca.fit_transform(embeddings)
    x = [step[0] for step in pointar]
    y = [step[1] for step in pointar]
    z = [step[2] for step in pointar]
    c = [step[3] for step in pointar]
    # t = [step[4] for step in pointar]
    # f = [step[5] for step in pointar]
    np.save(
        args.outputpath,
        np.column_stack((np.array(x), np.array(y), np.array(z), np.array(c))),
    )
    return x, y, z, c


def embedsong(name):  # replace with your song route in .wav format
    cluster = "kmeans"  # sets type of clustering to be used. Can be "AGG" - Agglomerate, "Kmeans" - KMeans, or "PCA" - PCA Dimension reduction

    # load wav
    input_data, sample_rate = librosa.load(name)
    np.array(input_data[1], dtype=float)

    # load model
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    )

    # load data and resample
    dataset = input_data
    sampling_rate = sample_rate
    resample_rate = processor.sampling_rate
    if resample_rate != sampling_rate:
        print(f"setting rate from {sampling_rate} to {resample_rate}")
        resampler = T.Resample(sampling_rate, resample_rate)
    else:
        resampler = None
    if resampler is None:
        input_audio = input_data
    else:
        tens = torch.from_numpy(input_data)
        doubled = tens.float()
        input_audio = resampler(doubled)

    # make segments
    segment_length = int(resample_rate * args.timeframe)
    num_segments = len(input_audio) // segment_length
    audio_segments = [
        input_audio[i * segment_length : (i + 1) * segment_length]
        for i in range(num_segments)
    ]

    # make embedding array
    embeddings = []
    for segment in audio_segments:
        inputs = processor(segment, sampling_rate=resample_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
        aggregator = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1)
        weighted_avg_hidden_states = aggregator(
            time_reduced_hidden_states.unsqueeze(0)
        ).squeeze()

        embeddings.append(weighted_avg_hidden_states.detach().numpy())

    sequences = torch.transpose(torch.from_numpy(np.array(embeddings)), 0, 1)
    sequences_np = sequences.numpy()

    # Z-Score Method for outlier removal (only for plotting)
    z_scores = np.abs(zscore(sequences_np))
    threshold = 3
    z_outliers = np.where(z_scores > threshold)[0]
    sequences_no_outliers_z = np.delete(sequences_np, z_outliers, axis=0)

    # Visualize without outliers (PCA for 2D visualization)
    pca = PCA(n_components=2)
    sequences_2d = pca.fit_transform(sequences_no_outliers_z)

    # Apply K-means and Agglomerative Clustering
    kmeans = sk.KMeans(n_clusters=args.clusters)
    kmeans_labels = kmeans.fit_predict(sequences_no_outliers_z)
    agg_clustering = sk.AgglomerativeClustering(n_clusters=args.clusters)
    agg_labels = agg_clustering.fit_predict(sequences_no_outliers_z)

    """fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # K-means clustering visualization
    axes[0].scatter(sequences_2d[:, 0], sequences_2d[:, 1], c=kmeans_labels, cmap='viridis')
    axes[0].set_title('K-means Clustering')
    axes[0].set_xlabel('PCA Component 1')
    axes[0].set_ylabel('PCA Component 2')

    # Agglomerative clustering visualization
    axes[1].scatter(sequences_2d[:, 0], sequences_2d[:, 1], c=agg_labels, cmap='viridis')
    axes[1].set_title('Agglomerative Clustering')
    axes[1].set_xlabel('PCA Component 1')
    axes[1].set_ylabel('PCA Component 2')

    plt.tight_layout()
    plt.show()"""

    # if cluster == "AGG":
    #     values = [np.array(point_maker(embeddings, agg_labels, x)) for x in range(0, 4)]
    #     x = values[0]
    #     y = values[1]
    #     z = values[2]
    #     c = values[3]
    # elif cluster == "kmeans":
    # values = [np.array(point_maker(embeddings, kmeans_labels, x)) for x in range(0, 4)]
    #     x = values[0]
    #     y = values[1]
    #     z = values[2]
    #     c = values[3]
    # else:
    #     x, y, z, c = PCAPointmaker(embeddings)
    values = [np.array(point_maker(embeddings, kmeans_labels, x)) for x in range(0, 4)]
    x = values[0]
    y = values[1]
    z = values[2]
    c = values[3]

    np.save(
        args.outputpath,
        np.column_stack((np.array(x), np.array(y), np.array(z), np.array(c))),
    )


if __name__ == "__main__":
    print(embedsong(args.musicpath))
