import argparse
import subprocess
import sys
import os
import shutil

venv_python = sys.executable


def check_extension(filename, ext):
    if not filename.lower().endswith(ext):
        raise argparse.ArgumentTypeError(f"Filename must have a {ext} extension")

    return filename


parser = argparse.ArgumentParser(
    prog="Music visualizer", description="Turns music into pretty visualizations"
)

parser.add_argument(
    "--music",
    type=lambda x: check_extension(x, ".wav"),
    help="Music file path with .wav extension",
    required=True,
)

parser.add_argument(
    "--embed_model",
    type=str,
    help="Embedding model to use",
    choices=["yamnet", "mert"],
    required=True,
)

parser.add_argument(
    "--visuals",
    choices=["snake", "3d"],
    help="The type of visualization",
    required=True,
)

parser.add_argument(
    "--output_video",
    type=lambda x: check_extension(x, ".mp4"),
    default="final_video.mp4",
)


args = parser.parse_args()


def create_or_clear_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Clear the folder by deleting its contents
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        # Create the folder if it does not exist
        os.makedirs(folder_path)


def generate_embedding(filepath, outputpath, embed_model):
    match embed_model:
        case "yamnet":
            command = [
                venv_python,
                "src/embedding/yamnet_embed.py",
                "--musicpath",
                filepath,
                "--outputpath",
                outputpath,
            ]
        case "mert":
            command = []
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode == 0:
        return True
    else:
        print(process.stderr)
        raise RuntimeError("Something went wrong when trying to embed file")


def generate_clusters(filepath, outputpath, visualization, embed_type):
    if embed_type == "mert":
        return
    match visualization:
        case "snake":
            clusters = "4"
        case "3d":
            clusters = "3"

    command = [
        venv_python,
        "src/clustering/kmean.py",
        "--clusters",
        clusters,
        "--filepath",
        filepath,
        "--outputpath",
        outputpath,
    ]

    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode == 0:
        return True
    else:
        print(process.stderr)
        raise RuntimeError("Something went wrong when trying to cluster file")


def generate_visuals(filepath, visuals, music, outputpath):
    match visuals:
        case "3d":
            command = [
                venv_python,
                "src/visuals/to_visual_xyz.py",
                "--embedding",
                filepath,
                "--embedding_interval",
                "0.48",
                "--outputpath",
                outputpath,
            ]
        case "snake":
            command = [
                venv_python,
                "src/visuals/snake_visual.py",
                "--embedding",
                filepath,
                "--embedding_interval",
                "0.48",
                "--outputpath",
                outputpath,
                "--musicpath",
                music,
            ]
    process = subprocess.run(command, capture_output=True, text=True)


def attach_audio(input_video, input_audio, output_video):
    command = [
        venv_python,
        "src/visuals/attach_audio.py",
        "--input_video",
        input_video,
        "--input_audio",
        input_audio,
        "--output_video",
        output_video,
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode == 0:
        return True
    else:
        print(process.stderr)
        raise RuntimeError("Something went wrong when trying to attach audio file")


create_or_clear_folder("tmp")
generate_embedding(args.music, "tmp/embeddings.npy")
generate_clusters("tmp/embeddings.npy", "tmp/reduced_embeddings.npy", args.visuals)
generate_visuals(
    "tmp/reduced_embeddings.npy", args.visuals, args.music, "tmp/visual_video.mp4"
)
attach_audio("tmp/visual_video.mp4", args.music, args.output_video)
