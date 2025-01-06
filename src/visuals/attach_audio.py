import moviepy.editor as mp
import argparse

parser = argparse.ArgumentParser(
    description="This script will attach an audio to a video"
)
parser.add_argument("--input_video", type=str)
parser.add_argument("--input_audio", type=str)
parser.add_argument("--output_video", type=str)

args = parser.parse_args()

# Load video file
video = mp.VideoFileClip(args.input_video)

# Load audio file
audio = mp.AudioFileClip(args.input_audio)

# Set the audio of the video
final_video = video.set_audio(audio)

# Write the final video file
final_video.write_videofile(args.output_video, codec="libx264", audio_codec="aac")
