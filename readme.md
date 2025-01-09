
# BEAN: Bridging Embeddings to Audio-visual Narratives 

## Brief description
In this project we aimed to create various representations of a sound track in the form of a video. Now if you remember windows XP used to have a visualizer that was based on pitch, tempo, bass, and meta-data from the file itself. We wanted to make something similar that was *actually* based on the music itself like vibe, instruments used, speed, how dancable it is, and other factors that are beyond just hard-coding some predefined effects.
## Demo

Sparks by Coldplay

[![Sparks by cold play](https://markdown-videos-api.jorgenkh.no/url?url=https%3A%2F%2Fyoutu.be%2FLzERy-Q9DaA)](https://youtu.be/LzERy-Q9DaA)

Somethings never change by Bathe Alone
[![Somethings never change](https://markdown-videos-api.jorgenkh.no/url?url=https%3A%2F%2Fyoutube.com%2Fshorts%2FkirL0AiGYDM%3Fsi%3DKhOLrt23l-EOuzsC)](https://youtube.com/shorts/kirL0AiGYDM?si=KhOLrt23l-EOuzsC)
## Underlying system
### Embedding
In this project, we're using Google's YAMnet to extract features from the soundtrack. YAMnet itself is trained on millions of soundtracks and it has learned 1024 **unique** features that it can extract given a 0.96 second time frame.
### Clustering
For our purposes, 1024 unique values is a *little too much* to process and visualize effectively, so we're currently using **kmean** clustering to reduce 1024 values to only 4 values which represent x position, y position, color pallete, size in the final output.
### Visualization
In this project we're using [Arcade](https://github.com/pythonarcade/arcade) to visualize the clustered embeddings. We essentially made a game that takes in the points and creates more points in between to make it look smoother and more connected. We then visualize in real-time and capture the rendered frame in memory, once all frames are captured the script creates a video and attaches the music to it.
## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/pooriaahmadi/music-visualizer.git
cd music-visualizer
```

### 2. Install depedencies
#### Windows
```bash
pip install -r requirements.txt
```
#### Linux
```bash
pip3 install -r requirements.txt
```

### 3. Download your soundtrack
We recommend finding your desired soundtrack on soundcloud and then passing it through [this soundcloud downloader](https://clouddownloader.net). Afterwards turn the downloaded mp3 to .wav using [this website](https:?/cloudconvert.com/mp3-to-wav).

Make sure to put the .wav file in the root of the project (in the same folder as main.py for ease-of-use)

### 4. Use the program!

#### Windows:
```bash
python main.py --music sparks.wav --visuals snake --embed_type yamnet
```

#### Linux:
```bash
python3 main.py --music sparks.wav --visuals snake --embed_type yamnet
```

This will take a while and you have to let the program visualize the whole thing first, DO NOT CLOSE the visualizer window. It will close itself and it will take a while to generate the final output video. BE PATIENT.

### 5. Play the video
If completed, you should be left with final_video.mp4 and you can use that.
## Authors

- [@pooriaahmadi](https://www.github.com/pooriaahmadi) Pooria Ahmadi
- [@dolev497](https://github.com/Dolev497) Dolev Klein
