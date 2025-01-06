import numpy as np
import cv2

# Assume embeddings_np is the array of embeddings from YAMNet
# embeddings_np shape: [Time, 1024]
embeddings_np = np.load("starwars.npy")

# Define video parameters
video_name = "yamnet_embeddings.mp4"
frame_rate = 2.0666666666  # Adjust the frame rate as needed

# Reshape the first embedding to determine frame dimensions
frame_height, frame_width = 32, 32  # Since each embedding is reshaped to 32x32

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    video_name, fourcc, frame_rate, (frame_width, frame_height)
)

# Process embeddings and write frames to the video
for embedding in embeddings_np:
    # Reshape the 1024-dimensional vector to 32x32
    embedding_reshaped = embedding.reshape(32, 32)

    # Normalize the data for visualization
    embedding_normalized = (
        embedding_reshaped / np.ptp(embedding_reshaped) * 255
    ).astype(np.uint8)

    # Create a color map image
    colormap = cv2.applyColorMap(embedding_normalized, cv2.COLORMAP_BONE)

    # Write the frame directly to the video
    video_writer.write(colormap)

# Release the video writer
video_writer.release()

print(f"Video saved as {video_name}")
