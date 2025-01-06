import numpy as np
import cv2

# Assume embeddings_np is the array of embeddings from YAMNet
# embeddings_np shape: [Time, 3]
embeddings_np = np.load("reduced_embeddings_xyz.npy")

# Define video parameters
video_name = "output_xyz_pixels.mp4"
frame_rate = 2.0666666666  # Adjust the frame rate as needed

# Reshape the first embedding to determine frame dimensions
frame_height, frame_width = 3, 1  # Since each embedding is reshaped to 1x3

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    video_name, fourcc, frame_rate, (frame_width, frame_height)
)

# Process embeddings and write frames to the video
for embedding in embeddings_np:
    # Reshape the 16-dimensional vector to 1x3
    embedding_reshaped = embedding.reshape(1, 3)

    # Normalize the data for visualization
    embedding_normalized = (embedding / np.ptp(embedding) * 255).astype(np.uint8)

    # Create a color map image
    colormap = cv2.applyColorMap(embedding_normalized, cv2.COLORMAP_BONE)

    # Write the frame directly to the video
    video_writer.write(colormap)

# Release the video writer
video_writer.release()

print(f"Video saved as {video_name}")
