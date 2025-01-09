import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

parser = argparse.ArgumentParser(
    prog="Visual 3D",
    description="This will turn an embedding into a 3D graph with respect to time",
)


def check_extension(filename, ext):
    if not filename.lower().endswith(ext):
        raise argparse.ArgumentTypeError(f"Filename must have a {ext} extension")

    return filename


parser.add_argument(
    "--embedding",
    type=lambda x: check_extension(x, ".npy"),
    help="The file path for the clustered embeddings",
)
parser.add_argument(
    "--embedding_interval",
    type=float,
    help="The time interval between each embedding sample, in seconds.",
)
parser.add_argument(
    "--outputpath",
    type=lambda x: check_extension(x, ".mp4"),
    default="output_video.mp4",
    help="the desired location of the output video",
)

args = parser.parse_args()

EMBEDDING_PATH = args.embedding
OUTPUT_PATH = args.outputpath
EMBEDDING_INTERVAL = args.embedding_interval

# Load the .npy file
data = np.load(EMBEDDING_PATH)

# Ensure the data is in the expected format
assert data.ndim == 2 and data.shape[1] == 3, "Expected data shape: (num_frames, 3)"

# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Set up the video writer
frame_width, frame_height = 640, 480
out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    1 / EMBEDDING_INTERVAL,
    (frame_width, frame_height),
)


# Function to remove outliers using IQR
def remove_outliers(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return data[np.all(np.abs(data - mean) < 2 * std, axis=1)]


# Remove outliers
clean_data = remove_outliers(data)
angle = 0
# Iterate through frames
for i in range(data.shape[0]):
    ax.cla()  # Clear the axis for the new frame

    # Extract XYZ coordinates up to the current frame
    x, y, z = data[: i + 1, 0], data[: i + 1, 1], data[: i + 1, 2]

    # Plot the points
    ax.scatter(x, y, z, c="b")

    # Draw lines between points
    ax.plot(x, y, z, c="r")

    # Set axis limits
    ax.set_xlim([np.min(clean_data[:, 0]), np.max(clean_data[:, 0])])
    ax.set_ylim([np.min(clean_data[:, 1]), np.max(clean_data[:, 1])])
    ax.set_zlim([np.min(clean_data[:, 2]), np.max(clean_data[:, 2])])

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # Rotate the view
    ax.view_init(elev=30, azim=angle)
    angle += 2  # Increment the angle to rotate the scene

    # Draw the plot
    plt.draw()
    fig.canvas.draw()

    # Convert plot to image
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    # Convert ARGB to RGB by removing the alpha channel
    img = img[:, :, 1:]  # Keep only the R, G, B channels

    # Resize the image to match the video frame size
    img = cv2.resize(img, (frame_width, frame_height))

    # Write the frame to the video
    out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Display the frame in a window using OpenCV
    # cv2.imshow("3D Plot", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # if cv2.waitKey(500) & 0xFF == ord("q"):  # Display each frame for 500ms
    #     break

# Release the video writer and close all OpenCV windows
out.release()
cv2.destroyAllWindows()

print("Video saved as " + OUTPUT_PATH)
