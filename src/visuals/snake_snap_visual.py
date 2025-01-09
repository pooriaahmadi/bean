import arcade
import numpy as np
import cv2
from scipy.interpolate import CubicSpline
import argparse

parser = argparse.ArgumentParser(
    prog="This program will turn clustered embeddings into visuals"
)
COLOUR_PALETTES = (
    ((255, 87, 51), (255, 195, 0)),
    ((52, 152, 219), (46, 204, 113)),
    ((44, 62, 80), (236, 240, 241)),
)


def normalize_array_to_0_1(array):
    min_val = array.min()
    max_val = array.max()
    return (array - min_val) / (max_val - min_val)


def remove_outliers(data, axis=1):
    # Calculate the mean and standard deviation
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Compute z-scores
    z_scores = (data - mean) / std

    # Define threshold for outliers (e.g., z-score > 3 or < -3)
    threshold = 2
    mask = np.abs(z_scores) < threshold

    # Keep only non-outliers
    if axis == 0:
        filtered_data = np.where(mask, data, mean)
    else:
        filtered_data = data.copy()
        for i in range(data.shape[1]):
            column_mean = mean[i]
            column_mask = mask[:, i]
            filtered_data[~column_mask, i] = column_mean
    return filtered_data


def check_extension(filename, ext):
    if not filename.lower().endswith(ext):
        raise argparse.ArgumentTypeError(f"Filename must have a {ext} extension")

    return filename


parser.add_argument("--window_width", type=int, default=500)
parser.add_argument("--window_height", type=int, default=500)
parser.add_argument(
    "--embedding",
    type=lambda x: check_extension(x, ".npy"),
    help="The file path for the clustered embeddings",
)
parser.add_argument(
    "--embedding_interval",
    type=float,
    help="The time interval between each embedding sample, in seconds.",
    required=True,
)
parser.add_argument(
    "--base_line_width",
    type=int,
    default=2,
    help="The smallest width that the lines can be",
)
parser.add_argument(
    "--base_line_range",
    type=int,
    default=4,
    help="The range that the line width can grow based on the clusters",
)
parser.add_argument(
    "--smooth_points",
    type=int,
    default=5,
    help="The amount of smooth points between each embedding",
)

args = parser.parse_args()


WINDOW_WIDTH = args.window_width
WINDOW_HEIGHT = args.window_height
WINDOW_TITLE = "Music visualizer"
FPS_MULTIPLIER = 1
BASE_LINE_WIDTH = args.base_line_width
BASE_LINE_RANGE = args.base_line_range

POINT_UPDATE_RATE = 1 / args.embedding_interval  # frames per second
SMOOTH_POINTS = args.smooth_points

# Open the window. Set the window title and dimensions
window = arcade.open_window(
    WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, resizable=True, fixed_rate=1
)
# Set the background color
arcade.set_background_color(arcade.color.WHITE)

# Clear screen and start render process
arcade.start_render()


music: np.ndarray = np.load(args.embedding)

colours = remove_outliers(music[:, 2], 0)
colours = normalize_array_to_0_1(colours)


sizes = remove_outliers(music[:, 3], 0)
sizes = normalize_array_to_0_1(sizes)

points_x = remove_outliers(music[:, 0], axis=0)
points_x = normalize_array_to_0_1(points_x)
points_x *= WINDOW_WIDTH - 30

points_y = remove_outliers(music[:, 1], axis=0)
points_y = normalize_array_to_0_1(points_y)
points_y *= WINDOW_HEIGHT - 30

points = np.column_stack((points_x, points_y))

# Create smooth points
t = np.linspace(0, 1, points.shape[0])
# Fit splines for x(t) and y(t)
spline_x = CubicSpline(t, points[:, 0])
spline_y = CubicSpline(t, points[:, 1])

# Generate smooth parameter values
t_smooth = np.linspace(0, 1, points.shape[0] * SMOOTH_POINTS)
points_smooth = np.column_stack((spline_x(t_smooth), spline_y(t_smooth)))
print(points.shape, colours.shape, sizes.shape, music.shape)


for i in range(points.shape[0]):
    if len(COLOUR_PALETTES) <= int(colours[i] * len(COLOUR_PALETTES)):
        colour = COLOUR_PALETTES[-1]
    else:
        colour = COLOUR_PALETTES[int(colours[i] * len(COLOUR_PALETTES))]
    size = int(sizes[i] * BASE_LINE_RANGE) + BASE_LINE_WIDTH
    arcade.draw_line_strip(
        points_smooth[SMOOTH_POINTS * i : SMOOTH_POINTS * (i + 1)],
        colour[0],
        size,
    )


arcade.finish_render()


arcade.enable_timings()


def yay(*args):
    arcade.get_image().save("big_picture.png")
    arcade.close_window()


window.on_fixed_update = yay
arcade.run()
