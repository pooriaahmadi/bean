"""
Starting Template

Once you have learned how to use classes, you can begin your program with this
template.

If Python and Arcade are installed, this example can be run from the command line with:
python -m arcade.examples.starting_template
"""
import arcade
import numpy as np
import cv2
from scipy.interpolate import CubicSpline
import argparse

parser = argparse.ArgumentParser(
    prog="This program will turn clustered embeddings into visuals"
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
parser.add_argument(
    "--musicpath",
    type=lambda x: check_extension(x, ".wav"),
    help="The file path for the audio .wav file.",
)
parser.add_argument("--window_width", type=int, default=500)
parser.add_argument("--window_height", type=int, default=500)
parser.add_argument(
    "--fading_count",
    type=int,
    default=5,
    help="The lifetime of the points before fading away",
)
parser.add_argument(
    "--fade_rate",
    type=int,
    default=10,
    help="The speed at which backgrounds and things fade",
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
FADING_COUNT = args.fading_count
FADE_RATE = args.fade_rate
FPS_MULTIPLIER = 1
BASE_LINE_WIDTH = args.base_line_width
BASE_LINE_RANGE = args.base_line_range

VIDEO_FILE = args.outputpath  # Output video file name
POINT_UPDATE_RATE = 1 / args.embedding_interval  # frames per second
VIDEO_FPS = POINT_UPDATE_RATE * 30  # Frames per second for the video
SMOOTH_POINTS = args.smooth_points

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


class GameView(arcade.View):
    """
    Main application class.

    NOTE: Go ahead and delete the methods you don't need.
    If you do need a method, delete the 'pass' and replace it
    with your own code. Don't leave 'pass' in this program.
    """

    def __init__(self):
        super().__init__()
        arcade.enable_timings()
        self.counter = 0
        self.index = 0
        self.smooth_counter = 0
        self.background_color = arcade.color.WHITE
        self.snake_sprites = arcade.SpriteList()

        # If you have sprite lists, you should create them here,
        # and set them to None
        self.music: np.ndarray = np.load(args.embedding)
        self.audio = arcade.load_sound(args.musicpath)

        self.colours = remove_outliers(self.music[:, 2], 0)
        self.colours = normalize_array_to_0_1(self.colours)

        self.sizes = remove_outliers(self.music[:, 3], 0)
        self.sizes = normalize_array_to_0_1(self.sizes)

        self.points = remove_outliers(self.music[:, :2])
        self.points = normalize_array_to_0_1(self.points)

        points_x = remove_outliers(self.music[:, 0], axis=0)
        points_x = normalize_array_to_0_1(points_x)

        points_y = remove_outliers(self.music[:, 1], axis=0)
        points_y = normalize_array_to_0_1(points_y)

        self.points = np.column_stack((points_x, points_y))
        self.points *= (WINDOW_WIDTH - 30, WINDOW_HEIGHT - 30)

        # Create smooth points

        t = np.linspace(0, 1, self.points.shape[0])
        # Fit splines for x(t) and y(t)
        spline_x = CubicSpline(t, self.points[:, 0])
        spline_y = CubicSpline(t, self.points[:, 1])

        # Generate smooth parameter values
        t_smooth = np.linspace(0, 1, self.points.shape[0] * SMOOTH_POINTS)
        self.points_smooth = np.column_stack((spline_x(t_smooth), spline_y(t_smooth)))
        print(self.points.shape, self.colours.shape, self.sizes.shape, self.music.shape)

        self.video_writer = cv2.VideoWriter(
            VIDEO_FILE,
            cv2.VideoWriter_fourcc(*"mp4v"),
            VIDEO_FPS,
            (WINDOW_WIDTH, WINDOW_HEIGHT),
        )
        self.background = (255, 255, 255)
        self.update_points()
        arcade.play_sound(self.audio)

    def reset(self):
        """Reset the game to the initial state."""
        # Do changes needed to restart the game here if you want to support that
        self.index = 0
        self.snake_sprites.clear()

    def on_draw(self):
        """
        Render the screen.
        """

        # This command should happen before we start drawing. It will clear
        # the screen to the background color, and erase what we drew last frame.
        # self.clear()
        # instead of clearing i'm gonna shade out the previously drawn frames
        if self.counter == VIDEO_FPS // POINT_UPDATE_RATE:
            self.update_points()
            self.counter = 0
            self.smooth_counter = 0

        arcade.draw_lbwh_rectangle_filled(
            0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, (*self.background, FADE_RATE)
        )

        # Call draw() on all your sprite lists below

        # self.snake_sprites.draw()

        for i, snake_sprite in enumerate(self.snake_sprites):
            if i == len(self.snake_sprites) - 1:
                fps_between_points = VIDEO_FPS / POINT_UPDATE_RATE
                fps_between_smooth_points = fps_between_points / SMOOTH_POINTS
                iterations = int(self.counter / fps_between_smooth_points)
                arcade.draw_line_strip(
                    self.points_smooth[
                        (self.index - 1)
                        * SMOOTH_POINTS : (self.index - 1)
                        * SMOOTH_POINTS
                        + iterations
                        + 2
                    ],
                    snake_sprite.color,
                    snake_sprite.size[0],
                )
            else:
                arcade.draw_line_strip(
                    self.points_smooth[
                        (self.index - len(self.snake_sprites) + i)
                        * SMOOTH_POINTS : (self.index - len(self.snake_sprites) + i + 1)
                        * SMOOTH_POINTS
                        + 1
                    ],
                    snake_sprite.color,
                    snake_sprite.size[0],
                )
                pass
        frame = arcade.get_image()
        frame = np.array(frame)
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        self.video_writer.write(frame)

        # arcade.draw_lbwh_rectangle_filled(10, 0, 100, 50, arcade.color.WHITE)
        # arcade.draw_text(f"FPS: {arcade.get_fps():.1f}", 10, 10, arcade.color.BLACK, 12)
        # arcade.draw_text(f"i: {self.index}", 10, 0, arcade.color.BLACK, 12)

        self.counter += 1

    def update_points(self):
        if self.index >= self.music.shape[0]:
            self.reset()
            self.video_writer.release()
            arcade.close_window()
            return
        new_point = self.points[self.index]
        if len(COLOUR_PALETTES) <= int(self.colours[self.index] * len(COLOUR_PALETTES)):
            colour = COLOUR_PALETTES[-1]
        else:
            colour = COLOUR_PALETTES[
                int(self.colours[self.index] * len(COLOUR_PALETTES))
            ]
        # colour = COLOUR_PALETTES[0]
        size = int(self.sizes[self.index] * BASE_LINE_RANGE) + BASE_LINE_WIDTH
        # size = BASE_LINE_WIDTH
        self.background = colour[1]
        sprite = arcade.SpriteCircle(
            size,
            colour[0],
            False,
        )
        sprite.center_x = new_point[0]
        sprite.center_y = new_point[1]

        self.snake_sprites.append(sprite)
        if len(self.snake_sprites) > FADING_COUNT:
            self.snake_sprites.pop(0)
        self.index += 1

    def on_update(self, delta_time):
        """
        All the logic to move, and the game logic goes here.
        Normally, you'll call update() on the sprite lists that
        need it.
        """
        pass

    def on_key_press(self, key, key_modifiers):
        """
        Called whenever a key on the keyboard is pressed.

        For a full list of keys, see:
        https://api.arcade.academy/en/latest/arcade.key.html
        """
        pass

    def on_key_release(self, key, key_modifiers):
        """
        Called whenever the user lets off a previously pressed key.
        """
        pass

    def on_mouse_motion(self, x, y, delta_x, delta_y):
        """
        Called whenever the mouse moves.
        """
        pass

    def on_mouse_press(self, x, y, button, key_modifiers):
        """
        Called when the user presses a mouse button.
        """
        pass

    def on_mouse_release(self, x, y, button, key_modifiers):
        """
        Called when a user releases a mouse button.
        """
        pass


def main():
    """Main function"""
    # Create a window class. This is what actually shows up on screen
    window = arcade.Window(
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        WINDOW_TITLE,
        fixed_rate=(1 / POINT_UPDATE_RATE) / FPS_MULTIPLIER,
        update_rate=(1 / VIDEO_FPS) / FPS_MULTIPLIER,
        draw_rate=(1 / VIDEO_FPS) / FPS_MULTIPLIER,
    )

    # Create and setup the GameView
    game = GameView()

    # # Show GameView on screen
    window.show_view(game)

    # # Start the arcade game loop
    arcade.run()


if __name__ == "__main__":
    main()
