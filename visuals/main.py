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


WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
WINDOW_TITLE = "Music visualizer"
FADING_COUNT = 5
FADE_RATE = 10
FPS_MULTIPLIER = 1

VIDEO_FPS = 60  # Frames per second for the video
VIDEO_FILE = "visuals/output_video.mp4"  # Output video file name

COLOUR_PALETTES = []


def normalize_array_to_0_1(array):
    min_val = array.min()
    max_val = array.max()
    return (array - min_val) / (max_val - min_val)


def remove_outliers(data):
    # Calculate the mean and standard deviation
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Compute z-scores
    z_scores = (data - mean) / std

    # Define threshold for outliers (e.g., z-score > 3 or < -3)
    threshold = 2
    mask = np.abs(z_scores) < threshold

    # Keep only non-outliers
    filtered_data = data[np.all(mask, axis=1)]
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
        self.background_color = arcade.color.WHITE

        # If you have sprite lists, you should create them here,
        # and set them to None
        self.music: np.ndarray = np.load("visuals/reduced_embeddings_6clusters.npy")

        self.points = remove_outliers(self.music[:, :2])
        self.points = normalize_array_to_0_1(self.points)

        self.points *= (WINDOW_WIDTH, WINDOW_HEIGHT)
        self.index = 0
        self.snake_sprites = arcade.SpriteList()

        self.video_writer = cv2.VideoWriter(
            VIDEO_FILE,
            cv2.VideoWriter_fourcc(*"mp4v"),
            VIDEO_FPS,
            (WINDOW_WIDTH, WINDOW_HEIGHT),
        )

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
        arcade.draw_lbwh_rectangle_filled(
            0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, (255, 255, 255, FADE_RATE)
        )

        # Call draw() on all your sprite lists below
        self.snake_sprites.draw()
        arcade.draw_line_strip(
            map(lambda x: x.position, self.snake_sprites), arcade.color.BLACK, 3
        )

        frame = arcade.get_image()
        frame = np.array(frame)
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        self.video_writer.write(frame)

        # arcade.draw_lbwh_rectangle_filled(10, 0, 100, 50, arcade.color.WHITE)
        # arcade.draw_text(f"FPS: {arcade.get_fps():.1f}", 10, 10, arcade.color.BLACK, 12)
        # arcade.draw_text(f"i: {self.index}", 10, 0, arcade.color.BLACK, 12)

    def on_fixed_update(self, delta_time: float):
        if self.index >= self.points.shape[0]:
            self.reset()
            self.video_writer.release()

            arcade.close_window()
        new_point = self.points[self.index]
        self.index += 1
        sprite = arcade.SpriteCircle(
            5,
            arcade.color.BLACK,
            False,
        )
        sprite.center_x = new_point[0]
        sprite.center_y = new_point[1]

        self.snake_sprites.append(sprite)
        if len(self.snake_sprites) > FADING_COUNT:
            self.snake_sprites.pop(0)

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
        fixed_rate=(1 / 2.066666666) / FPS_MULTIPLIER,
        update_rate=(1 / 60) / FPS_MULTIPLIER,
        draw_rate=(1 / 60) / FPS_MULTIPLIER,
    )

    # Create and setup the GameView
    game = GameView()

    # Show GameView on screen
    window.show_view(game)

    # Start the arcade game loop
    arcade.run()


if __name__ == "__main__":
    main()
