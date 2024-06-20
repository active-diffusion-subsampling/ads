import datetime
import numpy as np
from PIL import Image
from utils.lib import log

from utils.keras_utils import grayscale_to_rgb


def get_date_string(string: str = None):
    """Generate a date string for current time, according to format specified by
        `string`. Refer to the documentation of the datetime module for more information
        on the formatting options.
    mq
        If no string is specified, the default format is used: "%Y_%m_%d_%H%M%S".
    """
    if string is not None and not isinstance(string, str):
        raise TypeError("Input must be a string.")

    # Get the current time
    now = datetime.datetime.now()

    # If no string is specified, use the default format
    if string is None:
        string = "%Y_%m_%d_%H%M%S"

    # Generate the date string
    date_str = now.strftime(string)

    return date_str


def _preprocess_for_saving(images):
    """Preprocesses images for saving to GIF or MP4.

    Args:
        images (ndarray, list[ndarray]): Images. Must have shape (n_frames, height, width, channels)
            or (n_frames, height, width).
    """
    images = np.array(images)

    # Remove channel axis if it is 1 (grayscale image)
    if images.ndim == 4 and images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)

    # convert grayscale images to RGB
    if images.ndim == 3:
        images = [grayscale_to_rgb(image) for image in images]

    return images


def save_to_gif(images, filename, fps=20):
    """Saves a sequence of images to .gif file.
    Args:
        images: list of images (numpy arrays). Must have shape
            (n_frames, height, width, channels) or (n_frames, height, width).
            If channel axis is not present, or is 1, grayscale image is assumed,
            which is then converted to RGB. Images should be uint8.
        filename: string containing filename to which data should be written.
        fps: frames per second of rendered format.
    """
    images = _preprocess_for_saving(images)

    if fps > 50:
        log.warning(f"Cannot set fps ({fps}) > 50. Setting it automatically to 50.")
        fps = 50

    duration = 1 / (fps) * 1000  # milliseconds per frame

    pillow_img, *pillow_imgs = [Image.fromarray(img) for img in images]

    pillow_img.save(
        fp=filename,
        format="GIF",
        append_images=pillow_imgs,
        save_all=True,
        loop=0,
        duration=duration,
        interlace=False,
        optimize=False,
    )
    return log.success(f"Succesfully saved GIF to -> {log.yellow(filename)}")
