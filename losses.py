from typing import Union

import keras
from keras import layers, ops

from utils.keras_utils import grayscale_to_rgb


def get_loss(loss_name, *args, **kwargs):
    """Get a loss function by name."""
    if loss_name == "mse":
        return keras.losses.MeanSquaredError(*args, **kwargs)
    elif loss_name == "mae":
        return keras.losses.MeanAbsoluteError(*args, **kwargs)
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")
