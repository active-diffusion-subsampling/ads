import keras
from keras import layers

from models.layers import DownBlock, ResidualBlock, UpBlock, sinusoidal_embedding


def get_network(
    image_shape,
    widths,
    block_depth,
    embedding_min_frequency=1.0,
    embedding_max_frequency=1000.0,
    embedding_dims=32,
):
    assert (
        len(image_shape) == 3
    ), "image_shape must be a tuple of (height, width, channels)"

    image_height, image_width, n_channels = image_shape
    noisy_images = keras.Input(shape=(image_height, image_width, n_channels))
    noise_variances = keras.Input(shape=(1, 1, 1))

    @keras.saving.register_keras_serializable()
    def _sinusoidal_embedding(x):
        return sinusoidal_embedding(
            x, embedding_min_frequency, embedding_max_frequency, embedding_dims
        )

    e = layers.Lambda(_sinusoidal_embedding, output_shape=(1, 1, 32))(noise_variances)
    e = layers.UpSampling2D(size=(image_height, image_width), interpolation="nearest")(
        e
    )

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(n_channels, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")
