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


@keras.saving.register_keras_serializable()
class KID(keras.metrics.Metric):
    def __init__(self, name, image_shape, kid_image_shape, **kwargs):
        super().__init__(name=name, **kwargs)
        assert (
            len(image_shape) == 3
        ), "image_shape must be a tuple of (height, width, channels)"
        assert (
            len(kid_image_shape) == 3
        ), "kid_image_shape must be a tuple of (height, width, channels)"

        assert kid_image_shape[-1] == 3, "kid_image_shape must have 3 channels"

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining

        # convert grayscale to RGB
        if image_shape[-1] == 1:
            grayscale_to_rgb_layer = layers.Lambda(grayscale_to_rgb)
        else:
            grayscale_to_rgb_layer = keras.layers.Lambda(lambda x: x)

        self.encoder = keras.Sequential(
            [
                keras.Input(shape=image_shape),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_shape[0], width=kid_image_shape[1]),
                grayscale_to_rgb_layer,
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=kid_image_shape,
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = ops.cast(ops.shape(features_1)[1], dtype="float32")
        return (
            features_1 @ ops.transpose(features_2) / feature_dimensions + 1.0
        ) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = ops.shape(real_features)[0]
        # warnings.warn(f"batch size is fixed to {batch_size} in KID calculation")
        batch_size_f = ops.cast(batch_size, dtype="float32")
        mean_kernel_real = ops.sum(kernel_real * (1.0 - ops.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = ops.sum(
            kernel_generated * (1.0 - ops.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = ops.mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()
