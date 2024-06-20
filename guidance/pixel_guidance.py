import jax
import tensorflow as tf
from keras import ops
from keras.src import backend

from autograd_backend import AutoGrad
from guidance.utils import L2
from utils.keras_utils import check_keras_backend

check_keras_backend()

_GUIDANCE = {}


def register_guidance(cls=None, *, name=None):
    """A decorator for registering guidance classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _GUIDANCE:
            raise ValueError(f"Already registered guidance with name: {local_name}")
        _GUIDANCE[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_guidance(name):
    """Get guidance class for given name."""
    assert (
        name in _GUIDANCE
    ), f"Guidance {name} not found. Available guidance: {_GUIDANCE.keys()}"
    return _GUIDANCE[name]


@register_guidance(name="dps")
def get_dps(dm):
    """
    Returns a function with the following signature:
    (noisy_images, noise_rates, signal_rates) -> (measurement_error, (pred_noises, pred_images)), gradients
    """

    autograd = AutoGrad()

    def compute_measurement_error(
        noisy_images, measurement, operator, omega, noise_rates, signal_rates
    ):
        pred_noises, pred_images = dm.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        measurement_error = omega * L2(measurement - operator.forward(pred_images))
        return measurement_error, (pred_noises, pred_images)

    autograd.set_function(compute_measurement_error)
    return autograd.get_gradient_and_value_jit_fn(has_aux=True)
