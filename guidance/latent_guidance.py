"""Guidance module"""

import jax
import keras
import tensorflow as tf

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
    """Diffusion Posterior Sampling (DPS) guidance.
    see https://arxiv.org/abs/2209.14687
    """
    autograd = AutoGrad()

    def compute_measurement_error(
        noisy_images,
        measurement,
        operator,
        noise_rates,
        signal_rates,
        omega,
    ):
        pred_noises, pred_images = dm.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        pred_images_decoded = dm.decoder(pred_images)
        pred_measurement = operator.forward(pred_images_decoded)

        measurement_error = omega * L2(measurement - pred_measurement)
        return measurement_error, (pred_noises, pred_images)

    disable_jit = False
    if "StableDiffusion" in str(dm):
        disable_jit = True

    autograd.set_function(compute_measurement_error)
    return autograd.get_gradient_and_value_jit_fn(has_aux=True, disable_jit=disable_jit)


@register_guidance(name="psld")
def get_psld(dm):
    """Computes gradients to guide diffusion process using the PSLD algorithm
    see: https://arxiv.org/pdf/2307.00619.pdf

    Implemenation follows:
        https://github.com/LituRout/PSLD/blob/d734647bbc1ed0b1171521a804fc744973779f8c/stable-diffusion/ldm/models/diffusion/psld.py#L317

    Args:
        dm (StableDiffusionBase):  instance of diffusion model containing image_encoder and decoder functions to
            translate between image and latent space.
            operator (LinearOperator): operator representing the degredationg process A: x -> y,
            containing forward and transpose operations.
        measurement (Tensor): the measurements y, resulting from y = A(x).
        pred_z0 (Tensor): a prediction of the solution in the latent space, i.e. y = A(D(pred_z0))
        latent_prev (Tensor): the latent from the previous reverse diffusion step.
            This tensor must have gradient tracking enabled.

    Returns:
        next_latent (Tensor): the gradients of the guidance error function w.r.t. latent_prev
        error (int): the guidance error produced by latent
    """
    autograd = AutoGrad()

    def compute_measurement_error(
        noisy_images,
        measurement,
        operator,
        noise_rates,
        signal_rates,
        omega,
        gamma,
    ):
        pred_noises, pred_images = dm.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        pred_images_decoded = dm.decoder(pred_images)
        pred_measurement = operator.forward(pred_images_decoded)

        measurement_error = omega * L2(measurement - pred_measurement)

        # gluing term
        ortho_project = pred_images_decoded - operator.transpose(
            operator.forward(pred_images_decoded)
        )
        parallel_project = operator.transpose(measurement)
        inpainted_image = parallel_project + ortho_project
        encoded_z0 = dm.image_encoder(inpainted_image)
        inpaint_error = L2(encoded_z0 - pred_images)

        # overall error
        error = omega * measurement_error + gamma * inpaint_error
        return error, (pred_noises, pred_images)

    disable_jit = False
    if "StableDiffusion" in str(dm):
        disable_jit = True

    autograd.set_function(compute_measurement_error)
    return autograd.get_gradient_and_value_jit_fn(has_aux=True, disable_jit=disable_jit)


@register_guidance(name="no-guidance")
def no_guidance(dm):
    """No guidance, unconditional diffusion.
    Useful for testing
    """

    def compute_measurement_error(
        noisy_images,
        measurement,
        operator,
        noise_rates,
        signal_rates,
        **kwargs,
    ):
        # assert operator is None, "Operator must be None for unconditional diffusion"
        # assert measurement is None, "Measurement must be None for unconditional diffusion"
        pred_noises, pred_images = dm.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        measurement_error = 0.0
        return measurement_error, (pred_noises, pred_images)

    autograd = AutoGrad()
    autograd.set_function(compute_measurement_error)
    return autograd.get_gradient_and_value_jit_fn(has_aux=True)
