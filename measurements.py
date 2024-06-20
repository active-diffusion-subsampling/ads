"""Measurements module.
Handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.
Inspired by DPS repository:
- https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/guided_diffusion/measurements.py
"""

import abc

import numpy as np
from jax import tree_util
from keras import ops

from utils import mri
from utils.keras_utils import check_keras_backend

check_keras_backend()

_OPERATORS = {}


def register_operator(cls=None, *, name=None):
    """A decorator for registering operator classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _OPERATORS:
            raise ValueError(f"Already registered operator with name: {local_name}")
        _OPERATORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_operator(name):
    """Get operator class for given name."""
    assert (
        name in _OPERATORS
    ), f"Operator {name} not found. Available operators: {_OPERATORS.keys()}"
    return _OPERATORS[name]


class LinearOperator(abc.ABC):
    """Linear operator class y = Ax + n."""

    sigma = 0.0

    @abc.abstractmethod
    def forward(self, data):
        """Implements the forward operator A: x -> y."""
        raise NotImplementedError

    @abc.abstractmethod
    def corrupt(self, data):
        """Corrupt the data. Similar to forward but with noise."""
        raise NotImplementedError

    @abc.abstractmethod
    def transpose(self, data):
        """Implements the transpose operator A^T: y -> x."""
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        """String representation of the operator."""
        raise NotImplementedError

    @classmethod
    def _tree_unflatten(cls, aux, children):
        return cls(*children)

    def _tree_flatten(self):
        return (), ()


@register_operator(name="inpainting")
class InpaintingOperator(LinearOperator):
    """Inpainting operator A = I * M."""

    def __init__(self, mask):
        self.mask = mask

    def forward(self, data):
        return data * self.mask

    def corrupt(self, data):
        return self.forward(data)

    def transpose(self, data):
        return data * self.mask

    def __str__(self):
        return "y = Ax + n, where A = I * M"

    def _tree_flatten(self):
        return (self.mask,), ()


@register_operator(name="fourier")
class FourierOperator(LinearOperator):
    """Fourier operator A = F."""

    def forward(self, data):
        return mri.fft2c(data)

    def corrupt(self, data):
        return mri.fft2c(data)

    def transpose(self, data):
        # Fourier transform is unitary --> adjoint is inverse
        # https://math.stackexchange.com/questions/1429086/prove-the-fourier-transform-is-a-unitary-linear-operator
        raise mri.ifft2c(data)

    def __str__(self):
        return "y = F(x)"


@register_operator(name="masked_fourier")
class MaskedFourierOperator(LinearOperator):
    """Masked Fourier operator A = M*F, where M is a binary mask"""

    def __init__(self, mask):
        self.mask = mask

    def forward(self, data):
        return self.mask * mri.fft2c(data)

    def corrupt(self, data):
        return self.mask * mri.fft2c(data)

    def transpose(self, data):
        # Fourier transform is unitary --> adjoint is inverse
        # https://math.stackexchange.com/questions/1429086/prove-the-fourier-transform-is-a-unitary-linear-operator
        raise self.mask * mri.ifft2c(data)

    def __str__(self):
        return "y = M*F(x)"

    def _tree_flatten(self):
        return (self.mask,), ()


def prepare_measurement(operator_name, target_imgs, **measurement_kwargs):
    """
    Prepare measurement given operator name and target images.
    Just an easy way of quickly generating random measurements given clean images.

    Args:
        operator_name (str): The name of the operator to be used for the measurement process.
        target_imgs (Tensor): The target images for the measurement.
        measurement_kwargs (dict, optional): Additional keyword arguments to be passed to the operator.
            If not specified, default values will be used for each operator.

    Returns:
        tuple: A tuple containing the operator and the measurements.
            - operator: The chosen forward operator for the measurement process.
            - measurements: The corrupted measurements obtained using the chosen operator.

    Raises:
        ValueError: If the specified `operator_name` is not recognized.

    Note:
        - The function supports the following operator names:
            - "inpainting": Inpainting operator.
            - "masked_fourier": Operator first applying Fourier transform, and then a mask.

    """
    operator = get_operator(operator_name)

    # set defaults for each operator --  configurable by changing operator.mask
    if not measurement_kwargs:
        if operator_name == "masked_fourier":
            # default to a centered 4x acceleration mask.
            mask = (
                ops.zeros_like(target_imgs.shape[1:]).at[0, 32 + 16 : 64 + 16, 0].set(1)
            )
            measurement_kwargs = {"mask": mask}
        elif operator_name == "inpainting":
            # default to a mask hiding half pixels in the image.
            # Build measurement mask
            image_shape = target_imgs.shape[1:]
            mask = np.zeros(image_shape, dtype="float32")
            # mask out random half of pixels of the image
            n_total_samples = image_shape[0] * image_shape[1]
            random_idx = np.random.choice(
                n_total_samples, size=n_total_samples // 2, replace=False
            )
            random_idx = np.unravel_index(random_idx, image_shape[:-1])
            mask[random_idx] = 1
            mask = mask[None, :, :]  # add batch dimension
            measurement_kwargs = {"mask": mask}
        else:
            raise ValueError(f"Operator `{operator_name}` not recognised.")

    operator = operator(**measurement_kwargs)

    measurements = operator.corrupt(target_imgs)
    return operator, measurements


# register all classes for jax tree flattening
# allows us to use operator class as arguments in jitted jax functions
# https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
for cls in LinearOperator.__subclasses__():
    tree_util.register_pytree_node(
        cls,
        cls._tree_flatten,
        cls._tree_unflatten,
    )
