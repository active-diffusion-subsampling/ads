"""

TensorFlow and Jax implementations of basic MRI operations
Code adapted from: https://github.com/facebookresearch/fastMRI/

Keras3 doesn't yet support some of the relevant FFT functions
so for now it's easier to implement separately for different backends.

"""

import tensorflow as tf
import jax.numpy as jnp
import numpy as np
import jax
import keras.src.backend as backend
from typing import Optional, Sequence, Tuple, Union


def ifft2c(data, norm="ortho"):
    if backend.backend() == "tensorflow":
        return ifft2c_tf(data, norm)
    elif backend.backend() == "jax":
        return ifft2c_jax(data, norm)


def fft2c(data, norm="ortho"):
    if backend.backend() == "tensorflow":
        return fft2c_tf(data, norm)
    elif backend.backend() == "jax":
        return fft2c_jax(data, norm)


def ifftshift(data, axes):
    if backend.backend() == "tensorflow":
        return tf.signal.ifftshift(data, axes=axes)
    elif backend.backend() == "jax":
        return jnp.fft.ifftshift(data, axes=axes)


def fftshift(data, axes):
    if backend.backend() == "tensorflow":
        return tf.signal.fftshift(data, axes=axes)
    elif backend.backend() == "jax":
        return jnp.fft.fftshift(data, axes=axes)


def complex_abs(data):
    if backend.backend() == "tensorflow":
        return complex_abs_tf(data)
    elif backend.backend() == "jax":
        return complex_abs_jax(data)


def view_as_complex_jax(data):
    """
    Converts to complex-valued tensor, assuming the last dimension
    is of size 2 containing real and complex values, respectively
    """
    assert data.shape[-1] == 2
    return jax.lax.complex(data[..., -2], data[..., -1])


def view_as_real_jax(data):
    """
    Converts a complex-valued tensor to a real-valued tensor by
    simply taking the imaginary components as real components and
    stacking along the final axis.
    Aims to re-implement https://pytorch.org/docs/stable/generated/torch.view_as_real.html
    """
    real_part = jnp.real(data)
    imaginary_part = jnp.imag(data)
    return jnp.stack([real_part, imaginary_part], axis=-1)


def fft2c_jax(data, norm: str = "ortho"):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode.

    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = jnp.fft.ifftshift(data, axes=[-3, -2])
    data = view_as_real_jax(
        jnp.fft.fftn(view_as_complex_jax(data), axes=(-2, -1), norm=norm)
    )
    # data = jnp.fft.fftshift(data, axes=[-3, -2])

    return data


def fft2c_tf(data, norm: str = "ortho"):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode.

    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = tf.signal.ifftshift(data, axes=[-3, -2])
    data = view_as_complex_tf(data)
    data = view_as_real_tf(
        tf.signal.fftnd(data, data.shape[-2:], axes=(-2, -1), norm=norm)
    )
    # data = tf.signal.fftshift(data, axes=[-3, -2])

    return data


def complex_abs_tf(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.shape[-1] == 2
    return tf.sqrt(tf.reduce_sum(data**2, axis=-1))


def complex_abs_jax(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.shape[-1] == 2
    return jnp.sqrt(jnp.sum(data**2, axis=-1))


def view_as_real_tf(data):
    """
    Converts a complex-valued tensor to a real-valued tensor by
    simply taking the imaginary components as real components and
    stacking along the final axis.
    Aims to re-implement https://pytorch.org/docs/stable/generated/torch.view_as_real.html
    """
    real_part = tf.math.real(data)
    imaginary_part = tf.math.imag(data)
    return tf.stack([real_part, imaginary_part], axis=-1)


def view_as_complex_tf(data):
    """
    Converts to complex-valued tensor, assuming the last dimension
    is of size 2 containing real and complex values, respectively
    """
    assert data.shape[-1] == 2
    return tf.complex(data[..., -2], data[..., -1])


def ifft2c_jax(data, norm="ortho"):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``ops.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = jnp.fft.ifftshift(data, axes=[-3, -2])
    data = view_as_real_jax(
        jnp.fft.ifftn(view_as_complex_jax(data), axes=(-2, -1), norm=norm)
    )
    data = jnp.fft.fftshift(data, axes=[-3, -2])

    return data


def ifft2c_tf(data, norm="ortho"):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``ops.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = tf.signal.ifftshift(data, axes=[-3, -2])
    data = view_as_real_tf(
        tf.signal.ifftnd(view_as_complex_tf(data), axes=(-2, -1), norm=norm)
    )
    data = tf.signal.fftshift(data, axes=[-3, -2])

    return data


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``MaskFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``MaskFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass ``MaskFunc``
    and overwrite the ``sample_mask`` logic. See examples in ``RandomMaskFunc``
    and ``EquispacedMaskFunc``.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        seed: int = None,
        allow_any_combination: bool = False,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
        """
        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of accelerations "
                "if allow_any_combination is False."
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng_key = jax.random.key(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ):
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        center_mask, accel_mask, num_low_frequencies = self.sample_mask(shape, offset)

        # combine masks together
        return (
            jnp.logical_or(center_mask, accel_mask).astype(jnp.float32),
            num_low_frequencies,
        )

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ):
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask, shape: Sequence[int]):
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols

        return jnp.array(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_low_frequencies: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs

        return mask

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.allow_any_combination:
            return jax.random.choice(
                self.rng_key, self.center_fractions
            ), jax.random.choice(self.rng_key, self.accelerations)
        else:
            choice = int(
                jax.random.randint(
                    self.rng_key,
                    minval=0,
                    maxval=len(self.center_fractions),
                    shape=(1,),
                )
            )
            return self.center_fractions[choice], self.accelerations[choice]


class RandomMaskFunc(MaskFunc):
    """
    Creates a random sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / acceleration - num_low_frequencies) / (
            num_cols - num_low_frequencies
        )

        return jax.random.uniform(self.rng_key, shape=(num_cols,)) < prob


def apply_mask(
    data,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies
