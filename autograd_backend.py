""" Allow different backends for autograd.
Date: 22/01/2024
"""

import functools

import jax
import keras
import numpy as np
import tensorflow as tf
import torch


class AutoGrad:
    """Wrapper class for autograd using different backends."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.function = None

        if verbose:
            print(f"Using backend: {self.backend}")

    @property
    def backend(self):
        """Get Keras backend. Machine learning library of choice."""
        return keras.backend.backend()

    @backend.setter
    def backend(self, backend):
        """Set Keras backend. Machine learning library of choice."""
        raise ValueError("Cannot change backend currently. Needs reimport of keras.")
        # keras.config.set_backend(backend)

    def set_function(self, function):
        """Set the function to calculate the gradients of."""
        self.function = function

    def gradient(self, variable, **kwargs):
        """Returns the gradients of the function w.r.t. variable.

        Args:
            variable (Tensor): Input tensor.
            **kwargs: Keyword arguments to pass to self.function.

        Returns:
            gradients (Tensor): Gradients of the function at variable.
                ∇f(x)
        """
        variable = keras.ops.convert_to_tensor(variable)
        if self.function is None:
            raise ValueError(
                "Function not set. Use `set_function` to set a custom function."
            )
        assert self.backend in [
            "torch",
            "tensorflow",
            "jax",
        ], f"Unsupported backend: {self.backend}"

        if self.backend == "torch":
            variable = variable.detach().requires_grad_(True)
            out = self.function(variable, **kwargs)
            gradients = torch.autograd.grad(out, variable)[0]
            return gradients
        elif self.backend == "tensorflow":
            with tf.GradientTape() as tape:
                tape.watch(variable)
                out = self.function(variable, **kwargs)
            gradients = tape.gradient(out, variable)
            return gradients
        elif self.backend == "jax":
            func = functools.partial(self.function, **kwargs)
            return jax.grad(func)(variable)

    def gradient_and_value(self, variable, has_aux: bool = False, **kwargs):
        """Returns both the gradients w.r.t. variable and outputs of the function.

        Note that self.function should return a tuple of (out, aux) if has_aux=True.
        with aux being a tuple of auxiliary variables.
        If has_aux=False, self.function should return out only.

        Args:
            variable (Tensor): Input tensor.
            has_aux (bool): Whether the function returns auxiliary variables.
            **kwargs: Keyword arguments to pass to self.function.

        Returns:
            gradients (Tensor): Gradients of the function at variable.
                ∇f(x)
            out (Tuple or Tensor): Outputs of the function at variable.
                if has_aux: out = (f(x), aux)
                else: out = f(x)
        """
        variable = keras.ops.convert_to_tensor(variable)
        if self.function is None:
            raise ValueError(
                "Function not set. Use `set_function` to set a custom function."
            )
        assert self.backend in [
            "torch",
            "tensorflow",
            "jax",
        ], f"Unsupported backend: {self.backend}"

        if self.backend == "torch":
            variable = variable.detach().requires_grad_(True)
            if has_aux:
                out, aux = self.function(variable, **kwargs)
            else:
                out = self.function(variable, **kwargs)
            gradients = torch.autograd.grad(out, variable)[0]
        elif self.backend == "tensorflow":
            with tf.GradientTape() as tape:
                tape.watch(variable)
                if has_aux:
                    out, aux = self.function(variable, **kwargs)
                else:
                    out = self.function(variable, **kwargs)
            gradients = tape.gradient(out, variable)
        elif self.backend == "jax":
            out, gradients = jax.value_and_grad(
                self.function, argnums=0, has_aux=has_aux
            )(variable, **kwargs)
            if has_aux:
                out, aux = out

        if has_aux:
            return gradients, (out, aux)
        return gradients, out

    def get_gradient_jit_fn(self):
        """Returns a jitted function for calculating the gradients."""
        if self.backend == "jax":
            return jax.jit(self.gradient)
        elif self.backend == "tensorflow":
            return tf.function(self.gradient, jit_compile=True)
        elif self.backend == "torch":
            return torch.compile(self.gradient)

    def get_gradient_and_value_jit_fn(self, has_aux: bool = False, disable_jit=False):
        """Returns a jitted function for calculating the gradients and function outputs."""
        if self.backend == "jax":
            # some models may not be compatible with jit but we default to jit for speed
            if disable_jit:
                return lambda x, **kwargs: self.gradient_and_value(
                    x, has_aux=has_aux, **kwargs
                )
            return jax.jit(
                lambda x, **kwargs: self.gradient_and_value(
                    x, has_aux=has_aux, **kwargs
                )
            )

        elif self.backend == "tensorflow":
            return tf.function(
                lambda x, **kwargs: self.gradient_and_value(
                    x, has_aux=has_aux, **kwargs
                )
            )
        elif self.backend == "torch":
            raise NotImplementedError("Jitting not supported for torch backend.")
            # return torch.compile(
            #     lambda x, **kwargs: self.gradient_and_value(
            #         x, has_aux=has_aux, **kwargs
            #     )
            # )
