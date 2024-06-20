import keras.ops as ops


def L2(x):
    """
    Implementation of L2 norm: https://mathworld.wolfram.com/L2-Norm.html
    """
    return ops.sqrt(ops.sum(x**2))
