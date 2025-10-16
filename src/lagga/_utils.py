import jax.numpy as jnp


def _stdize(X: jnp.ndarray) -> jnp.ndarray:
    """Safely standardizes an ndarray along the first dimension"""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = jnp.where(std == 0, 1, std)
    result = (X - mean) / std
    return result
