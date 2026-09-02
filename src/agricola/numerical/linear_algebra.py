# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Numerical preprocessing and covariate linear algebra."""

import jax.numpy as jnp
from jaxtyping import Array


def stdize(X: Array) -> Array:
    """Safely standardize an array along its first dimension."""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    mean = jnp.where(std == 0, 0, mean)
    std = jnp.where(std == 0, 1, std)
    return (X - mean) / std


def assert_covar_full_rank(X: Array) -> None:
    """Raise ``ValueError`` if a covariate matrix lacks full column rank."""
    rank = jnp.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        raise ValueError(f"Collinearity detected in : rank={rank}, n_features={X.shape[1]}")
