# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Quantitative-trait ridge regression."""

import jax.numpy as jnp
from jaxtyping import Array
from jax.scipy.linalg import cho_factor, cho_solve


def ridge(X: Array, Y: Array, train_mask: Array, alphas: Array) -> Array:
    """Perform ridge regression for one or more training masks and penalties."""
    _, b = X.shape
    _, p = Y.shape
    _, k = train_mask.shape
    a = alphas.shape[0]

    Xm = X[:, :, None] * train_mask[:, None, :]
    XTY = jnp.einsum("nbk,np->kbp", Xm, Y)[None, :, :, :] + jnp.zeros((a, k, b, p))
    identity = jnp.eye(b, dtype=XTY.dtype)
    XTXs = jnp.einsum("nbk,nck->kbc", Xm, Xm)[None, :, :, :] + (
        alphas[:, None, None, None] * identity[None, None, :, :]
    )
    return cho_solve(cho_factor(XTXs), XTY)

__all__ = ["ridge"]
