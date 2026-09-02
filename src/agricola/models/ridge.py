# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Quantitative-trait ridge regression."""

import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from jaxtyping import Array


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


def ridge_lowmem(X: Array, Y: Array, train_mask: Array, alphas: Array) -> Array:
    """Perform ridge regression with low peak memory by solving one alpha at a time."""
    _, b = X.shape
    _, _ = Y.shape
    _, _ = train_mask.shape

    Xm = X[:, :, None] * train_mask[:, None, :]
    XTY = jnp.einsum("nbk,np->kbp", Xm, Y)
    XTX = jnp.einsum("nbk,nck->kbc", Xm, Xm)
    identity = jnp.eye(b, dtype=XTY.dtype)

    beta = []
    for alpha in alphas:
        beta.append(cho_solve(cho_factor(XTX + alpha * identity), XTY))
    return jnp.stack(beta, axis=0)


__all__ = ["ridge", "ridge_lowmem"]
