# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Functions for fitting regression models.

Key functions:
    ridge: ridge regression predictions with train mask
    logistic_ridge: logistic ridge regression predictions with train mask
    logistic_ridge_loo: leave-one-out logistic ridge regression
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.scipy.special import expit
from jaxtyping import Array

### ─────────────────────────────────────────────────────────────
### Quantitative Traits
### ─────────────────────────────────────────────────────────────


def ridge(X: Array, Y: Array, train_mask: Array, alphas: Array) -> Array:
    """Perform ridge regression using train masks

    train_mask should only take values 0 and 1.
    Passing "masks" like this allows data to have the
    same input size regardless of train/test split and avoid recompilation
    should we choose to use JIT compilation.


    Args:
        X: (N, V) jax array of predictors
        Y: (N, P) or (N,) jax array of outcome(s)
        train_mask: (N, 1) jax array indicating training set status (0/1)
        alphas: A 1d jax array of ridge penalty weights

    Returns:
        beta: A (V, P, len(alphas)) jax array of predictions (with non-test samples masked out)
    """
    _, b = X.shape

    ## Reshape weights to 1D and Y to 2D
    train_mask = train_mask.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    ## Perform ridge regression
    XTX = X.T @ (X * train_mask)
    XTY = X.T @ (Y * train_mask)
    I_ = jnp.eye(b, dtype=XTY.dtype)

    def ridge_fit(alpha):
        A = XTX + alpha * I_
        beta = jnp.linalg.solve(A, XTY)
        return beta

    beta = jax.vmap(ridge_fit)(alphas)
    beta = jnp.moveaxis(beta, 0, -1)

    return beta


### ─────────────────────────────────────────────────────────────
### Binary Traits
### ─────────────────────────────────────────────────────────────


def _logistic_ridge_step(
    beta: Array,
    X: Array,
    y: Array,
    offset: Array,
    train_mask: Array,
    alpha: float | Array,
) -> Array:
    """One Newton/IRLS update in logistic ridge regression."""
    eta = X @ beta + offset
    mu = expit(eta)
    r = (y - mu) * train_mask
    w = mu * (1 - mu) * train_mask
    XW = X * w[:, None]
    XT_r = X.T @ r
    H = (X.T @ XW) + (alpha * jnp.eye(X.shape[1]))
    delta = jnp.linalg.solve(H, XT_r)
    beta_new = beta + delta
    return beta_new


def logistic_ridge(
    X: Array,
    y: Array,
    offset: Array,
    train_mask: Array,
    alpha: float | Array,
    max_iter: int = 50,
) -> Array:
    """Perform logistic ridge regression using a training mask.

    Fits a logistic ridge regression model including offsets

    Args:
        X: (N, V) jax array of predictors
        y: (N,) jax array of the outcome
        offset: (N,) jax array with offset
        train_mask: (N,) jax array indicating training set status (0/1)
        alpha: ridge penalty weight
        max_iter: max number of iterations

    Returns:
        beta: (V,) jax array of coefficients
    """
    beta0 = jnp.zeros(X.shape[1])

    def body_fun(i, beta):
        return _logistic_ridge_step(beta, X, y, offset, train_mask, alpha)

    beta = lax.fori_loop(0, max_iter, body_fun, beta0)

    return beta


def logistic_ridge_loo(
    X: Array, y: Array, offset: Array, alpha: float | Array, max_iter: int = 50
) -> Array:
    """Perform logistic ridge regression with leave-one-out scheme

    Returns leave-one-out linear predictor

    Args:
        X: (N, V) jax array of predictors
        y: (N,) jax array of the outcome
        offset: (N,) jax array with offset
        alpha: ridge penalty weight
        max_iter: max number of iterations

    Returns:
        beta_loo: (V,N) jax array of loo coefficients
    """
    beta0 = jnp.zeros(X.shape[1])

    train_mask = jnp.ones((X.shape[0]))

    def body_fun(i, beta):
        return _logistic_ridge_step(beta, X, y, offset, train_mask, alpha)

    beta = lax.fori_loop(0, max_iter, body_fun, beta0)
    eta = X @ beta + offset
    mu = expit(eta)
    w = mu * (1 - mu)
    XW = X * w[:, None]
    H_inv = jax.scipy.linalg.inv((X.T @ XW) + (alpha * jnp.eye(X.shape[1])))

    def _quad_form_Hinv(x):
        return (x @ H_inv) @ x.T

    Gamma = w * jax.vmap(_quad_form_Hinv, in_axes=0)(X)

    beta_loo = beta[:, None] - ((H_inv @ X.T) * (y - mu) / (1 - Gamma))

    return beta_loo
