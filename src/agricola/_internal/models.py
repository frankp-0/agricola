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
from jax.scipy.linalg import cho_factor, cho_solve

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
        train_mask: (N, 5) jax array indicating training set status (0/1)
        alphas: A 1d jax array of ridge penalty weights

    Returns:
        beta: A (len(alphas), 5, V, P) jax array of coefficients
    """
    _, b = X.shape
    _, p = Y.shape
    _, k = train_mask.shape
    a = alphas.shape[0]

    Xm = X[:, :, None] * train_mask[:, None, :]
    XTY = jnp.einsum("nbk,np->kbp", Xm, Y)[None, :, :, :] + jnp.zeros((a, k, b, p))
    I_ = jnp.eye(b, dtype=XTY.dtype)
    XTXs = jnp.einsum("nbk,nck->kbc", Xm, Xm)[None, :, :, :] + (
        alphas[:, None, None, None] * I_[None, None, :, :]
    )
    Ls = cho_factor(XTXs)
    beta = cho_solve(Ls, XTY)

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
    L = cho_factor((X.T @ XW) + (alpha * jnp.eye(X.shape[1])))
    delta = cho_solve(L, XT_r)
    beta_new = beta + delta
    return beta_new


def logistic_ridge(
    X: Array,
    y: Array,
    offset: Array,
    train_mask: Array,
    alpha: float | Array,
    max_iter: int = 50,
    tol: float = 1e-6,
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
        tol: convergence tolerance

    Returns:
        beta: (V,) jax array of coefficients
    """

    beta0 = jnp.zeros(X.shape[1])

    def cond_fun(state):
        i, beta = state
        beta_new = _logistic_ridge_step(beta, X, y, offset, train_mask, alpha)
        return (i < max_iter) & (jnp.max(jnp.abs(beta_new - beta)) > tol)

    def body_fun(state):
        i, beta = state
        beta_new = _logistic_ridge_step(beta, X, y, offset, train_mask, alpha)
        return i + 1, beta_new

    _, beta = lax.while_loop(cond_fun, body_fun, (0, beta0))

    return beta


def logistic_ridge_loo(
    X: Array,
    y: Array,
    offset: Array,
    alpha: float | Array,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Array:
    """Perform logistic ridge regression with leave-one-out scheme

    Returns leave-one-out linear predictor

    Args:
        X: (N, V) jax array of predictors
        y: (N,) jax array of the outcome
        offset: (N,) jax array with offset
        alpha: ridge penalty weight
        max_iter: max number of iterations
        tol: convergence tolerance

    Returns:
        beta_loo: (V,N) jax array of loo coefficients
    """
    beta0 = jnp.zeros(X.shape[1])

    train_mask = jnp.ones((X.shape[0]))

    def cond_fun(state):
        i, beta = state
        beta_new = _logistic_ridge_step(beta, X, y, offset, train_mask, alpha)
        return (i < max_iter) & (jnp.max(jnp.abs(beta_new - beta)) > tol)

    def body_fun(state):
        i, beta = state
        beta_new = _logistic_ridge_step(beta, X, y, offset, train_mask, alpha)
        return i + 1, beta_new

    _, beta = lax.while_loop(cond_fun, body_fun, (0, beta0))
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
