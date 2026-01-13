# MIT License
# Copyright (c) 2025 Franklin Ockerman
# See LICENSE.txt file for full license text

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.scipy.special import expit

### ─────────────────────────────────────────────────────────────
### Quantitative Traits
### ─────────────────────────────────────────────────────────────


def ridge_masked_predict(X, Y, train_mask, test_mask, alphas):
    """Perform ridge regression using test/train masks

    Both train_mask and test_mask should only take values 0 and 1.
    Samples with test_mask = 0 have predicted value 0
    in the output. Passing "masks" like this allows data to have the
    same input size regardless of train/test split and avoid recompilation
    should we choose to use JIT compilation.


    Args:
        X: (N, V) jax array of predictors
        Y: (N, P) or (N,) jax array of outcome(s)
        train_mask: (N, 1) jax array indicating training set status (0/1)
        test_mask: (N, 1) jax array indicating test set status (0/1)
        alphas: A 1d jax array of ridge penalty weights

    Returns:
        preds: A (N, len(alphas), P) jax array of predictions (with non-test samples masked out)
    """
    _, b = X.shape

    ## Reshape weights to 1D and Y to 2D
    train_mask = train_mask.reshape(-1, 1)
    test_mask = test_mask.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    ## Perform ridge regression
    XTX = X.T @ (X * train_mask)
    XTY = X.T @ (Y * train_mask)
    I_ = jnp.eye(b, dtype=jnp.float32)

    def ridge_pred(alpha):
        A = XTX + alpha * I_
        beta = jnp.linalg.solve(A, XTY)
        return (X @ beta) * test_mask

    preds = jax.vmap(ridge_pred)(alphas)
    preds = jnp.moveaxis(preds, 0, -1)

    return preds


### ─────────────────────────────────────────────────────────────
### Binary Traits
### ─────────────────────────────────────────────────────────────


def _logistic_ridge_step(beta, X, y, offset, train_mask, alpha):
    """One Newton/IRLS update in logistic ridge regression

    Args:
        beta: (V,) jax array of current coefficients
        X: (N, V) jax array of predictors
        y: (N,) jax array of the outcome
        offset: (N,) jax array with offset
        train_mask: (N,) jax array indicating training set status (0/1)
        alpha: ridge penalty weight

    Returns:
        beta: (V,) jax array of coefficients
    """

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


def logistic_fit(X, y, offset, max_iter=20, alpha=0):
    """Perform logistic regression

    Returns estimated coefficients

    Args:
        X: (N, V) jax array of predictors
        y: (N,) jax array of the outcome
        offset: (N,) jax array with offset
        max_iter: max number of iterations

    Returns:
        beta: (N,) jax array of linear predictors
    """
    beta0 = jnp.zeros(X.shape[1])

    def body_fun(i, beta):
        return _logistic_ridge_step(beta, X, y, offset, jnp.ones(X.shape[0]), alpha)

    beta = lax.fori_loop(0, max_iter, body_fun, beta0)

    return beta


def logistic_ridge_predict(X, y, offset, train_mask, alpha, max_iter=50):
    """Perform logistic ridge regression

    Returns linear predictors

    Args:
        X: (N, V) jax array of predictors
        y: (N,) jax array of the outcome
        offset: (N,) jax array with offset
        train_mask: (N,) jax array indicating training set status (0/1)
        alpha: ridge penalty weight
        max_iter: max number of iterations

    Returns:
        eta: (N,) jax array of linear predictors
    """
    beta0 = jnp.zeros(X.shape[1])

    def body_fun(i, beta):
        return _logistic_ridge_step(beta, X, y, offset, train_mask, alpha)

    beta = lax.fori_loop(0, max_iter, body_fun, beta0)
    eta = X @ beta + offset

    return eta


def logistic_ridge_loo_predict(X, y, offset, alpha, max_iter=50):
    """Perform logistic ridge regression with leave-one-out scheme

    Returns leave-one-out linear predictor

    Args:
        X: (N, V) jax array of predictors
        y: (N,) jax array of the outcome
        offset: (N,) jax array with offset
        alpha: ridge penalty weight
        max_iter: max number of iterations

    Returns:
        eta_loo: (N,) jax array of linear predictions
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

    def foo(x):
        return (x @ H_inv) @ x.T

    Gamma = w * jax.vmap(foo, in_axes=0)(X)

    beta_loo = beta[:, None] - ((H_inv @ X.T) * (y - mu) / (1 - Gamma))
    eta_loo = jnp.sum(X * (beta_loo.T), axis=1) + offset

    return eta_loo
