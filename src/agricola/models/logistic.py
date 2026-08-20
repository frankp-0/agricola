# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Binary-trait logistic ridge regression."""

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from jax.scipy.special import expit
from jaxtyping import Array


def _logistic_ridge_step(
    beta: Array,
    X: Array,
    y: Array,
    offset: Array,
    train_mask: Array,
    alpha: float | Array,
) -> Array:
    eta = X @ beta + offset
    mu = expit(eta)
    residual = (y - mu) * train_mask
    weights = mu * (1 - mu) * train_mask
    weighted_x = X * weights[:, None]
    delta = cho_solve(
        cho_factor(X.T @ weighted_x + alpha * jnp.eye(X.shape[1])),
        X.T @ residual,
    )
    return beta + delta


def logistic_ridge(
    X: Array,
    y: Array,
    offset: Array,
    train_mask: Array,
    alpha: float | Array,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Array:
    """Fit logistic ridge regression with an offset and training mask."""
    beta, _ = logistic_ridge_with_convergence(X, y, offset, train_mask, alpha, max_iter, tol)
    return beta


def logistic_ridge_with_convergence(
    X: Array,
    y: Array,
    offset: Array,
    train_mask: Array,
    alpha: float | Array,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[Array, Array]:
    """Fit logistic ridge regression and return whether it converged."""
    beta0 = jnp.zeros(X.shape[1])

    def cond_fun(state):
        i, _, converged = state
        return (i < max_iter) & ~converged

    def body_fun(state):
        i, beta, _ = state
        beta_new = _logistic_ridge_step(beta, X, y, offset, train_mask, alpha)
        converged = jnp.max(jnp.abs(beta_new - beta)) <= tol
        return i + 1, beta_new, converged

    _, beta, converged = lax.while_loop(cond_fun, body_fun, (0, beta0, jnp.array(False)))
    return beta, converged


def logistic_ridge_loo(
    X: Array,
    y: Array,
    offset: Array,
    alpha: float | Array,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Array:
    """Fit logistic ridge regression and return leave-one-out coefficients."""
    beta = logistic_ridge(X, y, offset, jnp.ones(X.shape[0]), alpha, max_iter, tol)
    eta = X @ beta + offset
    mu = expit(eta)
    weights = mu * (1 - mu)
    weighted_x = X * weights[:, None]
    h_inv = jax.scipy.linalg.inv(X.T @ weighted_x + alpha * jnp.eye(X.shape[1]))

    def quad_form_h_inv(x):
        return (x @ h_inv) @ x.T

    gamma = weights * jax.vmap(quad_form_h_inv, in_axes=0)(X)
    return beta[:, None] - ((h_inv @ X.T) * (y - mu) / (1 - gamma))


__all__ = [
    "logistic_ridge",
    "logistic_ridge_loo",
    "logistic_ridge_with_convergence",
]
