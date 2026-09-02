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


def _logistic_ridge_gradient(
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
    return X.T @ residual - alpha * beta


def _logistic_ridge_step(
    beta: Array,
    X: Array,
    offset: Array,
    train_mask: Array,
    alpha: float | Array,
    gradient: Array,
) -> Array:
    eta = X @ beta + offset
    mu = expit(eta)
    weights = mu * (1 - mu) * train_mask
    weighted_x = X * weights[:, None]
    delta = cho_solve(
        cho_factor(X.T @ weighted_x + alpha * jnp.eye(X.shape[1])),
        gradient,
    )
    return beta + delta


def logistic_ridge(
    X: Array,
    y: Array,
    offset: Array,
    train_mask: Array,
    alpha: float | Array,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> Array:
    """Fit logistic ridge regression with an offset and training mask."""
    beta, _ = logistic_ridge_with_convergence(X, y, offset, train_mask, alpha, max_iter, tol)
    return beta


def logistic_ridge_lowmem(
    X: Array,
    y: Array,
    offset: Array,
    train_mask: Array,
    alphas: Array,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> Array:
    """Fit logistic ridge regression for many alphas with lower peak memory."""
    betas = []
    for alpha in alphas:
        beta = jax.vmap(
            lambda mask: logistic_ridge(X, y, offset, mask, alpha, max_iter, tol),
            in_axes=(1,),
        )(train_mask)
        betas.append(beta)
    return jnp.stack(betas, axis=0)


def logistic_ridge_with_convergence(
    X: Array,
    y: Array,
    offset: Array,
    train_mask: Array,
    alpha: float | Array,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> tuple[Array, Array]:
    """Fit logistic ridge regression and return whether it converged."""
    beta0 = jnp.zeros(X.shape[1])
    gradient0 = _logistic_ridge_gradient(beta0, X, y, offset, train_mask, alpha)
    total_weight = jnp.sum(train_mask)

    def cond_fun(state):
        i, _, _, converged = state
        return (i < max_iter) & ~converged

    def body_fun(state):
        i, beta, gradient, _ = state
        beta_new = _logistic_ridge_step(beta, X, offset, train_mask, alpha, gradient)
        gradient_new = _logistic_ridge_gradient(beta_new, X, y, offset, train_mask, alpha)
        max_gradient = jnp.max(jnp.abs(gradient_new)) / jnp.maximum(total_weight, 1)

        converged = (
            (total_weight > 0)
            & jnp.all(jnp.isfinite(beta_new))
            & jnp.all(jnp.isfinite(gradient_new))
            & (max_gradient <= tol)
        )

        return i + 1, beta_new, gradient_new, converged

    _, beta, _, converged = lax.while_loop(
        cond_fun,
        body_fun,
        (0, beta0, gradient0, jnp.array(False)),
    )

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


def logistic_ridge_loo_lowmem(
    X: Array,
    y: Array,
    offset: Array,
    alphas: Array,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Array:
    """Fit leave-one-out logistic ridge regression for many alphas with low memory."""
    betas = [logistic_ridge_loo(X, y, offset, alpha, max_iter, tol) for alpha in alphas]
    return jnp.stack(betas, axis=0)


__all__ = [
    "logistic_ridge",
    "logistic_ridge_loo",
    "logistic_ridge_loo_lowmem",
    "logistic_ridge_lowmem",
    "logistic_ridge_with_convergence",
]
