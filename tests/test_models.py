# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

import pytest
import numpy as np
import jax.numpy as jnp
from lagga._internal.models import (
    ridge,
    logistic_ridge,
    logistic_ridge_loo,
)


@pytest.fixture
def toy_data():
    """
    Simple deterministic dataset where OLS is well-defined.
    """
    X = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    Y = jnp.array([1.0, 2.0, 3.0])
    w_train = jnp.ones((X.shape[0]))
    w_test = jnp.ones((X.shape[0]))
    return X, Y, w_train, w_test


### ─────────────────────────────────────────────────────────────
### Ridge
### ─────────────────────────────────────────────────────────────


def test_ridge_OLS_when_lambda_0(toy_data):
    """
    Ridge with alpha=0 should match the OLS solution.
    """
    X, Y, w_train, _ = toy_data
    alphas = jnp.array([0.0])

    beta = ridge(X, Y, w_train, alphas)[:, 0]

    # Closed-form OLS
    beta_ols = jnp.linalg.solve(X.T @ X, X.T @ Y)

    np.testing.assert_allclose(
        beta,
        beta_ols,
        rtol=1e-5,
        atol=1e-5,
    )


def test_ridge_zero_when_big_lambda(toy_data):
    """
    Very large ridge penalty should drive coefficients to ~0,
    hence predictions ~0.
    """
    X, Y, w_train, _ = toy_data
    alphas = jnp.array([1e6])

    beta = ridge(X, Y, w_train, alphas)

    np.testing.assert_allclose(beta, jnp.zeros_like(beta), atol=1e-4)


### ─────────────────────────────────────────────────────────────
### Logistic Regression
### ─────────────────────────────────────────────────────────────


def test_logistic_ridge_runs():
    """test that logistic regression runs and returns correct coefficients"""
    X = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    y = jnp.array([0.0, 0.0, 1.0, 0.0])
    offset = jnp.zeros(X.shape[0])

    beta = logistic_ridge(X, y, offset, jnp.ones(X.shape[0]), 0)

    assert beta.shape == (2,)
    assert jnp.all(jnp.isfinite(beta))
    np.testing.assert_allclose(beta, np.array([-0.4196176, -0.4196176]), rtol=1e-6)


def test_logistic_ridge_loo_runs():
    """LOO ridge regression runs and returns finite values."""
    X = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    y = jnp.array([0.0, 0.0, 1.0, 0.0])
    offset = jnp.zeros(X.shape[0])

    beta = logistic_ridge_loo(X, y, offset, alpha=1.0)

    assert beta.shape == (2, 4)
    assert jnp.all(jnp.isfinite(beta))
