# MIT License
# Copyright (c) 2025 Franklin Ockerman
# See LICENSE.txt file for full license text

import pytest
import numpy as np
import jax.numpy as jnp
import jax
from lagga.models import (
    ridge_masked_predict,
    logistic_fit,
    logistic_ridge_predict,
    logistic_ridge_loo_predict,
)


@pytest.fixture
def simple_data():
    """
    Simple deterministic dataset where OLS is well-defined.
    """
    X = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    Y = jnp.array([1.0, 2.0, 3.0])
    w_train = jnp.ones((X.shape[0], 1))
    w_test = jnp.ones((X.shape[0], 1))
    return X, Y, w_train, w_test


### ─────────────────────────────────────────────────────────────
### Ridge
### ─────────────────────────────────────────────────────────────


def test_ridge_OLS_when_lambda_0(simple_data):
    """
    Ridge with alpha=0 should match the OLS solution.
    """
    X, Y, w_train, w_test = simple_data
    alphas = jnp.array([0.0])

    preds = ridge_masked_predict(X, Y, w_train, w_test, alphas)

    # Closed-form OLS
    beta_ols = jnp.linalg.solve(X.T @ X, X.T @ Y)
    preds_ols = (X @ beta_ols).reshape(-1, 1)

    np.testing.assert_allclose(
        preds[:, 0, 0],
        preds_ols[:, 0],
        rtol=1e-5,
        atol=1e-5,
    )


def test_ridge_zero_when_big_lambda(simple_data):
    """
    Very large ridge penalty should drive coefficients to ~0,
    hence predictions ~0.
    """
    X, Y, w_train, w_test = simple_data
    alphas = jnp.array([1e6])

    preds = ridge_masked_predict(X, Y, w_train, w_test, alphas)

    np.testing.assert_allclose(
        preds,
        jnp.zeros_like(preds),
        atol=1e-4,
    )


def test_ridge_zero_when_null_test_set(simple_data):
    """
    Predictions must be zeroed out when w_test == 0.
    """
    X, Y, w_train, _ = simple_data
    w_test = jnp.zeros((X.shape[0], 1))
    alphas = jnp.array([0.0, 1.0])

    preds = ridge_masked_predict(X, Y, w_train, w_test, alphas)

    np.testing.assert_array_equal(
        preds,
        jnp.zeros_like(preds),
    )


### ─────────────────────────────────────────────────────────────
### Logistic Regression
### ─────────────────────────────────────────────────────────────


def test_logistic_runs():
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

    beta = logistic_fit(X, y, offset)

    assert beta.shape == (2,)
    assert jnp.all(jnp.isfinite(beta))
    np.testing.assert_allclose(beta, np.array([-0.4196176, -0.4196176]), rtol=1e-6)


def test_logistic_ridge_runs():
    """Ridge regression returns linear predictors of length N."""
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
    w_train = jnp.ones(X.shape[0])

    eta = logistic_ridge_predict(X, y, offset, w_train, alpha=1.0)

    assert eta.shape == (4,)
    assert jnp.all(jnp.isfinite(eta))


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

    eta = logistic_ridge_loo_predict(X, y, offset, alpha=1.0)

    assert eta.shape == (4,)
    assert jnp.all(jnp.isfinite(eta))
