# MIT License
# Copyright (c) 2025 Franklin Ockerman
# See LICENSE.txt file for full license text

import pytest
import numpy as np
import jax.numpy as jnp
import jax
from lagga.models import ridge


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


def test_ridge_OLS_when_lambda_0(simple_data):
    """
    Ridge with alpha=0 should match the OLS solution.
    """
    X, Y, w_train, w_test = simple_data
    alphas = jnp.array([0.0])

    preds = ridge(X, Y, w_train, w_test, alphas)

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

    preds = ridge(X, Y, w_train, w_test, alphas)

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

    preds = ridge(X, Y, w_train, w_test, alphas)

    np.testing.assert_array_equal(
        preds,
        jnp.zeros_like(preds),
    )
