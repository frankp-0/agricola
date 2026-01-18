# MIT License
# Copyright (c) 2025 Franklin Ockerman
# See LICENSE.txt file for full license text

import pytest
import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.special import expit
from lagga.step1 import step1


### ─────────────────────────────────────────────────────────────
### Input Validation
### ─────────────────────────────────────────────────────────────


def valid_inputs(R=16, P=3, K=5, C=1, H=4):
    """Utility for constructing minimal valid inputs."""
    key0 = jax.random.key(8899134)
    key1 = jax.random.key(471464)
    key2 = jax.random.key(7314120)
    N = 20
    Z = {
        str(i): np.asarray(
            jax.random.normal(jax.random.split(key0, 3)[i], shape=(N, P, R))
        )
        for i in range(20, 23)
    }
    Y = jax.random.normal(key1, shape=(N, P))
    X = jax.random.normal(key2, shape=(N, C))
    train_mask = jnp.ones((N, K))
    test_mask = jnp.ones((N, K))
    h2_prior = jnp.linspace(0.1, 0.9, H)
    return Z, Y, X, train_mask, test_mask, h2_prior


def test_step1_Z_dim_error():
    """Check that bad Z dimensions throws error"""
    Z, Y, X, train, test, h2 = valid_inputs()
    Z["0"] = np.ones(shape=(1, 6, 4))

    with pytest.raises(ValueError, match="All Z arrays must have same"):
        step1(Z, Y, X, train, test, h2, "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError):
        step1(Z, Y, X, train, test, h2, "bt")


def test_step1_Y_dim_error():
    """Check that 1D Y throws error"""
    Z, _, X, train, test, h2 = valid_inputs()
    with pytest.raises(ValueError, match="Y must be 2D"):
        step1(Z, jnp.zeros(5), X, train, test, h2, "qt")

    with pytest.raises(ValueError, match="Y must be 2D"):
        step1(Z, jnp.zeros(5), X, train, test, h2, "bt")


def test_step1_X_dim_error():
    """Check that 1D X throws error"""
    Z, Y, _, train, test, h2 = valid_inputs()
    with pytest.raises(ValueError, match="X must be 2D"):
        step1(Z, Y, jnp.zeros((10,)), train, test, h2, "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="X must be 2D"):
        step1(Z, Y, jnp.zeros((10,)), train, test, h2, "bt")


def test_step1_X_matches_N_error():
    """Check that N mis-match with X throws error"""
    Z, Y, _, train, test, h2 = valid_inputs()
    with pytest.raises(ValueError, match="must match Y.shape"):
        step1(Z, Y, jnp.zeros((5, 2)), train, test, h2, "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="must match Y.shape"):
        step1(Z, Y, jnp.zeros((5, 2)), train, test, h2, "bt")


def test_step1_mask_dim_error():
    """Check that 1D train/test mask throws error"""
    Z, Y, X, _, _, h2 = valid_inputs()
    with pytest.raises(ValueError, match="train_mask and test_mask must be 2D"):
        step1(Z, Y, X, jnp.ones((10,)), jnp.ones((10, 2)), h2, "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="train_mask and test_mask must be 2D"):
        step1(Z, Y, X, jnp.ones((10,)), jnp.ones((10, 2)), h2, "bt")


def test_step1_mask_shape_mismatch_error():
    """Check that train/test mask shape mis-match throws error"""
    Z, Y, X, _, _, h2 = valid_inputs()
    with pytest.raises(ValueError, match="same shape"):
        step1(Z, Y, X, jnp.ones((10, 2)), jnp.ones((10, 3)), h2, "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="same shape"):
        step1(Z, Y, X, jnp.ones((10, 2)), jnp.ones((10, 3)), h2, "bt")


def test_step1_h2_prior_dim_error():
    """Check that h2_prior with wrong dimension throws error"""
    Z, Y, X, train, test, _ = valid_inputs()
    with pytest.raises(ValueError, match="h2_prior must be 1D"):
        step1(Z, Y, X, train, test, jnp.ones((3, 2)), "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="h2_prior must be 1D"):
        step1(Z, Y, X, train, test, jnp.ones((3, 2)), "bt")


def test_step1_h2_prior_domain_error():
    """Check that h2_prior outside (0,1) throws error"""
    Z, Y, X, train, test, _ = valid_inputs()
    with pytest.raises(ValueError, match="in the open interval"):
        step1(Z, Y, X, train, test, jnp.array([0.5, 0.0, 0.7]), "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="in the open interval"):
        step1(Z, Y, X, train, test, jnp.array([0.5, 0.0, 0.7]), "bt")


### ─────────────────────────────────────────────────────────────
### Other
### ─────────────────────────────────────────────────────────────


def test_step1_qt_valid_input():
    """Check that valid data throws no type errors"""
    Z, Y, X, train, test, h2 = valid_inputs()
    step1(Z, Y, X, train, test, h2, "qt")


def test_step1_bt_valid_input():
    """Check that valid data throws no type errors"""
    ## cross-validation
    Z, Y, X, train, test, h2 = valid_inputs()
    Y = jnp.round(expit(Y))
    step1(Z, Y, X, train, test, h2, "bt", False)

    ## loco
    Z, Y, X, train, test, h2 = valid_inputs()
    step1(Z, Y, X, None, None, h2, "bt", True)
