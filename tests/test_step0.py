# MIT License
# Copyright (c) 2025 Franklin Ockerman
# See LICENSE.txt file for full license text

import pytest
import jax.numpy as jnp
import jax
from lanctools import LancData
from lagga.step0 import step0


@pytest.fixture
def toy_data():
    data = [
        LancData(
            plink_prefix="tests/data/chr" + str(chr),
            lanc_file="tests/data/chr" + str(chr) + ".lanc",
        )
        for chr in range(20, 23)
    ]
    return data


### ─────────────────────────────────────────────────────────────
### Input Validation
### ─────────────────────────────────────────────────────────────


def valid_inputs(P=3, K=5, C=1, H=4):
    """Utility for constructing minimal valid inputs."""
    N = 20
    Y = jnp.zeros((N, P))
    key = jax.random.key(8899134)
    X = jax.random.normal(key, shape=(N, C))
    train_mask = jnp.zeros((N, K))
    test_mask = jnp.zeros((N, K))
    h2_prior = jnp.linspace(0.1, 0.9, H)
    return Y, X, train_mask, test_mask, h2_prior


def test_step0_dataset_elements_type_error():
    """Check that bad datasets throws error"""
    Y, X, train, test, h2 = valid_inputs()
    with pytest.raises(TypeError, match="must be LancData"):
        step0([object()], Y, X, train, test, h2)


def test_step0_Y_dim_error(toy_data):
    """Check that 1D Y throws error"""
    _, X, train, test, h2 = valid_inputs()
    with pytest.raises(ValueError, match="Y must be 2D"):
        step0(toy_data, jnp.zeros(5), X, train, test, h2)


def test_step0_X_dim_error(toy_data):
    """Check that 1D X throws error"""
    Y, _, train, test, h2 = valid_inputs()
    with pytest.raises(ValueError, match="X must be 2D"):
        step0(toy_data, Y, jnp.zeros((10,)), train, test, h2)


def test_step0_X_matches_N_error(toy_data):
    """Check that N mis-match with X throws error"""
    Y, _, train, test, h2 = valid_inputs()
    wrong_X = jnp.zeros((5, 2))
    with pytest.raises(ValueError, match="must match Y.shape"):
        step0(toy_data, Y, wrong_X, train, test, h2)


def test_step0_mask_dim_error(toy_data):
    """Check that 1D train/test mask throws error"""
    Y, X, _, _, h2 = valid_inputs()
    with pytest.raises(ValueError, match="train_mask and test_mask must be 2D"):
        step0(toy_data, Y, X, jnp.zeros((10,)), jnp.zeros((10, 2)), h2)


def test_step0_mask_shape_mismatch_error(toy_data):
    """Check that train/test mask shape mis-match throws error"""
    Y, X, _, _, h2 = valid_inputs()
    with pytest.raises(ValueError, match="same shape"):
        step0(toy_data, Y, X, jnp.zeros((10, 2)), jnp.zeros((10, 3)), h2)


def test_step0_h2_prior_dim_error(toy_data):
    """Check that h2_prior with wrong dimension throws error"""
    Y, X, train, test, _ = valid_inputs()
    with pytest.raises(ValueError, match="h2_prior must be 1D"):
        step0(toy_data, Y, X, train, test, jnp.zeros((3, 2)))


def test_step0_h2_prior_domain_error(toy_data):
    """Check that h2_prior outside (0,1) throws error"""
    Y, X, train, test, _ = valid_inputs()
    with pytest.raises(ValueError, match="in the open interval"):
        step0(toy_data, Y, X, train, test, jnp.array([0.5, 0.0, 0.7]))


@pytest.mark.parametrize("B", [0, -5, 3.14, "foo"])
def test_step0_B_error(B, toy_data):
    """Check that bad B throws error"""
    Y, X, train, test, h2 = valid_inputs()
    with pytest.raises(ValueError, match="B must be a positive integer"):
        step0(toy_data, Y, X, train, test, h2, B=B)


def test_step0_variants_type_error(toy_data):
    """Check that bad variants type throws error"""
    Y, X, train, test, h2 = valid_inputs()
    with pytest.raises(TypeError, match="variants must be a list of strings"):
        step0(toy_data, Y, X, train, test, h2, variants=[1, 2, 3])


### ─────────────────────────────────────────────────────────────
### Other
### ─────────────────────────────────────────────────────────────


def test_step0_validation_happy_path(toy_data):
    """Check that valid data throws no type errors"""
    Y, X, train, test, h2 = valid_inputs()
    step0(toy_data, Y, X, train, test, h2, B=100)
