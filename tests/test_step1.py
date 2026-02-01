# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

import pytest
import numpy as np
import jax.numpy as jnp
import jax
from lanctools import LancData
from jax.scipy.special import expit
from lagga.step1.level0 import level0
from lagga.step1.level1 import level1


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


def valid_inputs_0(P=3, K=5, C=1, H=4):
    """Utility for constructing minimal valid inputs for step 0."""
    N = 20
    Y = jnp.zeros((N, P))
    key = jax.random.key(8899134)
    X = jax.random.normal(key, shape=(N, C))
    train_mask = jnp.ones((N, K))
    test_mask = jnp.ones((N, K))
    h2_prior = jnp.linspace(0.1, 0.9, H)
    return Y, X, train_mask, test_mask, h2_prior


def valid_inputs_1(R=16, P=3, K=5, C=1, H=4):
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


### ─────────────────────────────────────────────────────────────
### Level 0
### ─────────────────────────────────────────────────────────────


def test_step0_dataset_elements_type_error():
    """Check that bad datasets throws error"""
    Y, X, train, test, h2 = valid_inputs_0()
    with pytest.raises(TypeError, match="must be LancData"):
        level0([object()], Y, X, train, test, h2)  # pyright: ignore


def test_level0_Y_dim_error(toy_data):
    """Check that 1D Y throws error"""
    _, X, train, test, h2 = valid_inputs_0()
    with pytest.raises(ValueError, match="Y must be 2D"):
        level0(toy_data, jnp.zeros(5), X, train, test, h2)


def test_level0_X_dim_error(toy_data):
    """Check that 1D X throws error"""
    Y, _, train, test, h2 = valid_inputs_0()
    with pytest.raises(ValueError, match="X must be 2D"):
        level0(toy_data, Y, jnp.zeros((10,)), train, test, h2)


def test_level0_X_matches_N_error(toy_data):
    """Check that N mis-match with X throws error"""
    Y, _, train, test, h2 = valid_inputs_0()
    wrong_X = jnp.zeros((5, 2))
    with pytest.raises(ValueError, match="must match Y.shape"):
        level0(toy_data, Y, wrong_X, train, test, h2)


def test_level0_mask_dim_error(toy_data):
    """Check that 1D train/test mask throws error"""
    Y, X, _, _, h2 = valid_inputs_0()
    with pytest.raises(ValueError, match="train_mask and test_mask must be 2D"):
        level0(toy_data, Y, X, jnp.ones((10,)), jnp.ones((10, 2)), h2)


def test_level0_mask_shape_mismatch_error(toy_data):
    """Check that train/test mask shape mis-match throws error"""
    Y, X, _, _, h2 = valid_inputs_0()
    with pytest.raises(ValueError, match="same shape"):
        level0(toy_data, Y, X, jnp.ones((10, 2)), jnp.ones((10, 3)), h2)


def test_level0_h2_prior_dim_error(toy_data):
    """Check that h2_prior with wrong dimension throws error"""
    Y, X, train, test, _ = valid_inputs_0()
    with pytest.raises(ValueError, match="h2_prior must be 1D"):
        level0(toy_data, Y, X, train, test, jnp.ones((3, 2)))


def test_level0_h2_prior_domain_error(toy_data):
    """Check that h2_prior outside (0,1) throws error"""
    Y, X, train, test, _ = valid_inputs_0()
    with pytest.raises(ValueError, match="in the open interval"):
        level0(toy_data, Y, X, train, test, jnp.array([0.5, 0.0, 0.7]))


@pytest.mark.parametrize("B", [0, -5, 3.14, "foo"])
def test_level0_B_error(B, toy_data):
    """Check that bad B throws error"""
    Y, X, train, test, h2 = valid_inputs_0()
    with pytest.raises(ValueError, match="B must be a positive integer"):
        level0(toy_data, Y, X, train, test, h2, B=B)


def test_level0_variants_type_error(toy_data):
    """Check that bad variants type throws error"""
    Y, X, train, test, h2 = valid_inputs_0()
    with pytest.raises(TypeError, match="variants must be a list of strings"):
        level0(toy_data, Y, X, train, test, h2, variants=[1, 2, 3])  # pyright: ignore


def test_level0_validation_happy_path(toy_data):
    """Check that valid data throws no type errors"""
    Y, X, train, test, h2 = valid_inputs_0()
    level0(toy_data, Y, X, train, test, h2, B=100)


### ─────────────────────────────────────────────────────────────
### Level 1
### ─────────────────────────────────────────────────────────────


def test_level1_Z_dim_error():
    """Check that bad Z dimensions throws error"""
    Z, Y, X, train, test, h2 = valid_inputs_1()
    Z["0"] = np.ones(shape=(1, 6, 4))

    with pytest.raises(ValueError, match="All Z arrays must have same"):
        level1(Z, Y, X, train, test, h2, "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError):
        level1(Z, Y, X, train, test, h2, "bt")


def test_level1_Y_dim_error():
    """Check that 1D Y throws error"""
    Z, _, X, train, test, h2 = valid_inputs_1()
    with pytest.raises(ValueError, match="Y must be 2D"):
        level1(Z, jnp.zeros(5), X, train, test, h2, "qt")

    with pytest.raises(ValueError, match="Y must be 2D"):
        level1(Z, jnp.zeros(5), X, train, test, h2, "bt")


def test_level1_X_dim_error():
    """Check that 1D X throws error"""
    Z, Y, _, train, test, h2 = valid_inputs_1()
    with pytest.raises(ValueError, match="X must be 2D"):
        level1(Z, Y, jnp.zeros((10,)), train, test, h2, "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="X must be 2D"):
        level1(Z, Y, jnp.zeros((10,)), train, test, h2, "bt")


def test_level1_X_matches_N_error():
    """Check that N mis-match with X throws error"""
    Z, Y, _, train, test, h2 = valid_inputs_1()
    with pytest.raises(ValueError, match="must match Y.shape"):
        level1(Z, Y, jnp.zeros((5, 2)), train, test, h2, "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="must match Y.shape"):
        level1(Z, Y, jnp.zeros((5, 2)), train, test, h2, "bt")


def test_level1_mask_dim_error():
    """Check that 1D train/test mask throws error"""
    Z, Y, X, _, _, h2 = valid_inputs_1()
    with pytest.raises(ValueError, match="train_mask and test_mask must be 2D"):
        level1(Z, Y, X, jnp.ones((10,)), jnp.ones((10, 2)), h2, "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="train_mask and test_mask must be 2D"):
        level1(Z, Y, X, jnp.ones((10,)), jnp.ones((10, 2)), h2, "bt")


def test_level1_mask_shape_mismatch_error():
    """Check that train/test mask shape mis-match throws error"""
    Z, Y, X, _, _, h2 = valid_inputs_1()
    with pytest.raises(ValueError, match="same shape"):
        level1(Z, Y, X, jnp.ones((10, 2)), jnp.ones((10, 3)), h2, "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="same shape"):
        level1(Z, Y, X, jnp.ones((10, 2)), jnp.ones((10, 3)), h2, "bt")


def test_level1_h2_prior_dim_error():
    """Check that h2_prior with wrong dimension throws error"""
    Z, Y, X, train, test, _ = valid_inputs_1()
    with pytest.raises(ValueError, match="h2_prior must be 1D"):
        level1(Z, Y, X, train, test, jnp.ones((3, 2)), "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="h2_prior must be 1D"):
        level1(Z, Y, X, train, test, jnp.ones((3, 2)), "bt")


def test_level1_h2_prior_domain_error():
    """Check that h2_prior outside (0,1) throws error"""
    Z, Y, X, train, test, _ = valid_inputs_1()
    with pytest.raises(ValueError, match="in the open interval"):
        level1(Z, Y, X, train, test, jnp.array([0.5, 0.0, 0.7]), "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="in the open interval"):
        level1(Z, Y, X, train, test, jnp.array([0.5, 0.0, 0.7]), "bt")


def test_level1_qt_valid_input():
    """Check that valid data throws no type errors"""
    Z, Y, X, train, test, h2 = valid_inputs_1()
    level1(Z, Y, X, train, test, h2, "qt")


def test_level1_bt_valid_input():
    """Check that valid data throws no type errors"""
    ## cross-validation
    Z, Y, X, train, test, h2 = valid_inputs_1()
    Y = jnp.round(expit(Y))
    level1(Z, Y, X, train, test, h2, "bt", False)

    ## loco
    Z, Y, X, train, test, h2 = valid_inputs_1()
    level1(Z, Y, X, None, None, h2, "bt", True)
