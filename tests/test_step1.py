# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

import os
import pytest
import numpy as np
import jax.numpy as jnp
import jax
from lanctools import LancData
from jax.scipy.special import expit
from agricola._internal.level0 import level0
from agricola._internal.level1 import level1


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
    key0 = jax.random.key(30091324)
    key = jax.random.key(8899134)
    Y = jax.random.normal(key0, shape=(N, P))
    X = jax.random.normal(key, shape=(N, C))
    train_mask = jnp.ones((N, K))
    test_mask = jnp.ones((N, K))
    h2_prior = jnp.linspace(0.1, 0.9, H)
    return Y, X, train_mask, test_mask, h2_prior


def valid_inputs_1(P=3, K=5, C=1, H=4):
    """Utility for constructing minimal valid inputs."""
    pref = "tests/data/level0/"
    level0_files = {"20": f"{pref}20.npy", "21": f"{pref}21.npy", "22": f"{pref}22.npy"}
    Y, X, train_mask, test_mask, h2_prior = valid_inputs_0(P, C, H)
    phenotypes = [str(i) for i in range(Y.shape[1])]
    h2_prior = jnp.linspace(0.1, 0.9, H)
    return level0_files, Y, X, phenotypes, train_mask, test_mask, h2_prior


### ─────────────────────────────────────────────────────────────
### Level 0
### ─────────────────────────────────────────────────────────────


def test_step0_dataset_elements_type_error():
    """Check that bad datasets throws error"""
    Y, X, train_mask, test_mask, h2 = valid_inputs_0()
    with pytest.raises(TypeError, match="must be LancData"):
        level0([object()], Y, X, train_mask, test_mask, h2)  # pyright: ignore


def test_level0_Y_dim_error(
    toy_data,
):
    """Check that 1D Y throws error"""
    _, X, train_mask, test_mask, h2 = valid_inputs_0()
    with pytest.raises(ValueError, match="Y must be 2D"):
        level0(toy_data, jnp.zeros(5), X, train_mask, test_mask, h2)


def test_level0_X_dim_error(
    toy_data,
):
    """Check that 1D X throws error"""
    Y, _, train_mask, test_mask, h2 = valid_inputs_0()
    with pytest.raises(ValueError, match="X must be 2D"):
        level0(toy_data, Y, jnp.zeros((10,)), train_mask, test_mask, h2)


def test_level0_X_matches_N_error(
    toy_data,
):
    """Check that N mis-match with X throws error"""
    Y, _, train_mask, test_mask, h2 = valid_inputs_0()
    wrong_X = jnp.zeros((5, 2))
    with pytest.raises(ValueError, match="must match Y.shape"):
        level0(toy_data, Y, wrong_X, train_mask, test_mask, h2)


def test_level0_h2_prior_dim_error(
    toy_data,
):
    """Check that h2_prior with wrong dimension throws error"""
    Y, X, train_mask, test_mask, _ = valid_inputs_0()
    with pytest.raises(ValueError, match="h2_prior must be 1D"):
        level0(toy_data, Y, X, train_mask, test_mask, jnp.ones((3, 2)))


def test_level0_h2_prior_domain_error(
    toy_data,
):
    """Check that h2_prior outside (0,1) throws error"""
    Y, X, train_mask, test_mask, _ = valid_inputs_0()
    with pytest.raises(ValueError, match="in the open interval"):
        level0(toy_data, Y, X, train_mask, test_mask, jnp.array([0.5, 0.0, 0.7]))


@pytest.mark.parametrize("B", [0, -5, 3.14, "foo"])
def test_level0_B_error(
    B,
    toy_data,
):
    """Check that bad B throws error"""
    Y, X, train_mask, test_mask, h2 = valid_inputs_0()
    with pytest.raises(ValueError, match="B must be a positive integer"):
        level0(toy_data, Y, X, train_mask, test_mask, h2, B=B)


def test_level0_variants_type_error(
    toy_data,
):
    """Check that bad variants type throws error"""
    Y, X, train_mask, test_mask, h2 = valid_inputs_0()
    with pytest.raises(TypeError, match="variants must be a list of strings"):
        level0(toy_data, Y, X, train_mask, test_mask, h2, variants=[1, 2, 3])  # pyright: ignore


def test_level0_validation_happy_path(
    toy_data,
):
    """Check that valid data throws no type errors"""
    Y, X, train_mask, test_mask, h2 = valid_inputs_0()
    level0(toy_data, Y, X, train_mask, test_mask, h2, B=100)


### ─────────────────────────────────────────────────────────────
### Level 1
### ─────────────────────────────────────────────────────────────


def test_level1_Y_dim_error():
    """Check that 1D Y throws error"""
    level0_files, _, X, phenos, train, test, h2 = valid_inputs_1()
    with pytest.raises(ValueError, match="Y must be 2D"):
        level1(level0_files, jnp.zeros(5), X, phenos, train, test, h2, "qt")

    with pytest.raises(ValueError, match="Y must be 2D"):
        level1(level0_files, jnp.zeros(5), X, phenos, train, test, h2, "bt")


def test_level1_X_dim_error():
    """Check that 1D X throws error"""
    level0_files, Y, _, phenos, train, test, h2 = valid_inputs_1()
    with pytest.raises(ValueError, match="X must be 2D"):
        level1(level0_files, Y, jnp.zeros((10,)), phenos, train, test, h2, "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="X must be 2D"):
        level1(level0_files, Y, jnp.zeros((10,)), phenos, train, test, h2, "bt")


def test_level1_X_matches_N_error():
    """Check that N mis-match with X throws error"""
    level0_files, Y, _, phenos, train, test, h2 = valid_inputs_1()
    with pytest.raises(ValueError, match="must match Y.shape"):
        level1(level0_files, Y, jnp.zeros((5, 2)), phenos, train, test, h2, "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="must match Y.shape"):
        level1(level0_files, Y, jnp.zeros((5, 2)), phenos, train, test, h2, "bt")


def test_level1_mask_dim_error():
    """Check that 1D train/test mask throws error"""
    level0_files, Y, X, phenos, _, _, h2 = valid_inputs_1()
    with pytest.raises(ValueError, match="train_mask and test_mask must be 2D"):
        level1(level0_files, Y, X, phenos, jnp.ones((10,)), jnp.ones((10, 2)), h2, "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="train_mask and test_mask must be 2D"):
        level1(level0_files, Y, X, phenos, jnp.ones((10,)), jnp.ones((10, 2)), h2, "bt")


def test_level1_mask_shape_mismatch_error():
    """Check that train/test mask shape mis-match throws error"""
    level0_files, Y, X, phenos, _, _, h2 = valid_inputs_1()
    with pytest.raises(ValueError, match="same shape"):
        level1(
            level0_files, Y, X, phenos, jnp.ones((10, 2)), jnp.ones((10, 3)), h2, "qt"
        )

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="same shape"):
        level1(
            level0_files, Y, X, phenos, jnp.ones((10, 2)), jnp.ones((10, 3)), h2, "bt"
        )


def test_level1_h2_prior_dim_error():
    """Check that h2_prior with wrong dimension throws error"""
    level0_files, Y, X, phenos, train, test, _ = valid_inputs_1()
    with pytest.raises(ValueError, match="h2_prior must be 1D"):
        level1(level0_files, Y, X, phenos, train, test, jnp.ones((3, 2)), "qt")

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="h2_prior must be 1D"):
        level1(level0_files, Y, X, phenos, train, test, jnp.ones((3, 2)), "bt")


def test_level1_h2_prior_domain_error():
    """Check that h2_prior outside (0,1) throws error"""
    level0_files, Y, X, phenos, train, test, _ = valid_inputs_1()
    with pytest.raises(ValueError, match="in the open interval"):
        level1(
            level0_files, Y, X, phenos, train, test, jnp.array([0.5, 0.0, 0.7]), "qt"
        )

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="in the open interval"):
        level1(
            level0_files, Y, X, phenos, train, test, jnp.array([0.5, 0.0, 0.7]), "bt"
        )


def test_level1_qt_valid_input():
    """Check that valid data throws no type errors"""
    level0_files, Y, X, phenos, train, test, h2 = valid_inputs_1()
    level1(level0_files, Y, X, phenos, train, test, h2, "qt")


def test_level1_bt_valid_input():
    """Check that valid data throws no type errors"""
    ## cross-validation
    level0_files, Y, X, phenos, train, test, h2 = valid_inputs_1()
    Y = jnp.round(expit(Y))
    level1(level0_files, Y, X, phenos, train, test, h2, "bt", False)

    ## loco
    level0_files, Y, X, phenos, train, test, h2 = valid_inputs_1()
    level1(level0_files, Y, X, phenos, train, test, h2, "bt", True)
