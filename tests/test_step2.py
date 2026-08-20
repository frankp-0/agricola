# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax.scipy.special import expit
from lanctools import LancData

from agricola import step2


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


def valid_inputs(P=3, C=1):
    """Utility for constructing minimal valid inputs."""
    key0 = jax.random.key(8899134)
    key1 = jax.random.key(471464)
    key2 = jax.random.key(7314120)
    N = 20
    step1_predictions = {}
    for i in range(20, 23):
        arr = jax.random.normal(jax.random.split(key0, 3)[i - 21], shape=(N, P))
        df = pd.DataFrame(arr, columns=pd.Index([str(i) for i in range(3)]))
        step1_predictions[str(i)] = df

    Y = jax.random.normal(key1, shape=(N, P))
    X = jax.random.normal(key2, shape=(N, C))
    return Y, X, step1_predictions


def test_step2_dataset_elements_type_error(tmp_path):
    """Check that bad datasets throws error"""
    Y, X, step1_predictions = valid_inputs()
    phenotypes = [str(i) for i in range(3)]
    outdir = tmp_path / "result"
    with pytest.raises(TypeError, match="must be LancData"):
        step2([object()], Y, X, step1_predictions, outdir, phenotypes, "qt")  # pyright: ignore

    Y = jnp.round(expit(Y))
    with pytest.raises(TypeError, match="must be LancData"):
        step2([object()], Y, X, step1_predictions, outdir, phenotypes, "bt")  # pyright: ignore


def test_step2_pred_dim_error(tmp_path):
    """Check that bad Z dimensions throws error"""
    Y, X, step1_predictions = valid_inputs()
    step1_predictions["21"] = np.ones(shape=(1, 6, 4))
    phenotypes = [i for i in range(3)]
    outdir = tmp_path / "result"
    with pytest.raises(TypeError, match="must be LancData"):
        step2([object()], Y, X, step1_predictions, outdir, phenotypes, "qt")  # pyright: ignore

    Y = jnp.round(expit(Y))
    with pytest.raises(TypeError, match="must be LancData"):
        step2([object()], Y, X, step1_predictions, outdir, phenotypes, "bt")  # pyright: ignore


def test_step2_Y_dim_error(tmp_path, toy_data):
    """Check that 1D Y throws error"""
    _, X, step1_predictions = valid_inputs()
    phenotypes = [str(i) for i in range(3)]
    outdir = tmp_path / "result"
    with pytest.raises(ValueError, match="Y must be 2D"):
        step2(toy_data, jnp.zeros(5), X, step1_predictions, outdir, phenotypes, "qt")  # pyright: ignore

    with pytest.raises(ValueError, match="Y must be 2D"):
        step2(toy_data, jnp.zeros(5), X, step1_predictions, outdir, phenotypes, "bt")  # pyright: ignore


def test_step2_X_dim_error(tmp_path, toy_data):
    """Check that 1D X throws error"""
    Y, _, step1_predictions = valid_inputs()
    phenotypes = [str(i) for i in range(3)]
    outdir = tmp_path / "result"
    with pytest.raises(ValueError, match="X must be 2D"):
        step2(
            toy_data,
            Y,
            jnp.zeros(10),
            step1_predictions,
            outdir,
            phenotypes,
            "qt",
        )  # pyright: ignore

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="X must be 2D"):
        step2(
            toy_data,
            Y,
            jnp.zeros(10),
            step1_predictions,
            outdir,
            phenotypes,
            "bt",
        )  # pyright: ignore


def test_step2_X_matches_N_error(tmp_path, toy_data):
    """Check that N mis-match with X throws error"""
    Y, _, step1_predictions = valid_inputs()
    phenotypes = [str(i) for i in range(3)]
    outdir = tmp_path / "result"
    with pytest.raises(ValueError, match="must match Y\\.shape"):
        step2(
            toy_data,
            Y,
            jnp.zeros((5, 2)),
            step1_predictions,
            outdir,
            phenotypes,
            "qt",
        )  # pyright: ignore

    Y = jnp.round(expit(Y))
    with pytest.raises(ValueError, match="must match Y\\.shape"):
        step2(
            toy_data,
            Y,
            jnp.zeros((5, 2)),
            step1_predictions,
            outdir,
            phenotypes,
            "bt",
        )  # pyright: ignore


### ─────────────────────────────────────────────────────────────
### Other
### ─────────────────────────────────────────────────────────────


def test_step2_valid_input_qt(tmp_path, toy_data):
    """Check that valid data throws no type errors"""
    Y, X, step1_predictions = valid_inputs()
    phenotypes = [str(i) for i in range(3)]
    outdir = tmp_path / "result"
    step2(toy_data, Y, X, step1_predictions, outdir, phenotypes, "qt")
    step2(
        toy_data,
        Y,
        X,
        step1_predictions,
        outdir,
        phenotypes,
        "qt",
        adjust_lanc=True,
    )


def test_step2_valid_input_bt(tmp_path, toy_data):
    """Check that valid data throws no type errors"""
    Y, X, step1_predictions = valid_inputs()
    Y = jnp.round(expit(Y))
    phenotypes = [str(i) for i in range(3)]
    outdir = tmp_path / "result"
    step2(toy_data, Y, X, step1_predictions, outdir, phenotypes, "bt")
    step2(
        toy_data,
        Y,
        X,
        step1_predictions,
        outdir,
        phenotypes,
        "bt",
        adjust_lanc=True,
    )
