# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax.scipy.special import expit
from lanctools import LancData

from agricola import step2
from agricola.pipeline.step2 import _prep_block


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


def test_step2_phenotype_and_prediction_validation(tmp_path, toy_data):
    Y, X, step1_predictions = valid_inputs()
    outdir = tmp_path / "result"

    with pytest.raises(ValueError, match="phenotype has length"):
        step2(toy_data, Y, X, step1_predictions, outdir, ["0"], "qt")

    missing = dict(step1_predictions)
    missing["20"] = missing["20"].drop(columns=["1"])
    with pytest.raises(KeyError):
        step2(toy_data, Y, X, missing, outdir, ["0", "1", "2"], "qt")


def test_step2_enum_batch_and_variant_validation(tmp_path, toy_data):
    Y, X, step1_predictions = valid_inputs()
    phenotypes = [str(i) for i in range(3)]
    outdir = tmp_path / "result"

    with pytest.raises(ValueError):
        step2(toy_data, Y, X, step1_predictions, outdir, phenotypes, "invalid")
    with pytest.raises(ValueError):
        step2(toy_data, Y, X, step1_predictions, outdir, phenotypes, "qt", test_type="invalid")
    with pytest.raises(ValueError, match="B must be a positive integer"):
        step2(toy_data, Y, X, step1_predictions, outdir, phenotypes, "qt", B=0)
    with pytest.raises(ValueError, match=r"p_het_threshold must be in \[0, 1\]"):
        step2(
            toy_data,
            Y,
            X,
            step1_predictions,
            outdir,
            phenotypes,
            "qt",
            p_het_threshold=-0.1,
        )
    with pytest.raises(TypeError, match="variants must be a list of strings"):
        step2(toy_data, Y, X, step1_predictions, outdir, phenotypes, "qt", variants=[1])
    with pytest.raises(ValueError, match="min_ac_g and min_ac_h must be non-negative"):
        step2(
            toy_data,
            Y,
            X,
            step1_predictions,
            outdir,
            phenotypes,
            "qt",
            min_ac_g=-1,
        )

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


def test_step2_zero_p_het_threshold(tmp_path, toy_data):
    """Check that a zero heterogeneity threshold suppresses difference tests."""
    Y, X, step1_predictions = valid_inputs()
    phenotypes = [str(i) for i in range(3)]
    outdir = tmp_path / "result"

    step2(
        toy_data,
        Y,
        X,
        step1_predictions,
        outdir,
        phenotypes,
        "qt",
        p_het_threshold=0,
    )

    output_files = list(outdir.glob("*/*.parquet"))
    assert output_files
    result = pd.read_parquet(output_files[0])
    assert result["LOG10P_HET_VS_HOM"].isna().all()


def test_step2_applies_independent_g_and_h_allele_thresholds(tmp_path, toy_data):
    Y, X, step1_predictions = valid_inputs()
    phenotypes = [str(i) for i in range(3)]
    outdir = tmp_path / "result"

    step2(
        toy_data,
        Y,
        X,
        step1_predictions,
        outdir,
        phenotypes,
        "qt",
        min_ac_g=0,
        min_ac_h=10_000,
    )

    output_files = list(outdir.glob("*/*.parquet"))
    assert output_files
    result = pd.read_parquet(output_files[0])
    assert not result.empty
    assert result["BETA_HOM"].isna().all()
    assert result["LOG10P_HOM"].isna().all()


def test_prep_block_filters_on_minor_allele_count():
    """Major-allele counts must not bypass G or H allele-count filters."""
    G = jnp.array([[[1, 1]], [[1, 1]], [[1, 1]], [[0, 0]]])
    L = jnp.ones_like(G)
    M = jnp.ones((4, 1))

    *_, allele_G_mask, allele_H_mask = _prep_block(G, L, M, 2, 3)

    # The coded allele counts are 3 (per ancestry) and 6 (total), but their
    # corresponding minor allele counts are 1 and 2.
    np.testing.assert_array_equal(allele_G_mask, [[[False], [False]]])
    np.testing.assert_array_equal(allele_H_mask, [[False]])


def test_step2_skips_diff_for_unconverged_binary_first_pass(tmp_path, toy_data, monkeypatch):
    """Check that failed first-pass binary fits do not run difference tests."""
    step2_module = import_module("agricola.pipeline.step2")
    original_no_diff = step2_module.bt_score_lanc_no_diff

    def unconverged_first_pass(*args):
        result = original_no_diff(*args)
        return (*result[:-1], jnp.zeros_like(result[-1], dtype=bool))

    def unexpected_diff_test(*args):
        raise AssertionError("difference test ran after first-pass non-convergence")

    monkeypatch.setattr(step2_module, "bt_score_lanc_no_diff", unconverged_first_pass)
    monkeypatch.setitem(
        step2_module._BT_FUNCTIONS,
        (step2_module.TestType.SCORE, True),
        unexpected_diff_test,
    )

    Y, X, step1_predictions = valid_inputs()
    phenotypes = [str(i) for i in range(3)]
    outdir = tmp_path / "result"
    step2(
        toy_data,
        jnp.round(expit(Y)),
        X,
        step1_predictions,
        outdir,
        phenotypes,
        "bt",
        p_het_threshold=0.999999,
    )

    output_files = list(outdir.glob("*/*.parquet"))
    assert output_files
    result = pd.read_parquet(output_files[0])
    assert not result["CONVERGED"].any()
    assert result["LOG10P_HET_VS_HOM"].isna().all()


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
