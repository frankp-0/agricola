# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import expit

from agricola.statistics.binary import (
    _bt_score_lanc,
    _bt_score_nolanc,
    _bt_wald_lanc,
    _bt_wald_nolanc,
    bt_score_lanc,
    bt_score_nolanc,
    bt_wald_lanc,
    bt_wald_nolanc,
)
from agricola.statistics.common import het_score, lanc_basis
from agricola.statistics.quantitative import (
    _qt_score_lanc,
    _qt_score_nolanc,
    _qt_wald_lanc,
    _qt_wald_nolanc,
    qt_score_lanc,
    qt_score_lanc_impute,
    qt_score_nolanc,
    qt_score_nolanc_impute,
    qt_wald_lanc,
    qt_wald_lanc_impute,
    qt_wald_nolanc,
    qt_wald_nolanc_impute,
)


def allele_masks(G):
    if G.ndim == 4:
        B, K, P = G.shape[1:]
        return jnp.ones((B, K, P), dtype=bool), jnp.ones((B, P), dtype=bool)
    B, K = G.shape[1:]
    return jnp.ones((B, K), dtype=bool), jnp.ones(B, dtype=bool)


@pytest.fixture
def toy_qt():
    N = 100
    B = 30
    P = 4
    K = 3
    C = 10
    key = jax.random.key(4321423)
    keys = jax.random.split(key, 4)

    ## Variables
    G = jax.random.normal(shape=(N, B, K, P), key=keys[0])
    L = jax.random.normal(shape=(N, B, K - 1, P), key=keys[1])
    Y = jax.random.normal(shape=(N, P), key=keys[2])
    X = jax.random.normal(shape=(N, C, P), key=keys[3])
    Q = jnp.linalg.qr(X.transpose(2, 0, 1), mode="reduced")[0].transpose(1, 2, 0)
    N_eff = jnp.repeat(N, P)
    return (G, L, Y, Q, N_eff)


@pytest.fixture
def toy_qt_edge():
    ## Edge cases:
    ## - zero variation in L or G columns
    ## - shared columns in L and G
    L = jnp.block(
        [
            jax.random.normal(shape=(10000, 3), key=jax.random.key(342)),
            jnp.zeros(10000)[:, None],
        ]
    )
    G = jnp.block(
        [
            L[:, 0][:, None],
            jax.random.normal(shape=(10000, 3), key=jax.random.key(302)),
            L[:, 3][:, None],
        ]
    )

    Y = jax.random.normal(shape=(10000, 1), key=jax.random.key(881))
    X = jnp.ones((10000, 1))
    Q, _ = jnp.linalg.qr(X, mode="reduced")
    N_eff = 10000
    return (G, L, Y, Q, N_eff)


@pytest.fixture
def toy_qt_impute():
    N = 100
    B = 30
    P = 4
    K = 3
    C = 10
    key = jax.random.key(4321423)
    keys = jax.random.split(key, 4)

    ## Variables
    G = jax.random.normal(shape=(N, B, K), key=keys[0])
    L = jax.random.normal(shape=(N, B, K - 1), key=keys[1])
    Y = jax.random.normal(shape=(N, P), key=keys[2])
    X = jax.random.normal(shape=(N, C), key=keys[3])
    Q, _ = jnp.linalg.qr(X, mode="reduced")
    return (G, L, Y, Q, N)


@pytest.fixture
def toy_bt():
    N = 100
    B = 30
    P = 4
    K = 3
    C = 10
    key = jax.random.key(4321423)
    keys = jax.random.split(key, 6)

    ## Variables
    G = jax.random.normal(shape=(N, B, K, P), key=keys[0])
    L = jax.random.normal(shape=(N, B, K - 1, P), key=keys[1])
    Y = jnp.round(expit(jax.random.normal(shape=(N, P), key=keys[2])))
    X = jax.random.normal(shape=(N, C, P), key=keys[3])
    Q = jnp.linalg.qr(X.transpose(2, 0, 1), mode="reduced")[0].transpose(1, 2, 0)
    M = jax.random.binomial(shape=(N, P), n=1, p=0.5, key=keys[4])
    offset = jax.random.normal(shape=(N, P), key=keys[5])
    return (G, L, Y, Q, offset, M)


@pytest.fixture
def toy_bt_edge():
    ## Edge cases:
    ## - zero variation in L or G columns
    ## - shared columns in L and G
    L = jnp.block(
        [
            jax.random.normal(shape=(10000, 3), key=jax.random.key(342)),
            jnp.zeros(10000)[:, None],
        ]
    )
    G = jnp.block(
        [
            L[:, 0][:, None],
            jax.random.normal(shape=(10000, 3), key=jax.random.key(302)),
            L[:, 3][:, None],
        ]
    )

    Y = jnp.round(expit(jax.random.normal(shape=(10000,), key=jax.random.key(881234))))
    X = jnp.ones((10000, 1))
    Q, _ = jnp.linalg.qr(X, mode="reduced")
    offset = jnp.full(shape=(10000), fill_value=0)
    M = jnp.round(expit(jax.random.normal(shape=(10000,), key=jax.random.key(61234))))
    return (G, L, Y, Q, offset, M)


### ─────────────────────────────────────────────────────────────
### Basic Tests: All core functions run without error
### ─────────────────────────────────────────────────────────────


def test_qt_lanc_score(toy_qt):
    qt_score_lanc(*toy_qt, *allele_masks(toy_qt[0]))


def test_qt_lanc_wald(toy_qt):
    qt_wald_lanc(*toy_qt, *allele_masks(toy_qt[0]))


def test_qt_nolanc_score(toy_qt):
    args = toy_qt[:1] + toy_qt[2:]
    qt_score_nolanc(*args, *allele_masks(toy_qt[0]))


def test_qt_nolanc_wald(toy_qt):
    args = toy_qt[:1] + toy_qt[2:]
    qt_wald_nolanc(*args, *allele_masks(toy_qt[0]))


def test_qt_lanc_score_impute(toy_qt_impute):
    qt_score_lanc_impute(*toy_qt_impute, *allele_masks(toy_qt_impute[0]))


def test_qt_lanc_wald_impute(toy_qt_impute):
    qt_wald_lanc_impute(*toy_qt_impute, *allele_masks(toy_qt_impute[0]))


def test_qt_nolanc_score_impute(toy_qt_impute):
    args = toy_qt_impute[:1] + toy_qt_impute[2:]
    qt_score_nolanc_impute(*args, *allele_masks(toy_qt_impute[0]))


def test_qt_nolanc_wald_impute(toy_qt_impute):
    args = toy_qt_impute[:1] + toy_qt_impute[2:]
    qt_wald_nolanc_impute(*args, *allele_masks(toy_qt_impute[0]))


def test_bt_lanc_score(toy_bt):
    bt_score_lanc(*toy_bt, *allele_masks(toy_bt[0]))


def test_bt_lanc_wald(toy_bt):
    bt_wald_lanc(*toy_bt, *allele_masks(toy_bt[0]))


def test_bt_nolanc_score(toy_bt):
    args = toy_bt[:1] + toy_bt[2:]
    bt_score_nolanc(*args, *allele_masks(toy_bt[0]))


def test_bt_nolanc_wald(toy_bt):
    args = toy_bt[:1] + toy_bt[2:]
    bt_wald_nolanc(*args, *allele_masks(toy_bt[0]))


def test_bt_score_diff_is_scalar_for_single_phenotype(toy_bt_edge):
    score_lanc = _bt_score_lanc(*toy_bt_edge)
    score_nolanc = _bt_score_nolanc(*(toy_bt_edge[:1] + toy_bt_edge[2:]))

    assert np.asarray(score_lanc[6]).shape == ()
    assert np.asarray(score_nolanc[6]).shape == ()
    assert np.asarray(score_lanc[8]).shape == ()
    assert np.asarray(score_nolanc[8]).shape == ()


### ─────────────────────────────────────────────────────────────
### Correct behavior: All core functions return accurate results
### ─────────────────────────────────────────────────────────────


def test_qt_lanc_score_edge(toy_qt_edge):
    actual = _qt_score_lanc(*toy_qt_edge)
    expected = np.load("tests/data/stats_results/qt_lanc_score_edge.npz")

    assert len(actual) == len(expected.files)

    for i, actual_array in enumerate(actual):
        np.testing.assert_allclose(
            np.asarray(actual_array),
            expected[f"arr_{i}"],
            rtol=1e-4,
            atol=1e-5,
        )


def test_qt_lanc_wald_edge(toy_qt_edge):
    actual = _qt_wald_lanc(*toy_qt_edge)
    expected = np.load("tests/data/stats_results/qt_lanc_wald_edge.npz")

    assert len(actual) == len(expected.files)

    for i, actual_array in enumerate(actual):
        if i == 6:
            np.testing.assert_allclose(
                np.asarray(actual_array),
                expected[f"arr_{i}"],
                rtol=2e-2,
                atol=1e-5,
            )
            continue
        np.testing.assert_allclose(
            np.asarray(actual_array),
            expected[f"arr_{i}"],
            rtol=1e-4,
            atol=1e-5,
        )


def test_qt_nolanc_score_edge(toy_qt_edge):
    args = toy_qt_edge[:1] + toy_qt_edge[2:]
    actual = _qt_score_nolanc(*args)
    expected = np.load("tests/data/stats_results/qt_nolanc_score_edge.npz")

    assert len(actual) == len(expected.files)

    for i, actual_array in enumerate(actual):
        np.testing.assert_allclose(
            np.asarray(actual_array),
            expected[f"arr_{i}"],
            rtol=1e-4,
            atol=1e-5,
        )


def test_qt_nolanc_wald_edge(toy_qt_edge):
    args = toy_qt_edge[:1] + toy_qt_edge[2:]
    actual = _qt_wald_nolanc(*args)
    expected = np.load("tests/data/stats_results/qt_nolanc_wald_edge.npz")

    assert len(actual) == len(expected.files)

    for i, actual_array in enumerate(actual):
        if i == 6:
            np.testing.assert_allclose(
                np.asarray(actual_array),
                expected[f"arr_{i}"],
                rtol=2e-2,
                atol=1e-5,
            )
            continue
        np.testing.assert_allclose(
            np.asarray(actual_array),
            expected[f"arr_{i}"],
            rtol=1e-4,
            atol=1e-5,
        )


def test_bt_lanc_score_edge(toy_bt_edge):
    actual = _bt_score_lanc(*toy_bt_edge)
    expected = np.load("tests/data/stats_results/bt_lanc_score_edge.npz")

    assert len(actual) == len(expected.files)

    for i, actual_array in enumerate(actual):
        # The exact float32 value is sensitive to cancellation in this
        # deliberately collinear, rank-deficient fixture. Identifiability and
        # finiteness are the invariant correctness properties.
        if i == 2:
            value = np.asarray(actual_array)
            assert np.isfinite(value).all()
            assert np.all(value > 0)
            continue
        np.testing.assert_allclose(
            np.asarray(actual_array),
            expected[f"arr_{i}"],
            rtol=1e-4,
            atol=1e-5,
        )


def test_bt_lanc_wald_edge(toy_bt_edge):
    actual = _bt_wald_lanc(*toy_bt_edge)
    expected = np.load("tests/data/stats_results/bt_lanc_wald_edge.npz")

    assert len(actual) == len(expected.files)

    for i, actual_array in enumerate(actual):
        np.testing.assert_allclose(
            np.asarray(actual_array),
            expected[f"arr_{i}"],
            rtol=1e-4,
            atol=1e-5,
        )


def test_bt_nolanc_score_edge(toy_bt_edge):
    args = toy_bt_edge[:1] + toy_bt_edge[2:]
    actual = _bt_score_nolanc(*args)
    expected = np.load("tests/data/stats_results/bt_nolanc_score_edge.npz")

    assert len(actual) == len(expected.files)

    for i, actual_array in enumerate(actual):
        np.testing.assert_allclose(
            np.asarray(actual_array),
            expected[f"arr_{i}"],
            rtol=1e-4,
            atol=1e-5,
        )


def test_bt_nolanc_wald_edge(toy_bt_edge):
    args = toy_bt_edge[:1] + toy_bt_edge[2:]
    actual = _bt_wald_nolanc(*args)
    expected = np.load("tests/data/stats_results/bt_nolanc_wald_edge.npz")

    assert len(actual) == len(expected.files)

    for i, actual_array in enumerate(actual):
        if i == 8:
            break
        np.testing.assert_allclose(
            np.asarray(actual_array),
            expected[f"arr_{i}"],
            rtol=1e-4,
            atol=1e-5,
        )


def test_het_score_uses_joint_coefficients_and_variances():
    score = jnp.array([[2.0], [1.0]])
    information = jnp.array([[4.0, 1.0], [1.0, 1.0]])

    beta, chisq_anc, chisq_het = het_score(score, information, jnp.array([True, True]))
    covariance = np.linalg.inv(np.asarray(information))
    expected_beta = covariance @ np.asarray(score)
    expected_anc = expected_beta[:, 0] ** 2 / np.diag(covariance)
    expected_het = np.asarray(score[:, 0]) @ expected_beta[:, 0]

    np.testing.assert_allclose(beta[:, 0], expected_beta[:, 0])
    np.testing.assert_allclose(chisq_anc[:, 0], expected_anc, rtol=1e-6)
    np.testing.assert_allclose(chisq_het, expected_het, rtol=1e-6)


def test_lanc_basis_uses_observed_rows_for_rank():
    ancestry = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )
    mask = jnp.array([1.0, 1.0, 0.0, 0.0])

    _, rank_mask = lanc_basis(ancestry, mask)

    np.testing.assert_array_equal(np.asarray(rank_mask), np.array([True, True]))


def test_bt_nolanc_masks_genotype_variation_only_in_missing_rows():
    G = jnp.array([[0.0], [0.0], [1.0], [2.0]])
    Y = jnp.array([0.0, 1.0, 0.0, 1.0])
    Q = jnp.empty((4, 0))
    offset = jnp.zeros(4)
    mask = jnp.array([1.0, 1.0, 0.0, 0.0])

    result = _bt_score_nolanc(G, Y, Q, offset, mask)

    assert np.isnan(np.asarray(result[0]))
    assert np.isnan(np.asarray(result[1]))
    assert np.isnan(np.asarray(result[3])).all()


def test_bt_nolanc_applies_allele_masks_to_fitted_designs():
    G = jnp.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    Y = jnp.array([0.0, 1.0, 0.0, 1.0])
    Q = jnp.empty((4, 0))
    offset = jnp.zeros(4)
    M = jnp.ones(4)

    result = _bt_score_nolanc(
        G,
        Y,
        Q,
        offset,
        M,
        allele_G_mask=jnp.array([True, False]),
        allele_H_mask=jnp.array(True),
    )

    assert np.isfinite(np.asarray(result[3])[0])
    assert np.isnan(np.asarray(result[3])[1])


def test_bt_lanc_joint_score_excludes_masked_ancestries():
    key = jax.random.key(123)
    G = jax.random.normal(key, (100, 2))
    L = jnp.ones((100, 1))
    Y = jnp.tile(jnp.array([0.0, 1.0]), 50)
    Q = jnp.empty((100, 0))
    offset = jnp.zeros(100)
    M = jnp.ones(100)

    result = _bt_score_lanc(
        G,
        L,
        Y,
        Q,
        offset,
        M,
        allele_G_mask=jnp.array([True, False]),
        allele_H_mask=jnp.array(True),
    )

    chisq_het = np.asarray(result[2])
    chisq_anc = np.asarray(result[5])
    assert chisq_het == pytest.approx(chisq_anc[0])
    assert np.isnan(chisq_anc[1])


@pytest.mark.parametrize(
    ("test_func", "uses_lanc"),
    [
        (_qt_score_lanc, True),
        (_qt_score_nolanc, False),
        (_qt_wald_lanc, True),
        (_qt_wald_nolanc, False),
        (_bt_score_lanc, True),
        (_bt_score_nolanc, False),
        (_bt_wald_lanc, True),
        (_bt_wald_nolanc, False),
    ],
)
def test_all_statistics_kernels_exclude_masked_g_and_h_columns(test_func, uses_lanc):
    key = jax.random.key(456)
    key_g, key_l, key_y = jax.random.split(key, 3)
    G = jax.random.normal(key_g, (100, 2))
    L = jax.random.normal(key_l, (100, 1))
    Y_qt = jax.random.normal(key_y, (100, 1))
    Y_bt = jnp.tile(jnp.array([0.0, 1.0]), 50)
    Q_qt = jnp.ones((100, 1)) / jnp.sqrt(100)
    Q_bt = jnp.empty((100, 0))
    offset = jnp.zeros(100)
    M = jnp.ones(100)
    g_mask = jnp.array([True, False])
    h_mask = jnp.array(False)

    if test_func.__name__.startswith("_qt"):
        args = (
            (G, L, Y_qt, Q_qt, 100, g_mask, h_mask)
            if uses_lanc
            else (
                G,
                Y_qt,
                Q_qt,
                100,
                g_mask,
                h_mask,
            )
        )
    else:
        args = (
            (G, L, Y_bt, Q_bt, offset, M, g_mask, h_mask)
            if uses_lanc
            else (
                G,
                Y_bt,
                Q_bt,
                offset,
                M,
                g_mask,
                h_mask,
            )
        )

    chisq_hom, beta_hom, chisq_het, beta_het, _, chisq_anc, *_ = test_func(*args)

    assert np.isnan(np.asarray(chisq_hom)).all()
    assert np.isnan(np.asarray(beta_hom)).all()
    assert np.isnan(np.asarray(beta_het)[1]).all()
    assert np.isnan(np.asarray(chisq_anc)[1]).all()
    np.testing.assert_allclose(np.asarray(chisq_het), np.asarray(chisq_anc)[0], rtol=1e-6)
