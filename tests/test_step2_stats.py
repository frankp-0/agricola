# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

import pytest
import jax
import jax.numpy as jnp
from jax.scipy.special import expit
from agricola._internal.step2_stats import (
    qt_score_lanc,
    qt_score_lanc_impute,
    qt_score_nolanc,
    qt_score_nolanc_impute,
    bt_score_lanc,
    bt_score_nolanc,
    qt_wald_lanc,
    qt_wald_lanc_impute,
    qt_wald_nolanc,
    qt_wald_nolanc_impute,
    bt_wald_lanc,
    bt_wald_nolanc,
)


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
    O = jax.random.normal(shape=(N, P), key=keys[5])
    N_eff = jnp.repeat(N, P)
    return (G, L, Y, Q, O, M, N_eff)


### ─────────────────────────────────────────────────────────────
### Basic Tests: All core functions run without error
### ─────────────────────────────────────────────────────────────


def test_qt_lanc_score(toy_qt):
    qt_score_lanc(*toy_qt)


def test_qt_lanc_wald(toy_qt):
    qt_wald_lanc(*toy_qt)


def test_qt_nolanc_score(toy_qt):
    args = toy_qt[:1] + toy_qt[2:]
    qt_score_nolanc(*args)


def test_qt_nolanc_wald(toy_qt):
    args = toy_qt[:1] + toy_qt[2:]
    qt_wald_nolanc(*args)


def test_qt_lanc_score_impute(toy_qt_impute):
    qt_score_lanc_impute(*toy_qt_impute)


def test_qt_lanc_wald_impute(toy_qt_impute):
    qt_wald_lanc_impute(*toy_qt_impute)


def test_qt_nolanc_score_impute(toy_qt_impute):
    args = toy_qt_impute[:1] + toy_qt_impute[2:]
    qt_score_nolanc_impute(*args)


def test_qt_nolanc_wald_impute(toy_qt_impute):
    args = toy_qt_impute[:1] + toy_qt_impute[2:]
    qt_wald_nolanc_impute(*args)


def test_bt_lanc_score(toy_bt):
    bt_score_lanc(*toy_bt)


def test_bt_lanc_wald(toy_bt):
    bt_wald_lanc(*toy_bt)


def test_bt_nolanc_score(toy_bt):
    args = toy_bt[:1] + toy_bt[2:]
    bt_score_nolanc(*args)


def test_bt_nolanc_wald(toy_bt):
    args = toy_bt[:1] + toy_bt[2:]
    bt_wald_nolanc(*args)
