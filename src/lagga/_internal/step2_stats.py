# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Single variant associations.

This module contains functions for calculating the tests statistics used in lagga step 2
"""

import jax
import jax.numpy as jnp
from jax.numpy.linalg import inv, matrix_rank, qr, solve
from jaxtyping import Array
from jax.scipy.special import expit
from .models import logistic_ridge

### ─────────────────────────────────────────────────────────────
### Helpers
### ─────────────────────────────────────────────────────────────


def wls_qr_res(X: Array, Q: Array, W: Array) -> Array:
    Xhat = Q @ (Q.T @ (X * W))
    Xhat = jnp.where(W == 0, 0, Xhat / W)
    return X - Xhat


def ols_qr_res(X: Array, Q: Array) -> Array:
    return X - Q @ (Q.T @ X)


def ols(X: Array, Y: Array) -> tuple[Array, Array, Array]:
    N, K = X.shape
    alpha = N * 1e-6
    XtX = X.T @ X
    beta = solve(XtX + alpha * jnp.identity(K), X.T @ Y)
    resid = Y - X @ beta
    sse = jnp.sum(resid**2, axis=0)
    return beta, sse, XtX


def logit(X: Array, Y: Array, O: Array, W: Array, alpha: float) -> tuple[Array, Array]:
    beta = logistic_ridge(X, Y, O, W, alpha, 10)
    mu = expit(X @ beta + O)
    w = mu * (1 - mu) * W
    XW = X * jnp.sqrt(w[:, None])
    XtX_W = XW.T @ XW
    return beta, XtX_W


### ─────────────────────────────────────────────────────────────
### Kernels (test for a single variant and phenotype)
### ─────────────────────────────────────────────────────────────


def _qt_score_lanc_core(
    G: Array, L: Array, Y: Array, Q: Array, M: Array, N_eff: int
) -> tuple[Array, Array, Array, Array]:
    K = G.shape[1]
    alpha = N_eff * 1e-6

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = ols_qr_res(G, Q) * M[:, None]
    L = ols_qr_res(L, Q) * M[:, None]
    H = ols_qr_res(H, Q) * M

    ## Fit null model: Y ~ L
    QL, _ = qr(L)
    r_L = ols_qr_res(Y, QL)
    mse_null = jnp.sum(r_L**2) / (N_eff - (K - 1))

    ## Residualize G, H by L
    G_res = ols_qr_res(G, QL)
    H_res = ols_qr_res(H, QL)

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ r_L
    GtG = G_res.T @ G_res
    GtG_inv = inv(GtG + alpha * jnp.identity(K))
    chisq_het = U.T @ GtG_inv @ U / mse_null
    beta_anc = U * jnp.diagonal(GtG_inv)
    df_het = matrix_rank(GtG)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ r_L
    HtH = jnp.sum(H_res**2)
    chisq_hom = (UH**2) / (HtH + alpha) / mse_null

    return chisq_hom, chisq_het, beta_anc, df_het


def _bt_score_lanc_core(
    G: Array,
    L: Array,
    Y: Array,
    Q_w: Array,
    W_sqrt: Array,
    O: Array,
    M: Array,
    N_eff: int,
) -> tuple[Array, Array, Array, Array]:
    K = G.shape[1]
    alpha = N_eff * 1e-6

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    H = wls_qr_res(H[:, None], Q_w, W_sqrt[:, None])[:, 0]
    G = wls_qr_res(G, Q_w, W_sqrt[:, None])
    L = wls_qr_res(L, Q_w, W_sqrt[:, None])

    ## Fit L + offset null model (logistic)
    beta_L = logistic_ridge(L, Y, O, M, 0.0, 10)
    mu = expit(L @ beta_L + O)
    R = Y - mu
    W_L_sqrt = jnp.sqrt(mu * (1.0 - mu)) * M

    ## Residualize G, H by L
    QL, _ = qr(L * W_L_sqrt[:, None])
    G_res = wls_qr_res(G, QL, W_L_sqrt[:, None])
    H_res = wls_qr_res(H[:, None], QL, W_L_sqrt[:, None])[:, 0]

    ## Score test for anc-deconvoluted genotypes
    U = G.T @ (R * M)
    GW = G_res * W_L_sqrt[:, None]
    GtG = GW.T @ GW
    GtG_inv = inv(GtG + alpha * jnp.identity(K))
    chisq_het = U.T @ GtG_inv @ U
    beta_anc = U * jnp.diagonal(GtG_inv)
    df_het = matrix_rank(GtG)

    ## Score test for genotypes
    UH = jnp.sum(H * R * M, axis=0)
    HW = H_res * W_L_sqrt
    HtH = jnp.sum(HW**2, axis=0)
    chisq_hom = (UH**2) / (HtH + alpha)

    return chisq_hom, chisq_het, beta_anc, df_het


def _qt_score_nolanc_core(
    G: Array, Y: Array, Q: Array, M: Array, N_eff: int
) -> tuple[Array, Array, Array, Array]:
    K = G.shape[1]
    alpha = N_eff * 1e-6

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = ols_qr_res(G, Q) * M[:, None]
    H = ols_qr_res(H, Q) * M
    mse_null = jnp.sum(Y**2, axis=0) / N_eff

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ Y
    GtG = G.T @ G
    GtG_inv = inv(G.T @ G + alpha * jnp.identity(K))
    chisq_het = U.T @ GtG_inv @ U / mse_null
    beta_anc = U * jnp.diagonal(GtG_inv)
    df_het = matrix_rank(GtG)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ Y
    HtH = jnp.sum(H**2)
    chisq_hom = (UH**2) / (HtH + alpha) / mse_null

    return chisq_hom, chisq_het, beta_anc, df_het


def _bt_score_nolanc_core(
    G: Array,
    Y: Array,
    Q_w: Array,
    W_sqrt: Array,
    O: Array,
    M: Array,
    N_eff: int,
) -> tuple[Array, Array, Array, Array]:
    K = G.shape[1]
    alpha = N_eff * 1e-6

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    H = wls_qr_res(H[:, None], Q_w, W_sqrt[:, None])[:, 0]
    G = wls_qr_res(G, Q_w, W_sqrt[:, None])

    ## Null model
    mu = expit(O)
    W = mu * (1 - mu) * M
    R = Y - mu

    ## Score test for anc-deconvoluted genotypes
    GW = G * jnp.sqrt(W[:, None])
    U = G.T @ (R * M)
    GtG = GW.T @ GW
    GtG_inv = inv(GtG + alpha * jnp.identity(K))
    chisq_het = U.T @ GtG_inv @ U
    beta_anc = U * jnp.diagonal(GtG_inv)
    df_het = matrix_rank(GtG)

    ## Score test for genotypes
    HW = H * jnp.sqrt(W)
    UH = jnp.sum(H * R * M, axis=0)
    HtH = jnp.sum(HW**2, axis=0)
    chisq_hom = (UH**2) / (HtH + alpha)

    return chisq_hom, chisq_het, beta_anc, df_het


def _qt_wald_lanc_core(
    G: Array, L: Array, Y: Array, Q: Array, M: Array, N_eff: int
) -> tuple[Array, Array, Array, Array]:
    K = G.shape[1]
    alpha = N_eff * 1e-6

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = ols_qr_res(G, Q) * M[:, None]
    L = ols_qr_res(L, Q) * M[:, None]
    H = ols_qr_res(H, Q) * M

    ## Wald test for anc-deconvoluted genotypes
    beta_G, sse_G, GtG = ols(jnp.concatenate([G, L], axis=1), Y)
    GtG_inv = inv(GtG + alpha * jnp.identity(2 * K - 1))
    chisq_het = (
        beta_G[:K].T
        @ inv(GtG_inv[:K, :K])
        @ beta_G[:K]
        / (sse_G / (N_eff - (2 * K - 1)))
    )
    df_het = matrix_rank(GtG) - matrix_rank(GtG[K:, K:])

    ## Wald test for genotypes
    beta_H, sse_H, HtH = ols(jnp.concatenate([H[:, None], L], axis=1), Y)
    chisq_hom = (beta_H[0] ** 2) * HtH[0, 0] / (sse_H / (N_eff - K))

    return chisq_hom, chisq_het, beta_G[:K], df_het


def _qt_wald_nolanc_core(
    G: Array, Y: Array, Q: Array, M: Array, N_eff: int
) -> tuple[Array, Array, Array, Array]:
    K = G.shape[1]

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = ols_qr_res(G, Q) * M[:, None]
    H = ols_qr_res(H, Q) * M

    ## Wald test for anc-deconvoluted genotypes
    beta_G, sse_G, GtG = ols(G, Y)
    chisq_het = beta_G.T @ GtG @ beta_G / (sse_G / (N_eff - K))
    df_het = matrix_rank(GtG)

    ## Wald test for genotypes
    beta_H, sse_H, HtH = ols(H[:, None], Y)
    chisq_hom = (beta_H[0] ** 2) * HtH[0, 0] / (sse_H / (N_eff - K))

    return chisq_hom, chisq_het, beta_G, df_het


def _bt_wald_lanc_core(
    G: Array,
    L: Array,
    Y: Array,
    Q_w: Array,
    W_sqrt: Array,
    O: Array,
    M: Array,
    N_eff: int,
) -> tuple[Array, Array, Array, Array]:
    K = G.shape[1]
    alpha = N_eff * 1e-6

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    H = wls_qr_res(H[:, None], Q_w, W_sqrt[:, None])[:, 0]
    G = wls_qr_res(G, Q_w, W_sqrt[:, None])
    L = wls_qr_res(L, Q_w, W_sqrt[:, None])

    ## Wald test for anc-deconvoluted genotypes
    beta_G, GtG = logit(jnp.concatenate([G, L], axis=1), Y, O, M, alpha)
    GtG_inv = inv(GtG + alpha * jnp.identity(2 * K - 1))
    chisq_het = beta_G[:K].T @ inv(GtG_inv[:K, :K]) @ beta_G[:K]
    df_het = matrix_rank(GtG) - matrix_rank(GtG[K:, K:])

    ## Wald test for genotypes
    beta_H, HtH = logit(jnp.concatenate([H[:, None], L], axis=1), Y, O, M, alpha)
    chisq_hom = beta_H[0] ** 2 * HtH[0, 0]

    return chisq_hom, chisq_het, beta_G[:K], df_het


def _bt_wald_nolanc_core(
    G: Array, Y: Array, Q_w: Array, W_sqrt: Array, O: Array, M: Array, N_eff: int
) -> tuple[Array, Array, Array, Array]:
    alpha = N_eff * 1e-6

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    H = wls_qr_res(H[:, None], Q_w, W_sqrt[:, None])[:, 0]
    G = wls_qr_res(G, Q_w, W_sqrt[:, None])

    ## Wald test for anc-deconvoluted genotypes
    beta_G, GtG = logit(G, Y, O, M, alpha)
    chisq_het = beta_G.T @ GtG @ beta_G
    df_het = matrix_rank(GtG)

    ## Wald test for genotypes
    beta_H, HtH = logit(H[:, None], Y, O, M, alpha)
    chisq_hom = beta_H[0] ** 2 * HtH[0, 0]

    return chisq_hom, chisq_het, beta_G, df_het


### ─────────────────────────────────────────────────────────────
### Block-wise functions
### ─────────────────────────────────────────────────────────────

qt_score_lanc = jax.jit(
    jax.vmap(
        jax.vmap(_qt_score_lanc_core, in_axes=(None, None, 1, None, 1, 0)),
        in_axes=(1, 1, None, None, None, None),
    )
)

bt_score_lanc = jax.jit(
    jax.vmap(
        jax.vmap(_bt_score_lanc_core, in_axes=(None, None, 1, 2, 1, 1, 1, 0)),
        in_axes=(1, 1, None, None, None, None, None, None),
    )
)


qt_score_nolanc = jax.jit(
    jax.vmap(
        jax.vmap(_qt_score_nolanc_core, in_axes=(None, 1, None, 1, 0)),
        in_axes=(1, None, None, None, None),
    )
)

bt_score_nolanc = jax.jit(
    jax.vmap(
        jax.vmap(_bt_score_nolanc_core, in_axes=(None, 1, 2, 1, 1, 1, 0)),
        in_axes=(1, None, None, None, None, None, None),
    )
)


qt_wald_lanc = jax.jit(
    jax.vmap(
        jax.vmap(_qt_wald_lanc_core, in_axes=(None, None, 1, None, 1, 0)),
        in_axes=(1, 1, None, None, None, None),
    )
)

bt_wald_lanc = jax.jit(
    jax.vmap(
        jax.vmap(_bt_wald_lanc_core, in_axes=(None, None, 1, 2, 1, 1, 1, 0)),
        in_axes=(1, 1, None, None, None, None, None, None),
    )
)


qt_wald_nolanc = jax.jit(
    jax.vmap(
        jax.vmap(_qt_wald_nolanc_core, in_axes=(None, 1, None, 1, 0)),
        in_axes=(1, None, None, None, None),
    )
)

bt_wald_nolanc = jax.jit(
    jax.vmap(
        jax.vmap(_bt_wald_nolanc_core, in_axes=(None, 1, 2, 1, 1, 1, 0)),
        in_axes=(1, None, None, None, None, None, None),
    )
)
