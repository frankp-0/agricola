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
    Xhat = jnp.einsum("ncp,mcp,mkp->nkp", Q, Q, X * W) / W
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


def logit(X: Array, Y: Array, O: Array) -> tuple[Array, Array]:
    N, _, _ = X.shape
    alpha = N * 1e-6
    beta = jax.vmap(logistic_ridge, in_axes=(2, 1, 1, None, None, None))(
        X, Y, O, jnp.ones(N), alpha, 10
    )
    mu = expit(jnp.einsum("nkp,pk->np", X, beta) + O)
    w = mu * (1 - mu)
    XW = X * jnp.sqrt(w[:, None, :])
    XtX_W = jnp.einsum("nkp,nlp->pkl", XW, XW)
    return beta, XtX_W


### ─────────────────────────────────────────────────────────────
### Kernels (test for a single variant and phenotype)
### ─────────────────────────────────────────────────────────────


def _qt_score_lanc_core(
    G: Array, L: Array, Y: Array, Q: Array
) -> tuple[Array, Array, Array, Array]:
    N, K = G.shape
    alpha = N * 1e-5

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = ols_qr_res(G, Q)
    L = ols_qr_res(L, Q)
    H = ols_qr_res(H, Q)

    ## Fit null model: Y ~ L
    QL, _ = qr(L)
    r_L = ols_qr_res(Y, QL)
    mse_null = jnp.sum(r_L**2, axis=0) / (N - (K - 1))

    ## Residualize G, H by L
    G_res = ols_qr_res(G, QL)
    H_res = ols_qr_res(H, QL)

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ r_L
    GtG = G_res.T @ G_res
    GtG_inv = inv(GtG + alpha * jnp.identity(K))
    chisq_het = jnp.einsum("kp,kl,lp->p", U, GtG_inv, U) / mse_null
    beta_anc = U * jnp.diagonal(GtG_inv)[:, None]
    df_het = matrix_rank(GtG)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ r_L
    HtH = jnp.sum(H_res**2)
    chisq_hom = (UH**2) / HtH / mse_null

    return chisq_hom, chisq_het, beta_anc, df_het


def _bt_score_lanc_core(
    G: Array, L: Array, Y: Array, Q_w: Array, W_sqrt: Array, O: Array
) -> tuple[Array, Array, Array, Array]:
    N, K = G.shape
    alpha = N * 1e-5

    ## Genotypes
    H = jnp.sum(G, axis=1, keepdims=True)

    ## Residualize by covariates
    H = wls_qr_res(H[:, :, None], Q_w, W_sqrt[:, None, :])[:, 0, :]
    G = wls_qr_res(G[:, :, None], Q_w, W_sqrt[:, None, :])
    L = wls_qr_res(L[:, :, None], Q_w, W_sqrt[:, None, :])

    ## Fit L + offset null model (logistic)
    beta_L = jax.vmap(logistic_ridge, in_axes=(2, 1, 1, None, None, None))(
        L, Y, O, jnp.ones(L.shape[0]), 0.0, 10
    )
    mu = expit(jnp.einsum("nap,pa->np", L, beta_L) + O)
    R = Y - mu
    W_L_sqrt = jnp.sqrt(mu * (1.0 - mu))

    ## Residualize G, H by L
    QL, _ = qr(
        jnp.moveaxis(L * W_L_sqrt[:, None, :], (0, 1, 2), (1, 2, 0)), mode="reduced"
    )
    QL = QL.transpose((1, 2, 0))
    G_res = wls_qr_res(G, QL, W_L_sqrt[:, None, :])
    H_res = wls_qr_res(H[:, None, :], QL, W_L_sqrt[:, None, :])[:, 0, :]

    ## Score test for anc-deconvoluted genotypes
    U = jnp.einsum("nkp,np->kp", G, R)
    GW = G_res * W_L_sqrt[:, None, :]
    GtG = jnp.einsum("nkp,nlp->pkl", GW, GW)
    GtG_inv = inv(GtG + alpha * jnp.identity(K)[None, :, :])
    chisq_het = jnp.einsum("kp,pkl,lp->p", U, GtG_inv, U)
    beta_anc = U * jnp.diagonal(GtG_inv, axis1=-1).T
    df_het = matrix_rank(GtG)

    ## Score test for genotypes
    UH = jnp.sum(H * R, axis=0)
    HW = H_res * W_L_sqrt
    HtH = jnp.sum(HW**2, axis=0)
    chisq_hom = (UH**2) / (HtH + alpha)

    return chisq_hom, chisq_het, beta_anc, df_het


def _qt_score_nolanc_core(
    G: Array, Y: Array, Q: Array
) -> tuple[Array, Array, Array, Array]:
    N, K = G.shape
    alpha = N * 1e-5

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = ols_qr_res(G, Q)
    H = ols_qr_res(H, Q)
    mse_null = jnp.mean(Y**2, axis=0)

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ Y
    GtG = G.T @ G
    GtG_inv = inv(G.T @ G + alpha * jnp.identity(K))
    chisq_het = jnp.einsum("kp,kl,lp->p", U, GtG_inv, U) / mse_null
    beta_anc = U * jnp.diagonal(GtG_inv)[:, None]
    df_het = matrix_rank(GtG)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ Y
    HtH = jnp.sum(H**2)
    chisq_hom = (UH**2) / (HtH + alpha) / mse_null

    return chisq_hom, chisq_het, beta_anc, df_het


def _bt_score_nolanc_core(
    G: Array, Y: Array, Q_w: Array, W_sqrt: Array, O: Array
) -> tuple[Array, Array, Array, Array]:
    N, K = G.shape
    alpha = N * 1e-5

    ## Genotypes
    H = jnp.sum(G, axis=1, keepdims=True)

    ## Residualize by covariates
    H = wls_qr_res(H[:, :, None], Q_w, W_sqrt[:, None, :])[:, 0, :]
    G = wls_qr_res(G[:, :, None], Q_w, W_sqrt[:, None, :])

    ## Null model
    mu = expit(O)
    w = mu * (1 - mu)
    R = Y - mu

    ## Score test for anc-deconvoluted genotypes
    GW = G * jnp.sqrt(w[:, None, :])
    U = jnp.einsum("nkp,np->kp", G, R)
    GtG = jnp.einsum("nkp,nlp->pkl", GW, GW)
    GtG_inv = inv(GtG + alpha * jnp.identity(K)[None, :, :])
    chisq_het = jnp.einsum("kp,pkl,lp->p", U, GtG_inv, U)
    beta_anc = U * jnp.diagonal(GtG_inv, axis1=-1).T
    df_het = matrix_rank(GtG)

    ## Score test for genotypes
    HW = H * jnp.sqrt(w)
    UH = jnp.sum(H * R, axis=0)
    HtH = jnp.sum(HW**2)
    chisq_hom = (UH**2) / (HtH + alpha)

    return chisq_hom, chisq_het, beta_anc, df_het


def _qt_wald_lanc_core(
    G: Array, L: Array, Y: Array, Q: Array
) -> tuple[Array, Array, Array, Array]:
    N, K = G.shape
    alpha = N * 1e-5

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = ols_qr_res(G, Q)
    L = ols_qr_res(L, Q)
    H = ols_qr_res(H, Q)

    ## Wald test for anc-deconvoluted genotypes
    beta_G, sse_G, GtG = ols(jnp.concatenate([G, L], axis=1), Y)
    GtG_inv = inv(GtG + alpha * jnp.identity(K))
    chisq_het = jnp.einsum(
        "kp,kl,lp->p", beta_G[:K, :], inv(GtG_inv[:K, :K]), beta_G[:K, :]
    ) / (sse_G / (N - (2 * K - 1)))
    df_het = matrix_rank(GtG) - matrix_rank(GtG[K:, K:])

    ## Wald test for genotypes
    beta_H, sse_H, HtH = ols(jnp.concatenate([H[:, None], L], axis=1), Y)
    chisq_hom = (beta_H[0, :] ** 2) * HtH[0, 0] / (sse_H / (N - K))

    return chisq_hom, chisq_het, beta_G[:K, :], df_het


def _qt_wald_nolanc_core(
    G: Array, Y: Array, Q: Array
) -> tuple[Array, Array, Array, Array]:
    N, K = G.shape

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = ols_qr_res(G, Q)
    H = ols_qr_res(H, Q)

    ## Wald test for anc-deconvoluted genotypes
    beta_G, sse_G, GtG = ols(G, Y)
    chisq_het = jnp.einsum("kp,kl,lp->p", beta_G, GtG, beta_G) / (sse_G / (N - K))
    df_het = matrix_rank(GtG)

    ## Wald test for genotypes
    beta_H, sse_H, HtH = ols(H[:, None], Y)
    chisq_hom = (beta_H[0, :] ** 2) * HtH[0, 0] / (sse_H / (N - K))

    return chisq_hom, chisq_het, beta_G, df_het


def _bt_wald_lanc_core(
    G: Array, L: Array, Y: Array, Q_w: Array, W_sqrt: Array, O: Array
) -> tuple[Array, Array, Array, Array]:
    N, K = G.shape
    alpha = N * 1e-5

    ## Genotypes
    H = jnp.sum(G, axis=1, keepdims=True)

    ## Residualize by covariates
    H = wls_qr_res(H[:, :, None], Q_w, W_sqrt[:, None, :])
    G = wls_qr_res(G[:, :, None], Q_w, W_sqrt[:, None, :])
    L = wls_qr_res(L[:, :, None], Q_w, W_sqrt[:, None, :])

    ## Wald test for anc-deconvoluted genotypes
    beta_G, GtG = logit(jnp.concatenate([G, L], axis=1), Y, O)
    GtG_inv = inv(GtG + alpha * jnp.identity(K)[None, :, :])
    chisq_het = jnp.einsum(
        "pk,pkl,pl->p", beta_G[:, :K], inv(GtG_inv[:, :K, :K]), beta_G[:, :K]
    )
    df_het = matrix_rank(GtG) - matrix_rank(GtG[:, K:, K:])

    ## Wald test for genotypes
    beta_H, HtH = logit(jnp.concatenate([H, L], axis=1), Y, O)
    chisq_hom = beta_H[:, 0] ** 2 * HtH[:, 0, 0]

    return chisq_hom, chisq_het, beta_G[:, :K].T, df_het


def _bt_wald_nolanc_core(
    G: Array, Y: Array, Q_w: Array, W_sqrt: Array, O: Array
) -> tuple[Array, Array, Array, Array]:
    N, K = G.shape

    ## Genotypes
    H = jnp.sum(G, axis=1, keepdims=True)

    ## Residualize by covariates
    H = wls_qr_res(H[:, :, None], Q_w, W_sqrt[:, None, :])
    G = wls_qr_res(G[:, :, None], Q_w, W_sqrt[:, None, :])

    ## Wald test for anc-deconvoluted genotypes
    beta_G, GtG = logit(G, Y, O)
    chisq_het = jnp.einsum("pk,pkl,pl->p", beta_G, GtG, beta_G)
    df_het = matrix_rank(GtG)

    ## Wald test for genotypes
    beta_H, HtH = logit(H, Y, O)
    chisq_hom = beta_H[:, 0] ** 2 * HtH[:, 0, 0]

    return chisq_hom, chisq_het, beta_G.T, df_het


### ─────────────────────────────────────────────────────────────
### Block-wise functions
### ─────────────────────────────────────────────────────────────

qt_score_lanc = jax.jit(jax.vmap(_qt_score_lanc_core, in_axes=(1, 1, None, None)))
bt_score_lanc = jax.jit(
    jax.vmap(_bt_score_lanc_core, in_axes=(1, 1, None, None, None, None))
)
qt_score_nolanc = jax.jit(jax.vmap(_qt_score_nolanc_core, in_axes=(1, None, None)))
bt_score_nolanc = jax.jit(
    jax.vmap(_bt_score_nolanc_core, in_axes=(1, None, None, None, None))
)

qt_wald_lanc = jax.jit(jax.vmap(_qt_wald_lanc_core, in_axes=(1, 1, None, None)))
bt_wald_lanc = jax.jit(
    jax.vmap(_bt_wald_lanc_core, in_axes=(1, 1, None, None, None, None))
)
qt_wald_nolanc = jax.jit(jax.vmap(_qt_wald_nolanc_core, in_axes=(1, None, None)))
bt_wald_nolanc = jax.jit(
    jax.vmap(_bt_wald_nolanc_core, in_axes=(1, None, None, None, None))
)
