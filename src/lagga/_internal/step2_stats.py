# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Single variant associations.

This module contains functions for calculating the tests statistics used in lagga step 2
"""

import jax
import jax.numpy as jnp
from jax.numpy.linalg import pinv, matrix_rank, qr
from jaxtyping import Array
from jax.scipy.special import expit
from typing import Callable, Any
from .models import logistic_ridge

### ─────────────────────────────────────────────────────────────
### Helpers
### ─────────────────────────────────────────────────────────────


def wls_resid(X: Array, Q: Array, W: Array) -> Array:
    Xhat = jnp.einsum("ncp,mcp,mkp->nkp", Q, Q, X * W) / W
    return X - Xhat


### ─────────────────────────────────────────────────────────────
### Kernels (test for a single variant and phenotype)
### ─────────────────────────────────────────────────────────────


def _qt_score_lanc_core(
    G: Array, L: Array, Y: Array, Q: Array
) -> tuple[Array, Array, Array, Array]:
    N, K = G.shape

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = G - Q @ (Q.T @ G)
    L = L - Q @ (Q.T @ L)
    H = H - Q @ (Q.T @ H)

    ## Fit null model: Y ~ L
    LtL_inv = pinv(L.T @ L)
    r_L = Y - L @ (LtL_inv @ L.T @ Y)
    mse_null = jnp.sum(r_L**2, axis=0) / (N - (K - 1))

    ## Residualize G, H by L
    G_res = G - L @ (LtL_inv @ (L.T @ G))
    H_res = H - L @ (LtL_inv @ (L.T @ H))

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ r_L
    I22_inv = pinv(G_res.T @ G_res)
    chisq_het = jnp.einsum("kp,kl,lp->p", U, I22_inv, U) / mse_null
    beta_anc = U * jnp.diagonal(I22_inv)[:, None]
    df_het = matrix_rank(I22_inv)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ r_L
    I22_inv_H = 1.0 / jnp.sum(H_res**2)
    chisq_hom = (UH**2) * I22_inv_H / mse_null

    return chisq_hom, chisq_het, beta_anc, df_het


def _bt_score_lanc_core(
    G: Array, L: Array, Y: Array, Q_w: Array, W_sqrt: Array, O: Array
) -> tuple[Array, Array, Array, Array]:
    ## Genotypes
    H = jnp.sum(G, axis=1, keepdims=True)

    ## Residualize by covariates
    H = wls_resid(H[:, :, None], Q_w, W_sqrt[:, None, :])[:, 0, :]
    G = wls_resid(G[:, :, None], Q_w, W_sqrt[:, None, :])
    L = wls_resid(L[:, :, None], Q_w, W_sqrt[:, None, :])

    ## Fit L + offset null model (logistic)
    ## logistic_ridge(X, y, offset, sample_weights, l2, maxiter) -> beta (K-1,)
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
    G_res = wls_resid(G, QL, W_L_sqrt[:, None, :])
    H_res = wls_resid(H[:, None, :], QL, W_L_sqrt[:, None, :])[:, 0, :]

    ## Score test for anc-deconvoluted genotypes
    U = jnp.einsum("nkp,np->kp", G, R)
    I22_inv = jnp.linalg.pinv(jnp.einsum("nkp,nlp->pkl", G_res, G_res))
    chisq_het = jnp.einsum("kp,pkl,lp->p", U, I22_inv, U)
    beta_anc = U * jnp.diagonal(I22_inv, axis1=-1).T
    df_het = jnp.linalg.matrix_rank(I22_inv)

    ## Score test for genotypes
    UH = jnp.sum(H * R, axis=0)
    I22_inv_H = (
        jnp.linalg.pinv(jnp.sum(H_res**2, axis=0).reshape((H_res.shape[1], 1, 1)))
        .squeeze(1)
        .squeeze(1)
    )
    chisq_hom = (UH**2) * I22_inv_H

    return chisq_hom, chisq_het, beta_anc, df_het


def _qt_score_nolanc_core(
    G: Array, Y: Array, Q: Array
) -> tuple[Array, Array, Array, Array]:
    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = G - Q @ (Q.T @ G)
    H = H - Q @ (Q.T @ H)
    mse_null = jnp.mean(Y**2, axis=0)

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ Y
    I22_inv = pinv(G.T @ G)
    chisq_het = jnp.einsum("kp,kl,lp->p", U, I22_inv, U) / mse_null
    beta_anc = U * jnp.diagonal(I22_inv)[:, None]
    df_het = matrix_rank(I22_inv)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ Y
    I22_inv_H = 1.0 / jnp.sum(H**2)
    chisq_hom = (UH**2) * I22_inv_H / mse_null

    return chisq_hom, chisq_het, beta_anc, df_het


def _bt_score_nolanc_core(
    G: Array, Y: Array, Q_w: Array, W_sqrt: Array, O: Array
) -> tuple[Array, Array, Array, Array]:
    ## Genotypes
    H = jnp.sum(G, axis=1, keepdims=True)

    ## Residualize by covariates
    H = wls_resid(H[:, :, None], Q_w, W_sqrt[:, None, :])[:, 0, :]
    G = wls_resid(G[:, :, None], Q_w, W_sqrt[:, None, :])

    ## Null model
    mu = expit(O)
    w = mu * (1 - mu)
    R = Y - mu

    ## Score test for anc-deconvoluted genotypes
    GW = G * jnp.sqrt(w[:, None, :])
    U = jnp.einsum("nkp,np->kp", G, R)
    I22_inv = jnp.linalg.pinv(jnp.einsum("nkp,nlp->pkl", GW, GW))
    chisq_het = jnp.einsum("kp,pkl,lp->p", U, I22_inv, U)
    beta_anc = U * jnp.diagonal(I22_inv, axis1=-1).T
    df_het = jnp.linalg.matrix_rank(I22_inv)

    ## Score test for genotypes
    HW = H * jnp.sqrt(w)
    UH = jnp.sum(H * R, axis=0)
    I22_inv_H = (
        jnp.linalg.pinv(jnp.sum(HW**2, axis=0).reshape((H.shape[1], 1, 1)))
        .squeeze(1)
        .squeeze(1)
    )
    chisq_hom = (UH**2) * I22_inv_H

    return chisq_hom, chisq_het, beta_anc, df_het


def _qt_wald_lanc_core(
    G: Array, L: Array, Y: Array, Q: Array
) -> tuple[Array, Array, Array, Array]:
    N, K = G.shape

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = G - Q @ (Q.T @ G)
    L = L - Q @ (Q.T @ L)
    H = H - Q @ (Q.T @ H)

    ## Helper
    def do_wald(X, Y):
        XtX = jnp.einsum("nc,nd->cd", X, X)
        I_ = jnp.identity(XtX.shape[1], XtX.dtype)
        XtX_inv = jnp.linalg.inv(XtX + 1e-8 * I_)
        XtY = jnp.einsum("nc,np->cp", X, Y)
        beta = XtX_inv @ XtY
        resid = Y - X @ beta
        sse = jnp.sum(resid**2, axis=0)
        return beta, sse, XtX_inv

    ## Wald test for anc-deconvoluted genotypes
    beta_G, sse_G, XtX_inv_G = do_wald(jnp.concatenate([G, L], axis=1), Y)
    chisq_het = jnp.einsum(
        "kp,kl,lp->p", beta_G[:K, :], pinv(XtX_inv_G[:K, :K]), beta_G[:K, :]
    ) / (sse_G / (N - (2 * K - 1)))
    df_het = matrix_rank(XtX_inv_G[:K, :K])

    ## Wald test for genotypes
    beta_H, sse_H, XtX_inv_H = do_wald(jnp.concatenate([H[:, None], L], axis=1), Y)
    chisq_hom = (beta_H[0, :] ** 2) / (1e-8 + XtX_inv_H[0, 0]) / (sse_H / (N - K))

    return chisq_hom, chisq_het, beta_G[:K, :], df_het


def _qt_wald_nolanc_core(
    G: Array, Y: Array, Q: Array
) -> tuple[Array, Array, Array, Array]:
    N, K = G.shape

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = G - Q @ (Q.T @ G)
    H = H - Q @ (Q.T @ H)

    ## Helper
    def do_wald(X, Y):
        XtX = jnp.einsum("nc,nd->cd", X, X)
        I_ = jnp.identity(XtX.shape[1], XtX.dtype)
        XtX_inv = jnp.linalg.inv(XtX + 1e-8 * I_)
        XtY = jnp.einsum("nc,np->cp", X, Y)
        beta = XtX_inv @ XtY
        resid = Y - X @ beta
        sse = jnp.sum(resid**2, axis=0)
        return beta, sse, XtX_inv

    ## Wald test for anc-deconvoluted genotypes
    beta_G, sse_G, XtX_inv_G = do_wald(G, Y)
    chisq_het = jnp.einsum("kp,kl,lp->p", beta_G, pinv(XtX_inv_G), beta_G) / (
        sse_G / (N - K)
    )
    df_het = matrix_rank(XtX_inv_G)

    ## Wald test for genotypes
    beta_H, sse_H, XtX_inv_H = do_wald(H[:, None], Y)
    chisq_hom = (beta_H[0, :] ** 2) / (1e-8 + XtX_inv_H[0, 0]) / (sse_H / (N - K))

    return chisq_hom, chisq_het, beta_G, df_het


def _bt_wald_lanc_core(
    G: Array, L: Array, Y: Array, Q_w: Array, W_sqrt: Array, O: Array
) -> tuple[Array, Array, Array, Array]:
    N, K = G.shape

    ## Genotypes
    H = jnp.sum(G, axis=1, keepdims=True)

    ## Residualize by covariates
    H = wls_resid(H[:, :, None], Q_w, W_sqrt[:, None, :])
    G = wls_resid(G[:, :, None], Q_w, W_sqrt[:, None, :])
    L = wls_resid(L[:, :, None], Q_w, W_sqrt[:, None, :])

    ## Helper
    def do_wald(X, Y, O):
        beta = jax.vmap(logistic_ridge, in_axes=(2, 1, 1, None, None, None))(
            X, Y, O, jnp.ones(N), 1e-8, 10
        )
        mu = expit(jnp.einsum("nkp,pk->np", X, beta) + O)
        w = mu * (1 - mu)
        XW = X * jnp.sqrt(w[:, None, :])
        I_inv = pinv(jnp.einsum("nkp,nlp->pkl", XW, XW))
        return (beta, I_inv)

    ## Wald test for anc-deconvoluted genotypes
    beta_G, I_inv_G = do_wald(jnp.concatenate([G, L], axis=1), Y, O)
    chisq_het = jnp.einsum(
        "pk,pkl,pl->p", beta_G[:, :K], pinv(I_inv_G[:, :K, :K]), beta_G[:, :K]
    )
    df_het = matrix_rank(I_inv_G[:, :K, :K])

    ## Wald test for genotypes
    beta_H, I_inv_H = do_wald(jnp.concatenate([H, L], axis=1), Y, O)
    chisq_hom = beta_H[:, 0] ** 2 / (I_inv_H[:, 0, 0] + 1e-8)

    return chisq_hom, chisq_het, beta_G[:, :K].T, df_het


def _bt_wald_nolanc_core(
    G: Array, Y: Array, Q_w: Array, W_sqrt: Array, O: Array
) -> tuple[Array, Array, Array, Array]:
    N, K = G.shape

    ## Genotypes
    H = jnp.sum(G, axis=1, keepdims=True)

    ## Residualize by covariates
    H = wls_resid(H[:, :, None], Q_w, W_sqrt[:, None, :])
    G = wls_resid(G[:, :, None], Q_w, W_sqrt[:, None, :])

    ## Helper
    def do_wald(X, Y, O):
        beta = jax.vmap(logistic_ridge, in_axes=(2, 1, 1, None, None, None))(
            X, Y, O, jnp.ones(N), 1e-8, 10
        )
        mu = expit(jnp.einsum("nkp,pk->np", X, beta) + O)
        w = mu * (1 - mu)
        XW = X * jnp.sqrt(w[:, None, :])
        I_inv = pinv(jnp.einsum("nkp,nlp->pkl", XW, XW))
        return (beta, I_inv)

    ## Wald test for anc-deconvoluted genotypes
    beta_G, I_inv_G = do_wald(G, Y, O)
    chisq_het = jnp.einsum(
        "pk,pkl,pl->p", beta_G[:, :K], pinv(I_inv_G[:, :K, :K]), beta_G[:, :K]
    )
    df_het = matrix_rank(I_inv_G[:, :K, :K])

    ## Wald test for genotypes
    beta_H, I_inv_H = do_wald(H, Y, O)
    chisq_hom = beta_H[:, 0] ** 2 / (I_inv_H[:, 0, 0] + 1e-8)

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
