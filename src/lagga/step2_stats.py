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
from .models import logistic_ridge

### ─────────────────────────────────────────────────────────────
### Kernels (test for a single variant and phenotype)
### ─────────────────────────────────────────────────────────────


def _qt_lanc_core(
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

    ## Score for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ r_L
    I22_inv = pinv(G_res.T @ G_res)
    chisq_het = jnp.einsum("kp,kl,lp->p", U, I22_inv, U) / mse_null
    beta_anc = U * jnp.diagonal(I22_inv)[:, None]
    df_het = matrix_rank(I22_inv)

    ## Score for genotypes (homogeneous test)
    UH = H.T @ r_L
    I22_inv_H = 1.0 / jnp.sum(H_res**2)
    chisq_hom = (UH**2) * I22_inv_H / mse_null

    return chisq_hom, chisq_het, beta_anc, df_het


def _bt_lanc_core(
    G: Array, L: Array, Y: Array, Q_w: Array, W_sqrt: Array, O: Array
) -> tuple[Array, Array, Array, Array]:
    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    H = H[:, None] - jnp.einsum(
        "ncp,cp->np", Q_w, jnp.einsum("ncp,np->cp", Q_w, H[:, None] * W_sqrt)
    )
    G = G[:, :, None] - jnp.einsum(
        "ncp,ckp->nkp",
        Q_w,
        jnp.einsum("ncp,nkp->ckp", Q_w, G[:, :, None] * W_sqrt[:, None, :]),
    )
    L = L[:, :, None] - jnp.einsum(
        "ncp,ckp->nkp",
        Q_w,
        jnp.einsum("ncp,nkp->ckp", Q_w, L[:, :, None] * W_sqrt[:, None, :]),
    )

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
    GW = G * W_L_sqrt[:, None, :]
    HW = H * W_L_sqrt
    G_res = GW - jnp.einsum("nap,map,mkp->nkp", QL, QL, GW)
    H_res = HW - jnp.einsum("nap,map,mp->np", QL, QL, HW)

    ## Score for anc-deconvoluted genotypes
    U = jnp.einsum("nkp,np->kp", G, R)
    I22_inv = jnp.linalg.pinv(jnp.einsum("nkp,nlp->pkl", G_res, G_res))
    chisq_het = jnp.einsum("kp,pkl,lp->p", U, I22_inv, U)
    beta_anc = U * jnp.diagonal(I22_inv, axis1=-1).T
    df_het = jnp.linalg.matrix_rank(I22_inv)

    ## Score for genotypes
    UH = jnp.sum(H * R, axis=0)
    I22_inv_H = (
        jnp.linalg.pinv(jnp.sum(H_res**2, axis=0).reshape((H_res.shape[1], 1, 1)))
        .squeeze(1)
        .squeeze(1)
    )
    chisq_hom = (UH**2) * I22_inv_H

    return chisq_hom, chisq_het, beta_anc, df_het


def _qt_nolanc_core(G: Array, Y: Array, Q: Array) -> tuple[Array, Array, Array, Array]:
    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = G - Q @ (Q.T @ G)
    H = H - Q @ (Q.T @ H)
    mse_null = jnp.mean(Y**2, axis=0)

    ## Score for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ Y
    I22_inv = pinv(G.T @ G)
    chisq_het = jnp.einsum("kp,kl,lp->p", U, I22_inv, U) / mse_null
    beta_anc = U * jnp.diagonal(I22_inv)[:, None]
    df_het = matrix_rank(I22_inv)

    ## Score for genotypes (homogeneous test)
    UH = H.T @ Y
    I22_inv_H = 1.0 / jnp.sum(H**2)
    chisq_hom = (UH**2) * I22_inv_H / mse_null

    return chisq_hom, chisq_het, beta_anc, df_het


def _bt_nolanc_core(
    G: Array, Y: Array, Q_w: Array, W_sqrt: Array, O: Array
) -> tuple[Array, Array, Array, Array]:
    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    H = H[:, None] - jnp.einsum(
        "ncp,cp->np", Q_w, jnp.einsum("ncp,np->cp", Q_w, H[:, None] * W_sqrt)
    )
    G = G[:, :, None] - jnp.einsum(
        "ncp,ckp->nkp",
        Q_w,
        jnp.einsum("ncp,nkp->ckp", Q_w, G[:, :, None] * W_sqrt[:, None, :]),
    )

    ## Null model
    mu = expit(O)
    R = Y - mu

    ## Score for anc-deconvoluted genotypes
    U = jnp.einsum("nkp,np->kp", G, R)
    I22_inv = jnp.linalg.pinv(jnp.einsum("nkp,nlp->pkl", G, G))
    chisq_het = jnp.einsum("kp,pkl,lp->p", U, I22_inv, U)
    beta_anc = U * jnp.diagonal(I22_inv, axis1=-1).T
    df_het = jnp.linalg.matrix_rank(I22_inv)

    ## Score for genotypes
    UH = jnp.sum(H * R, axis=0)
    I22_inv_H = (
        jnp.linalg.pinv(jnp.sum(H**2, axis=0).reshape((H.shape[1], 1, 1)))
        .squeeze(1)
        .squeeze(1)
    )
    chisq_hom = (UH**2) * I22_inv_H

    return chisq_hom, chisq_het, beta_anc, df_het


### ─────────────────────────────────────────────────────────────
### Tests across blocks and phenotypes
### ─────────────────────────────────────────────────────────────


@jax.jit
def qt_lanc(G: Array, L: Array, Y: Array, Q: Array):
    return jax.vmap(_qt_lanc_core, in_axes=(1, 1, None, None))(G, L, Y, Q)


@jax.jit
def bt_lanc(G: Array, L: Array, Y: Array, Q_w: Array, W_sqrt: Array, O: Array):
    return jax.vmap(_bt_lanc_core, in_axes=(1, 1, None, None, None, None))(
        G, L, Y, Q_w, W_sqrt, O
    )


@jax.jit
def qt_nolanc(G: Array, Y: Array, Q: Array):
    return jax.vmap(_qt_nolanc_core, in_axes=(1, None, None))(G, Y, Q)


@jax.jit
def bt_nolanc(G: Array, Y: Array, Q_w: Array, W_sqrt: Array, O: Array):
    return jax.vmap(_bt_nolanc_core, in_axes=(1, None, None, None, None))(
        G, Y, Q_w, W_sqrt, O
    )
