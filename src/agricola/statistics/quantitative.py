# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Quantitative-trait step-2 association statistic kernels."""

import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import Array

from .common import (
    adj_by_lanc,
    het_score,
    hom_score,
    make_blockwise,
    mask_score,
    mask_wald,
    masked_inv,
    prep_geno,
    prep_lanc_geno,
    qr_resid,
)

### ─────────────────────────────────────────────────────────────


def _qt_score_lanc(G: Array, L: Array, Y: Array, Q: Array, N_eff: Array) -> tuple[Array, ...]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    ## Get H and residualize all by covariates
    G, L, H = prep_lanc_geno(G, L, Q)

    ## Residualize genotypes against a rank-aware ancestry basis.
    QL, G, Gl, G_mask, H, Hl, H_mask = adj_by_lanc(G, H, L)
    rank_L = jnp.sum(jnp.sum(QL**2, axis=0) > 0)

    ## Fit null model: Y ~ L
    r_L = qr_resid(Y, QL)
    mse_null = jnp.sum(r_L**2, axis=0) / (N_eff - rank_L)

    ## Score test for the joint ancestry-specific effects.
    U = G.T @ r_L
    GltGl = Gl.T @ Gl
    beta_het, chisq_anc, chisq_het = het_score(U, GltGl, G_mask, mse_null)

    ## Score test for the common homogeneous effect.
    beta_hom, chisq_hom = hom_score(H.T @ r_L, jnp.sum(Hl**2), mse_null)

    return mask_score(beta_het, beta_hom, chisq_anc, chisq_het, chisq_hom, G_mask[:, None], H_mask)


def _qt_score_nolanc(G: Array, Y: Array, Q: Array, N_eff: Array) -> tuple[Array, ...]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    ## Get H and residualize all by covariates
    G, H = prep_geno(G, Q)

    mse_null = jnp.sum(Y**2, axis=0) / N_eff

    ## Mask out low variation columns
    G_mask = jnp.sum(G**2, axis=0) > 0
    G = G * G_mask[None, :]
    H_mask = jnp.sum(H**2, axis=0) > 0
    H = H * H_mask

    ## Score test for the joint ancestry-specific effects.
    U = G.T @ Y
    GtG = G.T @ G
    beta_het, chisq_anc, chisq_het = het_score(U, GtG, G_mask, mse_null)

    ## Score test for the common homogeneous effect.
    UH = H.T @ Y
    HtH = jnp.sum(H**2)
    beta_hom, chisq_hom = hom_score(UH, HtH, mse_null)

    return mask_score(beta_het, beta_hom, chisq_anc, chisq_het, chisq_hom, G_mask[:, None], H_mask)


def _qt_wald_lanc(G: Array, L: Array, Y: Array, Q: Array, N_eff: Array) -> tuple[Array, ...]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    K = G.shape[1]

    ## Get H and residualize all by covariates
    G, L, H = prep_lanc_geno(G, L, Q)

    ## Residualize genotypes against a rank-aware ancestry basis.
    QL, G, Gl, G_mask, H, Hl, H_mask = adj_by_lanc(G, H, L)
    rank_L = jnp.sum(jnp.sum(QL**2, axis=0) > 0)
    H = H[:, None]
    Hl = Hl[:, None]

    ## Fit null model: Y ~ L
    r_L = qr_resid(Y, QL)

    ## Wald test for the joint ancestry-specific effects.
    Gtr = Gl.T @ r_L
    GltGl = Gl.T @ Gl
    GltGl_inv = masked_inv(GltGl, G_mask)
    beta_het = GltGl_inv @ Gtr
    r_G = r_L - Gl @ beta_het
    sse_het = jnp.sum(r_G**2, axis=0)
    ## Use the effective fitted rank for the residual variance estimate.
    mse_het = sse_het / (N_eff - rank_L - jnp.sum(G_mask))
    chisq_anc = beta_het**2 / jnp.diagonal(GltGl_inv)[:, None] / mse_het
    chisq_het = jnp.einsum("kp,kl,lp->p", Gtr, GltGl_inv, Gtr) / mse_het

    ## Wald test for the common homogeneous effect.
    Htr = Hl.T @ r_L
    HltHl = Hl.T @ Hl
    beta_hom = Htr / HltHl
    r_H = r_L - (Hl @ beta_hom)
    sse_hom = jnp.sum(r_H**2, axis=0)
    mse_hom = sse_hom / (N_eff - rank_L - jnp.sum(H_mask))
    chisq_hom = (Htr**2) / HltHl / mse_hom

    ## LRT
    chisq_lrt = N_eff * jnp.log(sse_hom / sse_het)

    return mask_wald(
        beta_het,
        beta_hom,
        chisq_anc,
        chisq_het,
        chisq_hom,
        chisq_lrt,
        G_mask[:, None],
        H_mask,
    )


def _qt_wald_nolanc(G: Array, Y: Array, Q: Array, N_eff: Array) -> tuple[Array, ...]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    K = G.shape[1]

    ## Get H and residualize all by covariates
    G, H = prep_geno(G, Q)

    ## Mask out low variation columns
    G_mask = jnp.sum(G**2, axis=0) > 0
    G = G * G_mask[None, :]
    H_mask = jnp.sum(H**2, axis=0) > 0
    H = H * H_mask
    H = H[:, None]

    ## Wald test for the joint ancestry-specific effects.
    Gtr = G.T @ Y
    GtG = G.T @ G
    GtG_inv = masked_inv(GtG, G_mask)
    beta_het = GtG_inv @ Gtr
    r_G = Y - G @ beta_het
    sse_het = jnp.sum(r_G**2, axis=0)
    mse_het = sse_het / (N_eff - K)
    chisq_anc = beta_het**2 / jnp.diagonal(GtG_inv)[:, None] / mse_het
    chisq_het = jnp.einsum("kp,kl,lp->p", Gtr, GtG_inv, Gtr) / mse_het

    ## Wald test for the common homogeneous effect.
    Htr = H.T @ Y
    HtH = H.T @ H
    beta_hom = Htr / HtH
    r_H = Y - (H @ beta_hom)
    sse_hom = jnp.sum(r_H**2, axis=0)
    mse_hom = sse_hom / (N_eff - 1)
    chisq_hom = (Htr**2) / HtH / mse_hom

    chisq_lrt = N_eff * jnp.log(sse_hom / sse_het)

    return mask_wald(
        beta_het,
        beta_hom,
        chisq_anc,
        chisq_het,
        chisq_hom,
        chisq_lrt,
        G_mask[:, None],
        H_mask,
    )


### ─────────────────────────────────────────────────────────────
### Block-wise functions
### ─────────────────────────────────────────────────────────────

qt_score_lanc = make_blockwise(_qt_score_lanc, (1, 1, None, None, None), (3, 3, 1, 2, 0))

qt_score_lanc_impute = jit(
    vmap(_qt_score_lanc, in_axes=(1, 1, None, None, None)),
)

qt_score_nolanc = make_blockwise(_qt_score_nolanc, (1, None, None, None), (3, 1, 2, 0))


qt_score_nolanc_impute = jit(
    vmap(_qt_score_nolanc, in_axes=(1, None, None, None)),
)

qt_wald_lanc = make_blockwise(_qt_wald_lanc, (1, 1, None, None, None), (3, 3, 1, 2, 0))

qt_wald_lanc_impute = jit(
    vmap(_qt_wald_lanc, in_axes=(1, 1, None, None, None)),
)

qt_wald_nolanc = make_blockwise(_qt_wald_nolanc, (1, None, None, None), (3, 1, 2, 0))

qt_wald_nolanc_impute = jit(
    vmap(_qt_wald_nolanc, in_axes=(1, None, None, None)),
)
