# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Binary-trait step-2 association statistic kernels."""

import jax.numpy as jnp
from jax.numpy.linalg import solve
from jax.scipy.special import expit
from jaxtyping import Array

from .common import (
    adj_by_lanc,
    het_score,
    hom_score,
    logistic,
    make_blockwise,
    mask_score,
    mask_wald,
    masked_inv,
    prep_geno,
    prep_lanc_geno,
)

### ─────────────────────────────────────────────────────────────


def _bt_score_lanc(
    G: Array, L: Array, Y: Array, Q: Array, offset: Array, M: Array
) -> tuple[Array, ...]:
    ## Get H and residualize all by covariates
    G, L, H = prep_lanc_geno(G, L, Q)

    ## Fit G,H ~ L and mask out collinear columns
    _, G, Gl, G_mask, H, Hl, H_mask = adj_by_lanc(G, H, L)

    ## Fit null model L + offset
    L_mask = jnp.sum(L**2, axis=0) > 0
    beta_L = logistic(L, Y, offset, M, L_mask)
    mu = expit(L @ beta_L + offset)
    R = (Y - mu) * M
    W_L_sqrt = jnp.sqrt(mu * (1.0 - mu)) * M

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ R
    Glw = Gl * M[:, None] * W_L_sqrt[:, None]
    GltGl = Glw.T @ Glw
    beta_het, chisq_anc, chisq_het = het_score(U[:, None], GltGl, G_mask)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ R
    Hlw = Hl * M * W_L_sqrt
    HtH = Hlw.T @ Hlw
    beta_hom, chisq_hom = hom_score(UH, HtH)

    return mask_score(
        beta_het[:, 0], beta_hom, chisq_anc[:, 0], chisq_het, chisq_hom, G_mask, H_mask
    )


def _bt_score_nolanc(G: Array, Y: Array, Q: Array, offset: Array, M: Array) -> tuple[Array, ...]:
    ## Get H and residualize all by covariates
    G, H = prep_geno(G, Q)

    ## Mask out low variation columns
    G_mask = jnp.sum(G**2, axis=0) > 0
    H_mask = jnp.sum(H**2, axis=0) > 0

    ## Null model
    mu = expit(offset)
    R = (Y - mu) * M
    W_sqrt = jnp.sqrt(mu * (1.0 - mu)) * M

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ R
    Gw = G * M[:, None] * W_sqrt[:, None]
    GtG = Gw.T @ Gw
    beta_het, chisq_anc, chisq_het = het_score(U[:, None], GtG, G_mask)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ R
    Hw = H * M * W_sqrt
    HtH = Hw.T @ Hw
    beta_hom, chisq_hom = hom_score(UH, HtH)

    return mask_score(
        beta_het[:, 0], beta_hom, chisq_anc[:, 0], chisq_het, chisq_hom, G_mask, H_mask
    )


def _bt_wald_lanc(
    G: Array, L: Array, Y: Array, Q: Array, offset: Array, M: Array
) -> tuple[Array, ...]:
    K = G.shape[1]

    ## Get H and residualize all by covariates
    G, L, H = prep_lanc_geno(G, L, Q)

    ## Fit G,H ~ L and mask out collinear columns
    _, G, _, G_mask, H, _, H_mask = adj_by_lanc(G, H, L)
    H = H[:, None]
    L_mask = jnp.sum(L**2, axis=0) > 0

    ## Wald test for anc-deconvoluted genotypes
    Xg_mask = jnp.concatenate([G_mask, L_mask])
    Xg = jnp.concatenate([G, L], axis=1)
    beta_het = logistic(Xg, Y, offset, M, Xg_mask)
    etag = Xg @ beta_het + offset
    mu = expit(etag)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Xw = Xg * W_sqrt[:, None] * M[:, None]
    XtX = Xw.T @ Xw
    XtXw_inv = masked_inv(XtX, Xg_mask)
    chisq_anc = beta_het[:K] ** 2 / jnp.diagonal(XtXw_inv[:K, :K])
    chisq_het = beta_het[:K].T @ solve(XtXw_inv[:K, :K], beta_het[:K])

    ## Wald test for genotypes
    Xh_mask = jnp.concatenate([H_mask[None], L_mask])
    Xh = jnp.concatenate([H, L], axis=1)
    beta_hom = logistic(Xh, Y, offset, M, Xh_mask)
    etah = Xh @ beta_hom + offset
    mu = expit(etah)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Xw = Xh * W_sqrt[:, None] * M[:, None]
    XtX = Xw.T @ Xw
    XtXw_inv = masked_inv(XtX, Xh_mask)
    chisq_hom = beta_hom[0] ** 2 / XtXw_inv[0, 0]

    ## LRT
    l_het = (Y * etag - jnp.log(1 + jnp.exp(etag))) * M
    l_hom = (Y * etah - jnp.log(1 + jnp.exp(etah))) * M
    chisq_lrt = 2 * jnp.sum(l_het - l_hom)

    return mask_wald(
        beta_het[:K],
        beta_hom[0],
        chisq_anc,
        chisq_het,
        chisq_hom,
        chisq_lrt,
        G_mask,
        H_mask,
    )


def _bt_wald_nolanc(G: Array, Y: Array, Q: Array, offset: Array, M: Array) -> tuple[Array, ...]:
    K = G.shape[1]

    ## Get H and residualize all by covariates
    G, H = prep_geno(G, Q)
    H = H[:, None]

    ## Wald test for anc-deconvoluted genotypes
    G_mask = jnp.sum(G**2, axis=0) > 0
    beta_het = logistic(G, Y, offset, M, G_mask)
    etag = G @ beta_het + offset
    mu = expit(etag)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Gw = G * W_sqrt[:, None] * M[:, None]
    GtG = Gw.T @ Gw
    GtGw_inv = masked_inv(GtG, G_mask)
    chisq_anc = beta_het[:K] ** 2 / jnp.diagonal(GtGw_inv[:K, :K])
    chisq_het = beta_het[:K].T @ solve(GtGw_inv[:K, :K], beta_het[:K])

    ## Wald test for genotypes
    H_mask = jnp.sum(H**2, axis=0) > 0
    beta_hom = logistic(H, Y, offset, M, H_mask)
    etah = H @ beta_hom + offset
    mu = expit(etah)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Hw = H * W_sqrt[:, None] * M[:, None]
    HtH = Hw.T @ Hw
    chisq_hom = beta_hom**2 * HtH

    ## LRT
    l_het = (Y * etag - jnp.log(1 + jnp.exp(etag))) * M
    l_hom = (Y * etah - jnp.log(1 + jnp.exp(etah))) * M
    chisq_lrt = 2 * jnp.sum(l_het - l_hom)

    return mask_wald(
        beta_het[:K],
        beta_hom[0],
        chisq_anc,
        chisq_het,
        chisq_hom,
        chisq_lrt,
        G_mask,
        H_mask,
    )


### ─────────────────────────────────────────────────────────────
### Block-wise functions
### ─────────────────────────────────────────────────────────────

bt_score_lanc = make_blockwise(_bt_score_lanc, (1, 1, None, None, None, None), (3, 3, 1, 2, 1, 1))

bt_score_nolanc = make_blockwise(_bt_score_nolanc, (1, None, None, None, None), (3, 1, 2, 1, 1))

bt_wald_lanc = make_blockwise(_bt_wald_lanc, (1, 1, None, None, None, None), (3, 3, 1, 2, 1, 1))

bt_wald_nolanc = make_blockwise(_bt_wald_nolanc, (1, None, None, None, None), (3, 1, 2, 1, 1))
