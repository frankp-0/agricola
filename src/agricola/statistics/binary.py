# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Binary-trait step-2 association statistic kernels."""

from functools import partial

import jax.numpy as jnp
from jax.nn import softplus
from jax.numpy.linalg import solve
from jax.scipy.special import expit
from jaxtyping import Array

from .common import (
    adj_by_lanc,
    het_score,
    hom_score,
    lanc_basis,
    logistic_with_convergence,
    make_blockwise,
    mask_result,
    masked_inv,
    prep_geno,
    prep_lanc_geno,
)

### ─────────────────────────────────────────────────────────────


def _bt_score_lanc(
    G: Array, L: Array, Y: Array, Q: Array, offset: Array, M: Array, perform_diff: bool = True
) -> tuple[Array, ...]:
    ## Get H and residualize all by covariates
    G, L, H = prep_lanc_geno(G, L, Q)

    ## Detect ancestry rank on observed samples but retain fixed-size arrays.
    _, L_mask = lanc_basis(L, M)
    L = L * L_mask

    ## Null model
    beta_L, _ = logistic_with_convergence(L, Y, offset, M, L_mask, tol=1e-5)
    mu = expit(L @ beta_L + offset)
    R = (Y - mu) * M
    W_L_sqrt = jnp.sqrt(mu * (1.0 - mu)) * M

    ## Use the logistic-information projection for the efficient score.
    Lw = L * W_L_sqrt[:, None]
    Qw, Rw = jnp.linalg.qr(Lw, mode="reduced")
    Lw_mask = jnp.abs(jnp.diagonal(Rw)) > jnp.finfo(L.dtype).eps * max(L.shape)
    Qw = Qw * Lw_mask
    inv_W_L_sqrt = jnp.where(W_L_sqrt > 0, 1.0 / W_L_sqrt, 0.0)

    # Project in the logistic-information metric using a QR basis rather than
    # normal equations, which avoids platform-sensitive cancellation.
    G_scale = jnp.sum((G * W_L_sqrt[:, None]) ** 2, axis=0)
    Gw = G * W_L_sqrt[:, None]
    G = (Gw - Qw @ (Qw.T @ Gw)) * inv_W_L_sqrt[:, None]
    G_norm = jnp.sum((G * W_L_sqrt[:, None]) ** 2, axis=0)
    G_mask = G_norm > jnp.finfo(G.dtype).eps * max(G.shape) * G_scale
    G = G * G_mask
    U = G.T @ R

    H_scale = jnp.sum((H * W_L_sqrt) ** 2)
    Hw = H * W_L_sqrt
    H = (Hw - Qw @ (Qw.T @ Hw)) * inv_W_L_sqrt
    H_norm = jnp.sum((H * W_L_sqrt) ** 2)
    H_mask = H_norm > jnp.finfo(H.dtype).eps * max(H.shape) * H_scale
    H = H * H_mask
    UH = H.T @ R

    ## Score test for the joint ancestry-specific effects.
    Glw = G * W_L_sqrt[:, None]
    GltGl = Glw.T @ Glw
    beta_het, chisq_anc, chisq_het = het_score(U[:, None], GltGl, G_mask)

    ## Score test for the common homogeneous effect.
    Hlw = H * W_L_sqrt
    HtH = Hlw.T @ Hlw
    beta_hom, chisq_hom = hom_score(UH, HtH)
    if not perform_diff:
        chisq_diff = jnp.broadcast_to(jnp.nan, chisq_hom.shape)
        result = mask_result(
            beta_het[:, 0],
            beta_hom,
            chisq_anc[:, 0],
            chisq_het,
            chisq_hom,
            chisq_diff,
            G_mask,
            H_mask,
        )
        return (*result, jnp.asarray(jnp.nan))

    ## Fit the joint homogeneous null for the heterogeneity score test.
    X_hom = jnp.concatenate([L, H[:, None]], axis=1)
    X_hom_mask = jnp.concatenate([Lw_mask, jnp.atleast_1d(H_mask)])
    beta_hom_null, converged_hom = logistic_with_convergence(
        X_hom, Y, offset, M, X_hom_mask, tol=1e-5
    )
    mu_hom = expit(X_hom @ beta_hom_null + offset)
    R_hom = (Y - mu_hom) * M
    W_hom = jnp.sqrt(mu_hom * (1.0 - mu_hom)) * M
    G_diff = G[:, :-1] - G[:, -1, None]
    G_diff_w = G_diff * W_hom[:, None]
    X_hom_w = X_hom * W_hom[:, None]
    Q_hom, R_hom_design = jnp.linalg.qr(X_hom_w, mode="reduced")
    hom_rank_tol = (
        jnp.finfo(X_hom_w.dtype).eps
        * max(X_hom_w.shape)
        * jnp.max(jnp.abs(jnp.diagonal(R_hom_design)), initial=0)
    )
    hom_rank = jnp.abs(jnp.diagonal(R_hom_design)) > hom_rank_tol
    Q_hom = Q_hom * hom_rank
    G_diff_w = G_diff_w - Q_hom @ (Q_hom.T @ G_diff_w)
    Q_diff, R_diff = jnp.linalg.qr(G_diff_w, mode="reduced")
    rank_tol = (
        jnp.finfo(G_diff_w.dtype).eps
        * max(G_diff_w.shape)
        * jnp.max(jnp.abs(jnp.diagonal(R_diff)), initial=0)
    )
    diff_rank = jnp.abs(jnp.diagonal(R_diff)) > rank_tol
    score_design = Q_diff / jnp.where(W_hom[:, None] > 0, W_hom[:, None], 1)
    chisq_diff = jnp.sum((score_design.T @ R_hom) ** 2 * diff_rank)
    chisq_diff = jnp.where(jnp.sum(diff_rank) > 0, chisq_diff, jnp.nan)

    result = mask_result(
        beta_het[:, 0],
        beta_hom,
        chisq_anc[:, 0],
        chisq_het,
        chisq_hom,
        chisq_diff,
        G_mask,
        H_mask,
    )
    return (*result, converged_hom)


def _bt_score_nolanc(
    G: Array, Y: Array, Q: Array, offset: Array, M: Array, perform_diff: bool = True
) -> tuple[Array, ...]:
    ## Get H and residualize all by covariates
    G, H = prep_geno(G, Q)

    ## Mask out low variation columns
    G_mask = jnp.sum((G * M[:, None]) ** 2, axis=0) > 0
    H_mask = jnp.sum((H * M) ** 2) > 0

    ## Null model
    mu = expit(offset)
    R = (Y - mu) * M
    W_sqrt = jnp.sqrt(mu * (1.0 - mu)) * M

    ## Score test for the joint ancestry-specific effects.
    U = G.T @ R
    Gw = G * M[:, None] * W_sqrt[:, None]
    GtG = Gw.T @ Gw
    beta_het, chisq_anc, chisq_het = het_score(U[:, None], GtG, G_mask)

    ## Score test for the common homogeneous effect.
    UH = H.T @ R
    Hw = H * M * W_sqrt
    HtH = Hw.T @ Hw
    beta_hom, chisq_hom = hom_score(UH, HtH)
    if not perform_diff:
        chisq_diff = jnp.broadcast_to(jnp.nan, chisq_hom.shape)
        result = mask_result(
            beta_het[:, 0],
            beta_hom,
            chisq_anc[:, 0],
            chisq_het,
            chisq_hom,
            chisq_diff,
            G_mask,
            H_mask,
        )
        return (*result, jnp.asarray(jnp.nan))
    H_design = H[:, None]
    H_mask = jnp.atleast_1d(H_mask)
    beta_hom_null, converged_hom = logistic_with_convergence(
        H_design, Y, offset, M, H_mask, tol=1e-5
    )
    mu_hom = expit(H_design @ beta_hom_null + offset)
    R_hom = (Y - mu_hom) * M
    W_hom = jnp.sqrt(mu_hom * (1.0 - mu_hom)) * M
    G_diff = G - H_design * ((G * W_hom[:, None] ** 2).T @ H / jnp.sum((H * W_hom) ** 2)).T
    G_diff = G_diff[:, :-1]
    G_diff_w = G_diff * W_hom[:, None]
    Q_diff, R_diff = jnp.linalg.qr(G_diff_w, mode="reduced")
    rank_tol = (
        jnp.finfo(G_diff_w.dtype).eps
        * max(G_diff_w.shape)
        * jnp.max(jnp.abs(jnp.diagonal(R_diff)), initial=0)
    )
    diff_rank = jnp.abs(jnp.diagonal(R_diff)) > rank_tol
    score_design = Q_diff / jnp.where(W_hom[:, None] > 0, W_hom[:, None], 1)
    chisq_diff = jnp.sum((score_design.T @ R_hom) ** 2 * diff_rank)
    chisq_diff = jnp.where(jnp.sum(diff_rank) > 0, chisq_diff, jnp.nan)

    result = mask_result(
        beta_het[:, 0],
        beta_hom,
        chisq_anc[:, 0],
        chisq_het,
        chisq_hom,
        chisq_diff,
        G_mask,
        H_mask,
    )
    return (*result, converged_hom)


def _bt_wald_lanc(
    G: Array, L: Array, Y: Array, Q: Array, offset: Array, M: Array
) -> tuple[Array, ...]:
    K = G.shape[1]

    ## Get H and residualize all by covariates
    G, L, H = prep_lanc_geno(G, L, Q)

    ## Fit G,H ~ L and mask out collinear columns
    QL, G, _, G_mask, H, _, H_mask = adj_by_lanc(G, H, L, M)
    H = H[:, None]
    L_mask = jnp.sum(QL**2, axis=0) > 0
    L = L * L_mask

    ## Wald test for the joint ancestry-specific effects.
    Xg_mask = jnp.concatenate([G_mask, L_mask])
    Xg = jnp.concatenate([G, L], axis=1)
    beta_het, converged_het = logistic_with_convergence(Xg, Y, offset, M, Xg_mask, tol=1e-5)
    etag = Xg @ beta_het + offset
    mu = expit(etag)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Xw = Xg * W_sqrt[:, None] * M[:, None]
    XtX = Xw.T @ Xw
    XtXw_inv = masked_inv(XtX, Xg_mask)
    chisq_anc = beta_het[:K] ** 2 / jnp.diagonal(XtXw_inv[:K, :K])
    chisq_het = beta_het[:K].T @ solve(XtXw_inv[:K, :K], beta_het[:K])

    ## Wald test for the common homogeneous effect.
    Xh_mask = jnp.concatenate([H_mask[None], L_mask])
    Xh = jnp.concatenate([H, L], axis=1)
    beta_hom, converged_hom = logistic_with_convergence(Xh, Y, offset, M, Xh_mask, tol=1e-5)
    etah = Xh @ beta_hom + offset
    mu = expit(etah)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Xw = Xh * W_sqrt[:, None] * M[:, None]
    XtX = Xw.T @ Xw
    XtXw_inv = masked_inv(XtX, Xh_mask)
    chisq_hom = beta_hom[0] ** 2 / XtXw_inv[0, 0]

    ## LRT
    l_het = (Y * etag - softplus(etag)) * M
    l_hom = (Y * etah - softplus(etah)) * M
    chisq_diff = 2 * jnp.sum(l_het - l_hom)

    result = mask_result(
        beta_het[:K],
        beta_hom[0],
        chisq_anc,
        chisq_het,
        chisq_hom,
        chisq_diff,
        G_mask,
        H_mask,
    )
    return (*result, converged_het & converged_hom)


def _bt_wald_nolanc(G: Array, Y: Array, Q: Array, offset: Array, M: Array) -> tuple[Array, ...]:
    K = G.shape[1]

    ## Get H and residualize all by covariates
    G, H = prep_geno(G, Q)
    H = H[:, None]

    ## Wald test for the joint ancestry-specific effects.
    G_mask = jnp.sum((G * M[:, None]) ** 2, axis=0) > 0
    beta_het, converged_het = logistic_with_convergence(G, Y, offset, M, G_mask, tol=1e-5)
    etag = G @ beta_het + offset
    mu = expit(etag)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Gw = G * W_sqrt[:, None] * M[:, None]
    GtG = Gw.T @ Gw
    GtGw_inv = masked_inv(GtG, G_mask)
    chisq_anc = beta_het[:K] ** 2 / jnp.diagonal(GtGw_inv[:K, :K])
    chisq_het = beta_het[:K].T @ solve(GtGw_inv[:K, :K], beta_het[:K])

    ## Wald test for the common homogeneous effect.
    H_mask = jnp.sum((H * M[:, None]) ** 2, axis=0) > 0
    beta_hom, converged_hom = logistic_with_convergence(H, Y, offset, M, H_mask, tol=1e-5)
    etah = H @ beta_hom + offset
    mu = expit(etah)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Hw = H * W_sqrt[:, None] * M[:, None]
    HtH = Hw.T @ Hw
    chisq_hom = beta_hom**2 * HtH

    ## LRT
    l_het = (Y * etag - softplus(etag)) * M
    l_hom = (Y * etah - softplus(etah)) * M
    chisq_diff = 2 * jnp.sum(l_het - l_hom)

    result = mask_result(
        beta_het[:K],
        beta_hom[0],
        chisq_anc,
        chisq_het,
        chisq_hom,
        chisq_diff,
        G_mask,
        H_mask,
    )
    return (*result, converged_het & converged_hom)


### ─────────────────────────────────────────────────────────────
### Block-wise functions
### ─────────────────────────────────────────────────────────────

bt_score_lanc = make_blockwise(
    _bt_score_lanc,
    (1, 1, None, None, None, None),
    (3, 3, 1, 2, 1, 1),
)
bt_score_lanc_no_diff = make_blockwise(
    partial(_bt_score_lanc, perform_diff=False),
    (1, 1, None, None, None, None),
    (3, 3, 1, 2, 1, 1),
)

bt_score_nolanc = make_blockwise(
    _bt_score_nolanc,
    (1, None, None, None, None),
    (3, 1, 2, 1, 1),
)
bt_score_nolanc_no_diff = make_blockwise(
    partial(_bt_score_nolanc, perform_diff=False),
    (1, None, None, None, None),
    (3, 1, 2, 1, 1),
)

bt_wald_lanc = make_blockwise(
    _bt_wald_lanc,
    (1, 1, None, None, None, None),
    (3, 3, 1, 2, 1, 1),
)

bt_wald_nolanc = make_blockwise(
    _bt_wald_nolanc,
    (1, None, None, None, None),
    (3, 1, 2, 1, 1),
)

__all__ = ["bt_score_lanc", "bt_score_nolanc", "bt_wald_lanc", "bt_wald_nolanc"]
