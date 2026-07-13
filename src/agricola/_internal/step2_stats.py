# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Single variant associations.

This module contains functions for calculating the tests statistics used in agricola step 2
"""

from jax import jit, vmap
import jax.lax as lax
import jax.numpy as jnp
from jax.numpy.linalg import inv, qr, solve
from jaxtyping import Array
from jax.scipy.special import expit

### ─────────────────────────────────────────────────────────────
### Helpers
### ─────────────────────────────────────────────────────────────


def _naninf_to_0(X: Array) -> Array:
    return jnp.nan_to_num(X, posinf=0, neginf=0)


def _logistic(
    X: Array,
    y: Array,
    offset: Array,
    train_mask: Array,
    X_mask: Array,
    max_iter: int = 10,
) -> Array:
    beta0 = jnp.zeros(X.shape[1])

    def _body_fun(i, beta):
        eta = X @ beta + offset
        mu = expit(eta)
        r = (y - mu) * train_mask
        w = mu * (1 - mu) * train_mask
        XW = X * w[:, None]
        XT_r = X.T @ r
        delta = _masked_solve(X.T @ XW, X_mask, XT_r)
        beta_new = beta + delta
        return _naninf_to_0(beta_new)

    beta = lax.fori_loop(0, max_iter, _body_fun, beta0)

    return beta


def _qr_resid(X: Array, Q: Array) -> Array:
    return X - Q @ (Q.T @ X)


def _project_and_mask_collinear(X: Array, QL: Array) -> tuple[Array, Array, Array]:
    Xl = _qr_resid(X, QL)
    Xl_norm = jnp.sum(Xl**2, axis=0)
    X_norm = jnp.sum(X**2, axis=0)
    r2_Xl = 1 - Xl_norm / X_norm
    X_mask = r2_Xl < 0.99
    X_mask_shaped = jnp.reshape(X_mask, X_mask.shape + (1,) * (X.ndim - 1)).T
    Xl = Xl * X_mask_shaped
    X = X * X_mask_shaped

    return X, Xl, X_mask


def _masked_nan(X: Array, mask: Array) -> Array:
    return jnp.where(mask, X, jnp.nan)


def _masked_inv(A: Array, mask: Array) -> Array:
    return inv(A + jnp.diag((~mask).astype(A.dtype)))


def _masked_solve(A: Array, mask: Array, x: Array) -> Array:
    return solve(A + jnp.diag((~mask).astype(A.dtype)), x)


### ─────────────────────────────────────────────────────────────
### Quantitative Traits
### ─────────────────────────────────────────────────────────────


def _qt_score_lanc(
    G: Array, L: Array, Y: Array, Q: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    K = G.shape[1]

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = _qr_resid(G, Q)
    L = _qr_resid(L, Q)
    H = _qr_resid(H, Q)

    ## Fit null model: Y ~ L
    QL, _ = qr(L, mode="reduced")
    r_L = _qr_resid(Y, QL)
    mse_null = jnp.sum(r_L**2, axis=0) / (N_eff - (K - 1))

    ## Fit G,H ~ L and mask out collinear columns
    G, Gl, G_mask = _project_and_mask_collinear(G, QL)
    H, Hl, H_mask = _project_and_mask_collinear(H, QL)

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ r_L
    GltGl_inv = _masked_inv(Gl.T @ Gl, G_mask)
    chisq_het = jnp.einsum("kp,kl,lp->p", U, GltGl_inv, U) / mse_null
    beta_het = U * jnp.diagonal(GltGl_inv)[:, None]
    df_het = jnp.sum(G_mask)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ r_L
    HtH = jnp.sum(Hl**2)
    beta_hom = UH / (HtH)
    chisq_hom = (UH**2) / HtH / mse_null

    ## set low variation dimensions to nan
    beta_het = _masked_nan(beta_het, G_mask[:, None])
    beta_hom = _masked_nan(beta_hom, H_mask)

    return chisq_hom, beta_hom, chisq_het, beta_het, df_het


def _qt_score_nolanc(
    G: Array, Y: Array, Q: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = _qr_resid(G, Q)
    H = _qr_resid(H, Q)

    mse_null = jnp.sum(Y**2, axis=0) / N_eff

    ## Mask out low variation columns
    G_mask = jnp.sum(G**2, axis=0) > 0
    G = G * G_mask[None, :]
    H_mask = jnp.sum(H**2, axis=0) > 0
    H = H * H_mask

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ Y
    GtG_inv = _masked_inv(G.T @ G, G_mask)
    chisq_het = jnp.einsum("kp,kl,lp->p", U, GtG_inv, U) / mse_null
    beta_het = U * jnp.diagonal(GtG_inv)[:, None]
    df_het = jnp.sum(G_mask)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ Y
    HtH = jnp.sum(H**2)
    chisq_hom = (UH**2) / HtH / mse_null
    beta_hom = UH / (HtH)

    ## set low variation dimensions to nan
    beta_het = _masked_nan(beta_het, G_mask[:, None])
    beta_hom = _masked_nan(beta_hom, H_mask)

    return chisq_hom, beta_hom, chisq_het, beta_het, df_het


def _qt_wald_lanc(
    G: Array, L: Array, Y: Array, Q: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    K = G.shape[1]

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = _qr_resid(G, Q)
    L = _qr_resid(L, Q)
    H = _qr_resid(H, Q)

    ## Fit null model: Y ~ L
    QL, _ = qr(L, mode="reduced")
    r_L = _qr_resid(Y, QL)

    ## Fit G,H ~ L and mask out collinear columns
    G, Gl, G_mask = _project_and_mask_collinear(G, QL)
    H, Hl, H_mask = _project_and_mask_collinear(H, QL)
    H = H[:, None]
    Hl = Hl[:, None]

    ## Wald test for anc-deconvoluted genotypes (heterogeneous test)
    Gtr = Gl.T @ r_L
    GltGl_inv = _masked_inv(Gl.T @ Gl, G_mask)
    beta_het = GltGl_inv @ Gtr
    r_G = r_L - G @ beta_het
    sse_het = jnp.sum(r_G**2, axis=0)
    mse_het = sse_het / (N_eff - (2 * K - 1))
    chisq_het = jnp.einsum("kp,kl,lp->p", Gtr, GltGl_inv, Gtr) / mse_het
    df_het = jnp.sum(G_mask)

    ## Wald test for anc-deconvoluted genotypes (heterogeneous test)
    Htr = Hl.T @ r_L
    HltHl = Hl.T @ Hl
    beta_hom = Htr / HltHl
    r_H = r_L - (H @ beta_hom)
    sse_hom = jnp.sum(r_H**2, axis=0)
    mse_hom = sse_hom / (N_eff - K)
    chisq_hom = (Htr**2) / HltHl / mse_hom
    df_hom = jnp.sum(H_mask)

    chisq_lrt = N_eff * jnp.log(sse_hom / sse_het)
    df_lrt = df_het - df_hom

    ## set low variation dimensions to nan
    beta_het = _masked_nan(beta_het, G_mask[:, None])
    beta_hom = _masked_nan(beta_hom, H_mask)

    return chisq_hom, beta_hom, chisq_het, beta_het, df_het, chisq_lrt, df_lrt


def _qt_wald_nolanc(
    G: Array, Y: Array, Q: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    K = G.shape[1]

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = _qr_resid(G, Q)
    H = _qr_resid(H, Q)

    ## Mask out low variation columns
    G_mask = jnp.sum(G**2, axis=0) > 0
    G = G * G_mask[None, :]
    H_mask = jnp.sum(H**2, axis=0) > 0
    H = H * H_mask
    H = H[:, None]

    ## Wald test for anc-deconvoluted genotypes (heterogeneous test)
    Gtr = G.T @ Y
    GtG_inv = _masked_inv(G.T @ G, G_mask)
    beta_het = GtG_inv @ Gtr
    r_G = Y - G @ beta_het
    sse_het = jnp.sum(r_G**2, axis=0)
    mse_het = sse_het / (N_eff - K)
    chisq_het = jnp.einsum("kp,kl,lp->p", Gtr, GtG_inv, Gtr) / mse_het
    df_het = jnp.sum(G_mask)

    ## Wald test for anc-deconvoluted genotypes (heterogeneous test)
    Htr = H.T @ Y
    HtH = H.T @ H
    beta_hom = Htr / HtH
    r_H = Y - (H @ beta_hom)
    sse_hom = jnp.sum(r_H**2, axis=0)
    mse_hom = sse_hom / (N_eff - 1)
    chisq_hom = (Htr**2) / HtH / mse_hom
    df_hom = jnp.sum(H_mask)

    chisq_lrt = N_eff * jnp.log(sse_hom / sse_het)
    df_lrt = df_het - df_hom

    ## set low variation dimensions to nan
    beta_het = _masked_nan(beta_het, G_mask[:, None])
    beta_hom = _masked_nan(beta_hom, H_mask)

    return chisq_hom, beta_hom, chisq_het, beta_het, df_het, chisq_lrt, df_lrt


### ─────────────────────────────────────────────────────────────
### Binary Traits
### ─────────────────────────────────────────────────────────────


def _bt_score_lanc(
    G: Array, L: Array, Y: Array, Q: Array, O: Array, M: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array]:
    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = _qr_resid(G, Q)
    L = _qr_resid(L, Q)
    H = _qr_resid(H, Q)

    ## Fit G,H ~ L and mask out collinear columns
    QL, _ = qr(L, mode="reduced")
    G, Gl, G_mask = _project_and_mask_collinear(G * M[:, None], QL)
    H, Hl, H_mask = _project_and_mask_collinear(H * M, QL)

    ## Fit null model L + offset
    L_mask = jnp.sum(L**2, axis=0) > 0
    beta_L = _logistic(L, Y, O, M, L_mask)
    mu = expit(L @ beta_L + O)
    R = (Y - mu) * M
    W_L_sqrt = jnp.sqrt(mu * (1.0 - mu)) * M

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ R
    Glw = Gl * M[:, None] * W_L_sqrt[:, None]
    GltGl = Glw.T @ Glw
    GltGl_inv = _masked_inv(GltGl, G_mask)
    chisq_het = U.T @ GltGl_inv @ U
    beta_het = U * jnp.diagonal(GltGl_inv)
    df_het = jnp.sum(G_mask)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ R
    Hlw = Hl * M * W_L_sqrt
    HtH = Hlw.T @ Hlw
    beta_hom = UH / HtH
    chisq_hom = (UH**2) / HtH

    ## set low variation dimensions to nan
    beta_het = _masked_nan(beta_het, G_mask)
    beta_hom = _masked_nan(beta_hom, H_mask)

    return chisq_hom, beta_hom, chisq_het, beta_het, df_het


def _bt_score_nolanc(
    G: Array, Y: Array, Q: Array, O: Array, M: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array]:
    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = _qr_resid(G, Q)
    H = _qr_resid(H, Q)

    ## Mask out low variation columns
    G_mask = jnp.sum(G**2, axis=0) > 0
    H_mask = jnp.sum(H**2, axis=0) > 0

    ## Null model
    mu = expit(O)
    R = (Y - mu) * M
    W_sqrt = jnp.sqrt(mu * (1.0 - mu)) * M

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ R
    Gw = G * M[:, None] * W_sqrt[:, None]
    GtG = Gw.T @ Gw
    GtG_inv = _masked_inv(GtG, G_mask)
    chisq_het = U.T @ GtG_inv @ U
    beta_het = U * jnp.diagonal(GtG_inv)
    df_het = jnp.sum(G_mask)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ R
    Hw = H * M * W_sqrt
    HtH = Hw.T @ Hw
    beta_hom = UH / HtH
    chisq_hom = (UH**2) / HtH

    ## set low variation dimensions to nan
    beta_het = _masked_nan(beta_het, G_mask)
    beta_hom = _masked_nan(beta_hom, H_mask)

    return chisq_hom, beta_hom, chisq_het, beta_het, df_het


def _bt_wald_lanc(
    G: Array, L: Array, Y: Array, Q: Array, O: Array, M: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    K = G.shape[1]

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = _qr_resid(G, Q)
    L = _qr_resid(L, Q)
    H = _qr_resid(H, Q)

    ## Fit G,H ~ L and mask out collinear columns
    QL, _ = qr(L * M[:, None], mode="reduced")
    G, _, G_mask = _project_and_mask_collinear(G * M[:, None], QL)
    H, _, H_mask = _project_and_mask_collinear(H * M, QL)
    L_mask = jnp.sum(L**2, axis=0) > 0

    ## Wald test for anc-deconvoluted genotypes
    Xg_mask = jnp.concatenate([G_mask, L_mask])
    Xg = jnp.concatenate([G, L], axis=1)
    beta_het = _logistic(Xg, Y, O, M, Xg_mask)
    etag = Xg @ beta_het + O
    mu = expit(etag)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Xw = Xg * W_sqrt[:, None] * M[:, None]
    XtX = Xw.T @ Xw
    XtXw_inv = _masked_inv(XtX, Xg_mask)
    chisq_het = beta_het[:K].T @ solve(XtXw_inv[:K, :K], beta_het[:K])
    df_het = jnp.sum(G_mask)

    ## Wald test for genotypes
    Xh_mask = jnp.concatenate([H_mask[None], L_mask])
    Xh = jnp.concatenate([H[:, None], L], axis=1)
    beta_hom = _logistic(Xh, Y, O, M, Xh_mask)
    etah = Xh @ beta_hom + O
    mu = expit(etah)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Xw = Xh * W_sqrt[:, None] * M[:, None]
    XtX = Xw.T @ Xw
    XtXw_inv = _masked_inv(XtX, Xh_mask)
    chisq_hom = beta_hom[0] ** 2 / XtXw_inv[0, 0]
    df_hom = jnp.sum(H_mask)

    ## LRT
    l_het = (Y * etag - jnp.log(1 + jnp.exp(etag))) * M
    l_hom = (Y * etah - jnp.log(1 + jnp.exp(etah))) * M
    chisq_lrt = 2 * jnp.sum(l_het - l_hom)
    df_lrt = df_het - df_hom

    ## set low variation dimensions to nan
    beta_het = _masked_nan(beta_het, Xg_mask)
    beta_hom = _masked_nan(beta_hom, Xh_mask)

    return chisq_hom, beta_hom[0], chisq_het, beta_het[:K], df_het, chisq_lrt, df_lrt


def _bt_wald_nolanc(
    G: Array, Y: Array, Q: Array, O: Array, M: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    K = G.shape[1]

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = _qr_resid(G, Q)
    H = _qr_resid(H, Q)
    H = H[:, None]

    ## Wald test for anc-deconvoluted genotypes
    G_mask = jnp.sum(G**2, axis=0) > 0
    beta_het = _logistic(G, Y, O, M, G_mask)
    etag = G @ beta_het + O
    mu = expit(etag)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Gw = G * W_sqrt[:, None] * M[:, None]
    GtG = Gw.T @ Gw
    GtGw_inv = _masked_inv(GtG, G_mask)
    chisq_het = beta_het[:K].T @ solve(GtGw_inv[:K, :K], beta_het[:K])
    df_het = jnp.sum(G_mask)

    ## Wald test for genotypes
    H_mask = jnp.sum(H**2, axis=0) > 0
    beta_hom = _logistic(H, Y, O, M, H_mask)
    etah = H @ beta_hom + O
    mu = expit(etah)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Hw = H * W_sqrt[:, None] * M[:, None]
    HtH = H.T @ Hw
    HtHw_inv = _masked_inv(HtH, H_mask)
    chisq_hom = beta_hom[0] ** 2 / HtHw_inv[0, 0]
    df_hom = jnp.sum(H_mask)

    ## LRT
    l_het = (Y * etag - jnp.log(1 + jnp.exp(etag))) * M
    l_hom = (Y * etah - jnp.log(1 + jnp.exp(etah))) * M
    chisq_lrt = 2 * jnp.sum(l_het - l_hom)
    df_lrt = df_het - df_hom

    ## set low variation dimensions to nan
    beta_het = _masked_nan(beta_het, G_mask)
    beta_hom = _masked_nan(beta_hom, H_mask)

    return chisq_hom, beta_hom, chisq_het, beta_het, df_het, chisq_lrt, df_lrt


### ─────────────────────────────────────────────────────────────
### Block-wise functions
### ─────────────────────────────────────────────────────────────

qt_score_lanc = jit(
    vmap(
        vmap(_qt_score_lanc, in_axes=(1, 1, None, None, None)),
        in_axes=(3, 3, 1, 2, 0),
        out_axes=-1,
    )
)

qt_score_lanc_impute = jit(
    vmap(_qt_score_lanc, in_axes=(1, 1, None, None, None)),
)

qt_score_nolanc = jit(
    vmap(
        vmap(_qt_score_nolanc, in_axes=(1, None, None, None)),
        in_axes=(3, 1, 2, 0),
        out_axes=-1,
    )
)

qt_score_nolanc_impute = jit(
    vmap(_qt_score_nolanc, in_axes=(1, None, None, None)),
)

qt_wald_lanc = jit(
    vmap(
        vmap(_qt_wald_lanc, in_axes=(1, 1, None, None, None)),
        in_axes=(3, 3, 1, 2, 0),
        out_axes=-1,
    )
)

qt_wald_lanc_impute = jit(
    vmap(_qt_wald_lanc, in_axes=(1, 1, None, None, None)),
)

qt_wald_nolanc = jit(
    vmap(
        vmap(_qt_wald_nolanc, in_axes=(1, None, None, None)),
        in_axes=(3, 1, 2, 0),
        out_axes=-1,
    )
)

qt_wald_nolanc_impute = jit(
    vmap(_qt_wald_nolanc, in_axes=(1, None, None, None)),
)

bt_score_lanc = jit(
    vmap(
        vmap(_bt_score_lanc, in_axes=(1, 1, None, None, None, None, None)),
        in_axes=(3, 3, 1, 2, 1, 1, 0),
        out_axes=-1,
    )
)

bt_score_nolanc = jit(
    vmap(
        vmap(_bt_score_nolanc, in_axes=(1, None, None, None, None, None)),
        in_axes=(3, 1, 2, 1, 1, 0),
        out_axes=-1,
    )
)


bt_wald_lanc = jit(
    vmap(
        vmap(_bt_wald_lanc, in_axes=(1, 1, None, None, None, None, None)),
        in_axes=(3, 3, 1, 2, 1, 1, 0),
        out_axes=-1,
    )
)

bt_wald_nolanc = jit(
    vmap(
        vmap(_bt_wald_nolanc, in_axes=(1, None, None, None, None, None)),
        in_axes=(3, 1, 2, 1, 1, 0),
        out_axes=-1,
    )
)
