# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Single variant associations.

This module contains functions for calculating the tests statistics used in agricola step 2
"""

from jax import jit, vmap
import jax.numpy as jnp
from jax.numpy.linalg import inv, matrix_rank
from jaxtyping import Array
from jax.scipy.special import expit
from .models import logistic_ridge

### ─────────────────────────────────────────────────────────────
### Helpers
### ─────────────────────────────────────────────────────────────


def ols_block(
    X: Array, L: Array, Y: Array, N_eff: Array, alpha: Array
) -> tuple[Array, Array, Array, Array]:
    XtX = X.T @ X
    XtX_inv = inv(XtX + alpha * jnp.identity(X.shape[1]))
    XtL = X.T @ L
    LtL = L.T @ L
    XtXl_inv22 = inv(LtL - XtL.T @ XtX_inv @ XtL + alpha * jnp.identity(L.shape[1]))
    XtXl_inv12 = -XtX_inv @ XtL @ XtXl_inv22
    XtXl_inv11 = XtX_inv - (XtXl_inv12 @ XtL.T @ XtX_inv)
    XtY = X.T @ Y
    LtY = L.T @ Y
    betax = XtXl_inv11 @ XtY + XtXl_inv12 @ LtY
    betal = XtXl_inv12.T @ XtY + XtXl_inv22 @ LtY
    res = Y - X @ betax - L @ betal
    sse = jnp.sum(res**2, axis=0)
    mse = sse / (N_eff - X.shape[1] - L.shape[1])
    stat = jnp.einsum("kp,kl,lp->p", betax, inv(XtXl_inv11), betax) / mse
    df = matrix_rank(XtX)
    return stat, betax, df, sse


def ols(
    X: Array, Y: Array, N_eff: Array, alpha: Array
) -> tuple[Array, Array, Array, Array]:
    XtX = X.T @ X
    XtY = X.T @ Y
    XtX_inv = inv(XtX + alpha * jnp.identity(X.shape[1]))
    betax = XtX_inv @ XtY
    res = Y - X @ betax
    sse = res.T @ res
    mse = sse / (N_eff - X.shape[1])
    stat = jnp.einsum("kp,kl,lp->p", betax, XtX, betax) / mse
    df = matrix_rank(XtX)
    return stat, betax, df, sse


### ─────────────────────────────────────────────────────────────
### Quantitative Traits
### ─────────────────────────────────────────────────────────────


def _qt_score_lanc(
    G: Array, L: Array, Y: Array, Q: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    K = G.shape[1]
    alpha = N_eff * 1e-6

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = G - Q @ (Q.T @ G)
    L = L - Q @ (Q.T @ L)
    H = H - Q @ (Q.T @ H)

    LtL_inv = jnp.linalg.inv(L.T @ L + alpha * jnp.identity(K - 1))

    ## Fit null model: Y ~ L
    LtY = L.T @ Y
    r_L = Y - L @ (LtL_inv @ LtY)
    mse_null = jnp.sum(r_L**2, axis=0) / (N_eff - (K - 1))

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    LtG = L.T @ G
    Gl = G - L @ (LtL_inv @ LtG)
    U = G.T @ r_L
    GtG = Gl.T @ Gl
    GtG_inv = inv(GtG + alpha * jnp.identity(K))
    chisq_het = jnp.einsum("kp,kl,lp->p", U, GtG_inv, U) / mse_null
    beta_het = U * jnp.diagonal(GtG_inv)[:, None]
    df_het = matrix_rank(GtG)

    ## Score test for genotypes (homogeneous test)
    LtH = L.T @ H
    Hl = H - L @ (LtL_inv @ LtH)
    UH = H.T @ r_L
    HtH = jnp.sum(Hl**2)
    beta_hom = UH / (HtH + alpha)
    chisq_hom = (UH**2) / HtH / mse_null

    return chisq_hom, beta_hom, chisq_het, beta_het, df_het


def _qt_score_nolanc(
    G: Array, Y: Array, Q: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    K = G.shape[1]
    alpha = N_eff * 1e-6

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = G - Q @ (Q.T @ G)
    H = H - Q @ (Q.T @ H)

    mse_null = jnp.sum(Y**2, axis=0) / N_eff

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ Y
    GtG = G.T @ G
    GtG_inv = inv(GtG + alpha * jnp.identity(K))
    chisq_het = jnp.einsum("kp,kl,lp->p", U, GtG_inv, U) / mse_null
    beta_het = U * jnp.diagonal(GtG_inv)[:, None]
    df_het = matrix_rank(GtG)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ Y
    HtH = jnp.sum(H**2)
    chisq_hom = (UH**2) / HtH / mse_null
    beta_hom = UH / (HtH + alpha)

    return chisq_hom, beta_hom, chisq_het, beta_het, df_het


def _qt_wald_lanc(
    G: Array, L: Array, Y: Array, Q: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    alpha = N_eff * 1e-6

    ## Genotypes
    H = jnp.sum(G, axis=1, keepdims=True)

    ## Residualize by covariates
    G = G - Q @ (Q.T @ G)
    L = L - Q @ (Q.T @ L)
    H = H - Q @ (Q.T @ H)

    ## Tests
    chisq_het, beta_het, df_het, sse_het = ols_block(G, L, Y, N_eff, alpha)
    chisq_hom, beta_hom, df_hom, sse_hom = ols_block(H, L, Y, N_eff, alpha)
    chisq_lrt = N_eff * jnp.log(sse_hom / sse_het)
    df_lrt = df_het - df_hom

    return chisq_hom, beta_hom[0, :], chisq_het, beta_het, df_het, chisq_lrt, df_lrt


def _qt_wald_nolanc(
    G: Array, Y: Array, Q: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    alpha = N_eff * 1e-6

    ## Genotypes
    H = jnp.sum(G, axis=1, keepdims=True)

    ## Residualize by covariates
    G = G - Q @ (Q.T @ G)
    H = H - Q @ (Q.T @ H)

    ## Tests
    chisq_het, beta_het, df_het, sse_het = ols(G, Y, N_eff, alpha)
    chisq_hom, beta_hom, df_hom, sse_hom = ols(H, Y, N_eff, alpha)
    chisq_lrt = N_eff * jnp.log(sse_hom / sse_het)
    df_lrt = df_het - df_hom

    return chisq_hom, beta_hom[0, :], chisq_het, beta_het, df_het, chisq_lrt, df_lrt


### ─────────────────────────────────────────────────────────────
### Binary Traits
### ─────────────────────────────────────────────────────────────


def _bt_score_lanc(
    G: Array, L: Array, Y: Array, Q: Array, O: Array, M: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array]:
    alpha = N_eff * 1e-6
    K = G.shape[1]

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = G - Q @ (Q.T @ G)
    L = L - Q @ (Q.T @ L)
    H = H - Q @ (Q.T @ H)

    ## Fit null model L + offset
    beta_L = logistic_ridge(L, Y, O, M, alpha, 10)
    mu = expit(L @ beta_L + O)
    R = (Y - mu) * M
    W_L_sqrt = jnp.sqrt(mu * (1.0 - mu)) * M

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ R
    LtL_inv = jnp.linalg.inv(L.T @ L + alpha * jnp.identity(K - 1))
    LtG = L.T @ G
    Glw = (G - L @ (LtL_inv @ LtG)) * M[:, None] * W_L_sqrt[:, None]
    GtG = Glw.T @ Glw
    GtG_inv = inv(GtG + alpha * jnp.identity(K))
    chisq_het = U.T @ GtG_inv @ U
    beta_het = U * jnp.diagonal(GtG_inv)
    df_het = matrix_rank(GtG)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ R
    LtH = L.T @ H
    Hlw = (H - L @ (LtL_inv @ LtH)) * M * W_L_sqrt
    HtH = Hlw.T @ Hlw
    beta_hom = UH / (HtH + alpha)
    chisq_hom = (UH**2) / HtH

    return chisq_hom, beta_hom, chisq_het, beta_het, df_het


def _bt_score_nolanc(
    G: Array, Y: Array, Q: Array, O: Array, M: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array]:
    alpha = N_eff * 1e-6
    K = G.shape[1]

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = G - Q @ (Q.T @ G)
    H = H - Q @ (Q.T @ H)

    mu = expit(O)
    R = (Y - mu) * M
    W_sqrt = jnp.sqrt(mu * (1.0 - mu)) * M

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ R
    Gw = G * M[:, None] * W_sqrt[:, None]
    GtG = Gw.T @ Gw
    GtG_inv = inv(GtG + alpha * jnp.identity(K))
    chisq_het = U.T @ GtG_inv @ U
    beta_het = U * jnp.diagonal(GtG_inv)
    df_het = matrix_rank(GtG)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ R
    Hw = H * M * W_sqrt
    HtH = Hw.T @ Hw
    beta_hom = UH / (HtH + alpha)
    chisq_hom = (UH**2) / HtH

    return chisq_hom, beta_hom, chisq_het, beta_het, df_het


def _bt_wald_lanc(
    G: Array, L: Array, Y: Array, Q: Array, O: Array, M: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    alpha = N_eff * 1e-6
    K = G.shape[1]

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = G - Q @ (Q.T @ G)
    L = L - Q @ (Q.T @ L)
    H = H - Q @ (Q.T @ H)

    ## Wald test for anc-deconvoluted genotypes
    Xg = jnp.concatenate([G, L], axis=1)
    betag = logistic_ridge(Xg, Y, O, M, alpha, 10)
    etag = Xg @ betag + O
    mu = expit(etag)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Xw = Xg * W_sqrt[:, None] * M[:, None]
    XtX = Xw.T @ Xw
    XtXw_inv = inv(XtX + alpha * jnp.identity(2 * K - 1))
    chisq_het = betag[:K].T @ inv(XtXw_inv[:K, :K]) @ betag[:K]
    df_het = matrix_rank(XtX[:K, :K])

    ## Wald test for genotypes
    Xh = jnp.concatenate([H[:, None], L], axis=1)
    betah = logistic_ridge(Xh, Y, O, M, alpha, 10)
    etah = Xh @ betah + O
    mu = expit(etah)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Xw = Xh * W_sqrt[:, None] * M[:, None]
    XtX = Xw.T @ Xw
    XtXw_inv = inv(XtX + alpha * jnp.identity(K))
    chisq_hom = betah[0] ** 2 / XtXw_inv[0, 0]
    df_hom = matrix_rank(XtX[0, 0])

    ## LRT
    l_het = Y * etag - jnp.log(1 + jnp.exp(etag))
    l_hom = Y * etah - jnp.log(1 + jnp.exp(etah))
    chisq_lrt = 2 * jnp.sum(l_het - l_hom)
    df_lrt = df_het - df_hom

    return chisq_hom, betah[0], chisq_het, betag[:K], df_het, chisq_lrt, df_lrt


def _bt_wald_nolanc(
    G: Array, Y: Array, Q: Array, O: Array, M: Array, N_eff: Array
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    alpha = N_eff * 1e-6

    ## Genotypes
    H = jnp.sum(G, axis=1)

    ## Residualize by covariates
    G = G - Q @ (Q.T @ G)
    H = H - Q @ (Q.T @ H)

    ## Wald test for anc-deconvoluted genotypes
    betag = logistic_ridge(G, Y, O, M, alpha, 10)
    etag = G @ betag + O
    mu = expit(etag)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Gw = G * W_sqrt[:, None] * M[:, None]
    GtG = Gw.T @ Gw
    chisq_het = betag.T @ GtG @ betag
    df_het = matrix_rank(GtG)

    ## Wald test for genotypes
    betah = logistic_ridge(H[:, None], Y, O, M, alpha, 10)[0]
    etah = H * betah + O
    mu = expit(etah)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Hw = H * W_sqrt * M
    HtH = Hw.T @ Hw
    chisq_hom = betah**2 * HtH
    df_hom = matrix_rank(HtH)

    ## LRT
    l_het = Y * etag - jnp.log(1 + jnp.exp(etag))
    l_hom = Y * etah - jnp.log(1 + jnp.exp(etah))
    chisq_lrt = 2 * jnp.sum(l_het - l_hom)
    df_lrt = df_het - df_hom

    return chisq_hom, betah, chisq_het, betag, df_het, chisq_lrt, df_lrt


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
