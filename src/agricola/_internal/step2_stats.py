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
    tol: float = 1e-6,
) -> Array:
    beta0 = jnp.zeros(X.shape[1])

    def _body_fun(state):
        i, beta = state
        eta = X @ beta + offset
        mu = expit(eta)
        r = (y - mu) * train_mask
        w = mu * (1 - mu) * train_mask
        XW = X * w[:, None]
        XT_r = X.T @ r
        delta = _masked_solve(X.T @ XW, X_mask, XT_r)
        beta_new = _naninf_to_0(beta + delta)
        return i + 1, beta_new

    def _cond_fun(state):
        i, beta = state
        _, beta_new = _body_fun(state)
        return (i < max_iter) & (jnp.linalg.norm(beta_new - beta) > tol)

    _, beta = lax.while_loop(_cond_fun, _body_fun, (0, beta0))

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


def _prep_geno(G: Array, Q: Array) -> tuple[Array, Array]:
    H = jnp.sum(G, axis=1)
    return _qr_resid(G, Q), _qr_resid(H, Q)


def _prep_lanc_geno(G: Array, L: Array, Q: Array) -> tuple[Array, Array, Array]:
    H = jnp.sum(G, axis=1)
    return (_qr_resid(G, Q), _qr_resid(L, Q), _qr_resid(H, Q))


def _adj_by_lanc(
    G: Array, H: Array, L: Array
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    QL, _ = qr(L, mode="reduced")
    G, Gl, G_mask = _project_and_mask_collinear(G, QL)
    H, Hl, H_mask = _project_and_mask_collinear(H, QL)
    return QL, G, Gl, G_mask, H, Hl, H_mask


def _het_score(
    U: Array, covariance: Array, mask: Array, scale: Array = jnp.array([1.0])
) -> tuple[Array, Array, Array, Array]:
    inv_cov = _masked_inv(covariance, mask)
    diag = jnp.diagonal(inv_cov)

    beta_het = U * diag[..., None]
    chisq_anc = U**2 * diag[..., None] / scale
    chisq_het = jnp.einsum("kp,kl,lp->p", U, inv_cov, U) / scale

    return beta_het, chisq_anc, chisq_het, jnp.sum(mask)


def _hom_score(UH: Array, HtH: Array, scale: Array = jnp.array([1.0])):
    beta_hom = UH / HtH
    chisq = UH**2 / HtH / scale
    return beta_hom, chisq


def _mask_score(
    beta_het: Array,
    beta_hom: Array,
    chisq_anc: Array,
    chisq_het: Array,
    chisq_hom: Array,
    G_mask: Array,
    H_mask: Array,
) -> tuple[Array, ...]:
    df_het = jnp.sum(G_mask)
    return (
        _masked_nan(chisq_hom, H_mask),
        _masked_nan(beta_hom, H_mask),
        _masked_nan(chisq_het, df_het != 0),
        _masked_nan(beta_het, G_mask),
        df_het,
        _masked_nan(chisq_anc, G_mask),
    )


def _mask_wald(
    beta_het: Array,
    beta_hom: Array,
    chisq_anc: Array,
    chisq_het: Array,
    chisq_hom: Array,
    chisq_lrt: Array,
    G_mask: Array,
    H_mask: Array,
) -> tuple[Array, ...]:
    df_het = jnp.sum(G_mask)
    df_hom = jnp.sum(H_mask)
    df_lrt = df_het - df_hom
    return (
        _masked_nan(chisq_hom, H_mask),
        _masked_nan(beta_hom, H_mask),
        _masked_nan(chisq_het, df_het != 0),
        _masked_nan(beta_het, G_mask),
        df_het,
        _masked_nan(chisq_anc, G_mask),
        _masked_nan(chisq_lrt, df_lrt != 0),
        df_lrt,
    )


### ─────────────────────────────────────────────────────────────
### Quantitative Traits
### ─────────────────────────────────────────────────────────────


def _qt_score_lanc(
    G: Array, L: Array, Y: Array, Q: Array, N_eff: Array
) -> tuple[Array, ...]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    ## Get H and residualize all by covariates
    G, L, H = _prep_lanc_geno(G, L, Q)

    ## Fit G,H ~ L and mask out collinear columns
    QL, G, Gl, G_mask, H, Hl, H_mask = _adj_by_lanc(G, H, L)

    ## Fit null model: Y ~ L
    r_L = _qr_resid(Y, QL)
    mse_null = jnp.sum(r_L**2, axis=0) / (N_eff - (G.shape[1] - 1))

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ r_L
    GltGl = Gl.T @ Gl
    beta_het, chisq_anc, chisq_het, df_het = _het_score(U, GltGl, G_mask, mse_null)

    ## Score test for genotypes (homogeneous test)
    beta_hom, chisq_hom = _hom_score(H.T @ r_L, jnp.sum(Hl**2), mse_null)

    return _mask_score(
        beta_het, beta_hom, chisq_anc, chisq_het, chisq_hom, G_mask[:, None], H_mask
    )


def _qt_score_nolanc(G: Array, Y: Array, Q: Array, N_eff: Array) -> tuple[Array, ...]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    ## Get H and residualize all by covariates
    G, H = _prep_geno(G, Q)

    mse_null = jnp.sum(Y**2, axis=0) / N_eff

    ## Mask out low variation columns
    G_mask = jnp.sum(G**2, axis=0) > 0
    G = G * G_mask[None, :]
    H_mask = jnp.sum(H**2, axis=0) > 0
    H = H * H_mask

    ## Score test for anc-deconvoluted genotypes (heterogeneous test)
    U = G.T @ Y
    GtG = G.T @ G
    beta_het, chisq_anc, chisq_het, df_het = _het_score(U, GtG, G_mask, mse_null)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ Y
    HtH = jnp.sum(H**2)
    beta_hom, chisq_hom = _hom_score(UH, HtH, mse_null)

    return _mask_score(
        beta_het, beta_hom, chisq_anc, chisq_het, chisq_hom, G_mask[:, None], H_mask
    )


def _qt_wald_lanc(
    G: Array, L: Array, Y: Array, Q: Array, N_eff: Array
) -> tuple[Array, ...]:
    Y = jnp.reshape(Y, Y.shape + (1,) * (2 - Y.ndim))

    K = G.shape[1]

    ## Get H and residualize all by covariates
    G, L, H = _prep_lanc_geno(G, L, Q)

    ## Fit G,H ~ L and mask out collinear columns
    QL, G, Gl, G_mask, H, Hl, H_mask = _adj_by_lanc(G, H, L)
    H = H[:, None]
    Hl = Hl[:, None]

    ## Fit null model: Y ~ L
    r_L = _qr_resid(Y, QL)

    ## Wald test for anc-deconvoluted genotypes (heterogeneous test)
    Gtr = Gl.T @ r_L
    GltGl_inv = _masked_inv(Gl.T @ Gl, G_mask)
    beta_het = GltGl_inv @ Gtr
    r_G = r_L - G @ beta_het
    sse_het = jnp.sum(r_G**2, axis=0)
    mse_het = sse_het / (N_eff - (2 * K - 1))
    chisq_anc = Gtr**2 * jnp.diagonal(GltGl_inv)[:, None] / mse_het
    chisq_het = jnp.einsum("kp,kl,lp->p", Gtr, GltGl_inv, Gtr) / mse_het

    ## Wald test for anc-deconvoluted genotypes (heterogeneous test)
    Htr = Hl.T @ r_L
    HltHl = Hl.T @ Hl
    beta_hom = Htr / HltHl
    r_H = r_L - (H @ beta_hom)
    sse_hom = jnp.sum(r_H**2, axis=0)
    mse_hom = sse_hom / (N_eff - K)
    chisq_hom = (Htr**2) / HltHl / mse_hom

    ## LRT
    chisq_lrt = N_eff * jnp.log(sse_hom / sse_het)

    return _mask_wald(
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
    G, H = _prep_geno(G, Q)

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
    chisq_anc = Gtr**2 * jnp.diagonal(GtG_inv)[:, None] / mse_het
    chisq_het = jnp.einsum("kp,kl,lp->p", Gtr, GtG_inv, Gtr) / mse_het

    ## Wald test for anc-deconvoluted genotypes (heterogeneous test)
    Htr = H.T @ Y
    HtH = H.T @ H
    beta_hom = Htr / HtH
    r_H = Y - (H @ beta_hom)
    sse_hom = jnp.sum(r_H**2, axis=0)
    mse_hom = sse_hom / (N_eff - 1)
    chisq_hom = (Htr**2) / HtH / mse_hom

    chisq_lrt = N_eff * jnp.log(sse_hom / sse_het)

    return _mask_wald(
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
### Binary Traits
### ─────────────────────────────────────────────────────────────


def _bt_score_lanc(
    G: Array, L: Array, Y: Array, Q: Array, O: Array, M: Array, N_eff: Array
) -> tuple[Array, ...]:
    ## Get H and residualize all by covariates
    G, L, H = _prep_lanc_geno(G, L, Q)

    ## Fit G,H ~ L and mask out collinear columns
    _, G, Gl, G_mask, H, Hl, H_mask = _adj_by_lanc(G, H, L)

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
    beta_het, chisq_anc, chisq_het, df_het = _het_score(U[:, None], GltGl, G_mask)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ R
    Hlw = Hl * M * W_L_sqrt
    HtH = Hlw.T @ Hlw
    beta_hom, chisq_hom = _hom_score(UH, HtH)

    return _mask_score(
        beta_het[:, 0], beta_hom, chisq_anc[:, 0], chisq_het, chisq_hom, G_mask, H_mask
    )


def _bt_score_nolanc(
    G: Array, Y: Array, Q: Array, O: Array, M: Array, N_eff: Array
) -> tuple[Array, ...]:
    ## Get H and residualize all by covariates
    G, H = _prep_geno(G, Q)

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
    beta_het, chisq_anc, chisq_het, df_het = _het_score(U[:, None], GtG, G_mask)

    ## Score test for genotypes (homogeneous test)
    UH = H.T @ R
    Hw = H * M * W_sqrt
    HtH = Hw.T @ Hw
    beta_hom, chisq_hom = _hom_score(UH, HtH)

    return _mask_score(
        beta_het[:, 0], beta_hom, chisq_anc[:, 0], chisq_het, chisq_hom, G_mask, H_mask
    )


def _bt_wald_lanc(
    G: Array, L: Array, Y: Array, Q: Array, O: Array, M: Array, N_eff: Array
) -> tuple[Array, ...]:
    K = G.shape[1]

    ## Get H and residualize all by covariates
    G, L, H = _prep_lanc_geno(G, L, Q)

    ## Fit G,H ~ L and mask out collinear columns
    _, G, _, G_mask, H, _, H_mask = _adj_by_lanc(G, H, L)
    H = H[:, None]
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
    chisq_anc = beta_het[:K] ** 2 / jnp.diagonal(XtXw_inv[:K, :K])
    chisq_het = beta_het[:K].T @ solve(XtXw_inv[:K, :K], beta_het[:K])

    ## Wald test for genotypes
    Xh_mask = jnp.concatenate([H_mask[None], L_mask])
    Xh = jnp.concatenate([H, L], axis=1)
    beta_hom = _logistic(Xh, Y, O, M, Xh_mask)
    etah = Xh @ beta_hom + O
    mu = expit(etah)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Xw = Xh * W_sqrt[:, None] * M[:, None]
    XtX = Xw.T @ Xw
    XtXw_inv = _masked_inv(XtX, Xh_mask)
    chisq_hom = beta_hom[0] ** 2 / XtXw_inv[0, 0]

    ## LRT
    l_het = (Y * etag - jnp.log(1 + jnp.exp(etag))) * M
    l_hom = (Y * etah - jnp.log(1 + jnp.exp(etah))) * M
    chisq_lrt = 2 * jnp.sum(l_het - l_hom)

    return _mask_wald(
        beta_het[:K],
        beta_hom[0],
        chisq_anc,
        chisq_het,
        chisq_hom,
        chisq_lrt,
        G_mask,
        H_mask,
    )


def _bt_wald_nolanc(
    G: Array, Y: Array, Q: Array, O: Array, M: Array, N_eff: Array
) -> tuple[Array, ...]:
    K = G.shape[1]

    ## Get H and residualize all by covariates
    G, H = _prep_geno(G, Q)
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
    chisq_anc = beta_het[:K] ** 2 / jnp.diagonal(GtGw_inv[:K, :K])
    chisq_het = beta_het[:K].T @ solve(GtGw_inv[:K, :K], beta_het[:K])

    ## Wald test for genotypes
    H_mask = jnp.sum(H**2, axis=0) > 0
    beta_hom = _logistic(H, Y, O, M, H_mask)
    etah = H @ beta_hom + O
    mu = expit(etah)
    W_sqrt = jnp.sqrt(mu * (1 - mu))
    Hw = H * W_sqrt[:, None] * M[:, None]
    HtH = Hw.T @ Hw
    chisq_hom = beta_hom**2 * HtH

    ## LRT
    l_het = (Y * etag - jnp.log(1 + jnp.exp(etag))) * M
    l_hom = (Y * etah - jnp.log(1 + jnp.exp(etah))) * M
    chisq_lrt = 2 * jnp.sum(l_het - l_hom)

    return _mask_wald(
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


def _make_blockwise(fun, inner_axes, outer_axes, out_axes=-1):
    return jit(
        vmap(
            vmap(fun, in_axes=inner_axes),
            in_axes=outer_axes,
            out_axes=out_axes,
        )
    )


qt_score_lanc = _make_blockwise(
    _qt_score_lanc, (1, 1, None, None, None), (3, 3, 1, 2, 0)
)

qt_score_lanc_impute = jit(
    vmap(_qt_score_lanc, in_axes=(1, 1, None, None, None)),
)

qt_score_nolanc = _make_blockwise(_qt_score_nolanc, (1, None, None, None), (3, 1, 2, 0))


qt_score_nolanc_impute = jit(
    vmap(_qt_score_nolanc, in_axes=(1, None, None, None)),
)

qt_wald_lanc = _make_blockwise(_qt_wald_lanc, (1, 1, None, None, None), (3, 3, 1, 2, 0))

qt_wald_lanc_impute = jit(
    vmap(_qt_wald_lanc, in_axes=(1, 1, None, None, None)),
)

qt_wald_nolanc = _make_blockwise(_qt_wald_nolanc, (1, None, None, None), (3, 1, 2, 0))

qt_wald_nolanc_impute = jit(
    vmap(_qt_wald_nolanc, in_axes=(1, None, None, None)),
)

bt_score_lanc = _make_blockwise(
    _bt_score_lanc, (1, 1, None, None, None, None, None), (3, 3, 1, 2, 1, 1, 0)
)

bt_score_nolanc = _make_blockwise(
    _bt_score_nolanc, (1, None, None, None, None, None), (3, 1, 2, 1, 1, 0)
)

bt_wald_lanc = _make_blockwise(
    _bt_wald_lanc, (1, 1, None, None, None, None, None), (3, 3, 1, 2, 1, 1, 0)
)

bt_wald_nolanc = _make_blockwise(
    _bt_wald_nolanc, (1, None, None, None, None, None), (3, 1, 2, 1, 1, 0)
)
