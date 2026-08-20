# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Shared numerical helpers for step-2 association statistics."""

from jax import jit, vmap
import jax.lax as lax
import jax.numpy as jnp
from jax.numpy.linalg import inv, qr, solve
from jaxtyping import Array
from jax.scipy.special import expit

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
        return (i < max_iter) & (jnp.max(jnp.abs(beta_new - beta)) > tol)

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
    diag = jnp.diagonal(covariance)

    beta_het = U / diag[..., None]
    chisq_anc = U**2 / diag[..., None] / scale
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


def _make_blockwise(fun, inner_axes, outer_axes, out_axes=-1):
    return jit(
        vmap(
            vmap(fun, in_axes=inner_axes),
            in_axes=outer_axes,
            out_axes=out_axes,
        )
    )
