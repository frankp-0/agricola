# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Shared numerical helpers for step-2 association statistics."""

import jax.lax as lax
import jax.numpy as jnp
from jax import jit, vmap
from jax.numpy.linalg import inv, qr, solve
from jax.scipy.special import expit
from jaxtyping import Array

### ─────────────────────────────────────────────────────────────

DEFAULT_SCALE = jnp.array([1.0])


def _naninf_to_0(X: Array) -> Array:
    return jnp.nan_to_num(X, posinf=0, neginf=0)


def logistic(
    X: Array,
    y: Array,
    offset: Array,
    train_mask: Array,
    X_mask: Array,
    max_iter: int = 30,
    tol: float = 1e-6,
) -> Array:
    """Fit masked logistic regression by iteratively reweighted least squares."""
    beta0 = jnp.zeros(X.shape[1])

    def _body_fun(state):
        i, beta = state
        eta = X @ beta + offset
        mu = expit(eta)
        r = (y - mu) * train_mask
        w = mu * (1 - mu) * train_mask
        XW = X * w[:, None]
        XT_r = X.T @ r
        delta = masked_solve(X.T @ XW, X_mask, XT_r)
        beta_new = _naninf_to_0(beta + delta)
        return i + 1, beta_new

    def _cond_fun(state):
        i, beta = state
        _, beta_new = _body_fun(state)
        return (i < max_iter) & (jnp.max(jnp.abs(beta_new - beta)) > tol)

    _, beta = lax.while_loop(_cond_fun, _body_fun, (0, beta0))

    return beta


def logistic_with_convergence(
    X: Array,
    y: Array,
    offset: Array,
    train_mask: Array,
    X_mask: Array,
    max_iter: int = 30,
    tol: float = 1e-6,
) -> tuple[Array, Array]:
    """Fit masked logistic regression and report convergence status."""
    beta0 = jnp.zeros(X.shape[1])
    total_weight = jnp.sum(train_mask)

    def gradient(beta):
        eta = X @ beta + offset
        mu = expit(eta)
        return X.T @ ((y - mu) * train_mask)

    gradient0 = gradient(beta0)

    def body_fun(state):
        i, beta, current_gradient, _ = state
        eta = X @ beta + offset
        mu = expit(eta)
        w = mu * (1 - mu) * train_mask
        XW = X * w[:, None]
        delta = masked_solve(X.T @ XW, X_mask, current_gradient)
        beta_new = _naninf_to_0(beta + delta)
        gradient_new = gradient(beta_new)
        max_gradient = jnp.max(jnp.abs(gradient_new)) / jnp.maximum(total_weight, 1)
        converged = (
            (total_weight > 0)
            & jnp.all(jnp.isfinite(beta_new))
            & jnp.all(jnp.isfinite(gradient_new))
            & (max_gradient <= tol)
        )
        return i + 1, beta_new, gradient_new, converged

    def cond_fun(state):
        i, _, _, converged = state
        return (i < max_iter) & ~converged

    _, beta, _, converged = lax.while_loop(
        cond_fun,
        body_fun,
        (0, beta0, gradient0, jnp.array(False)),
    )
    return beta, converged


def qr_resid(X: Array, Q: Array) -> Array:
    return X - Q @ (Q.T @ X)


def _project_and_mask_collinear(X: Array, QL: Array) -> tuple[Array, ...]:
    Xl = qr_resid(X, QL)
    Xl_norm = jnp.sum(Xl**2, axis=0)
    X_norm = jnp.sum(X**2, axis=0)
    rank_tol = jnp.finfo(X.dtype).eps * max(X.shape[0], QL.shape[1])
    X_mask = Xl_norm > rank_tol * X_norm
    X_mask_shaped = jnp.reshape(X_mask, X_mask.shape + (1,) * (X.ndim - 1)).T
    Xl = Xl * X_mask_shaped
    X = X * X_mask_shaped

    return X, Xl, X_mask


def _masked_nan(X: Array, mask: Array) -> Array:
    return jnp.where(mask, X, jnp.nan)


def masked_inv(A: Array, mask: Array) -> Array:
    """Invert a matrix after replacing inactive diagonal entries with one."""
    return inv(A + jnp.diag((~mask).astype(A.dtype)))


def masked_solve(A: Array, mask: Array, x: Array) -> Array:
    """Solve a masked linear system without changing its compiled shape."""
    return solve(A + jnp.diag((~mask).astype(A.dtype)), x)


def prep_geno(G: Array, Q: Array) -> tuple[Array, ...]:
    H = jnp.sum(G, axis=1)
    return qr_resid(G, Q), qr_resid(H, Q)


def prep_lanc_geno(G: Array, L: Array, Q: Array) -> tuple[Array, ...]:
    H = jnp.sum(G, axis=1)
    return (qr_resid(G, Q), qr_resid(L, Q), qr_resid(H, Q))


def lanc_basis(L: Array, mask: Array | None = None) -> tuple[Array, Array]:
    """Return a fixed-width, rank-aware QR basis for local ancestry.

    ``mask`` restricts rank detection to observed samples while retaining all
    rows in the returned basis for JAX shape stability.
    """
    L_for_rank = L if mask is None else L * mask[:, None]
    QL, R = qr(L_for_rank, mode="reduced")
    r_diag = jnp.abs(jnp.diagonal(R))
    rank_tol = jnp.finfo(L.dtype).eps * max(L.shape) * jnp.max(r_diag)
    L_mask = r_diag > rank_tol
    return QL * L_mask, L_mask


def adj_by_lanc(
    G: Array, H: Array, L: Array, mask: Array | None = None
) -> tuple[Array, ...]:
    """Residualize genotype designs against a rank-aware ancestry basis."""
    QL, _ = lanc_basis(L, mask)
    G, Gl, G_mask = _project_and_mask_collinear(G, QL)
    H, Hl, H_mask = _project_and_mask_collinear(H, QL)
    return QL, G, Gl, G_mask, H, Hl, H_mask


def het_score(
    U: Array, covariance: Array, mask: Array, scale: Array = DEFAULT_SCALE
) -> tuple[Array, ...]:
    """Return joint, ancestry-specific, and scaled heterogeneous score results."""
    solved_U = masked_solve(covariance, mask, U)
    covariance_of_beta = masked_solve(
        covariance, mask, jnp.eye(covariance.shape[0], dtype=covariance.dtype)
    )

    beta_het = solved_U
    chisq_anc = beta_het**2 / jnp.diagonal(covariance_of_beta)[..., None] / scale
    chisq_het = jnp.sum(U * solved_U, axis=0) / scale

    return beta_het, chisq_anc, chisq_het


def hom_score(UH: Array, HtH: Array, scale: Array = DEFAULT_SCALE):
    """Return the homogeneous coefficient and its scaled score statistic."""
    beta_hom = UH / HtH
    chisq = UH**2 / HtH / scale
    return beta_hom, chisq


def mask_score(
    beta_het: Array,
    beta_hom: Array,
    chisq_anc: Array,
    chisq_het: Array,
    chisq_hom: Array,
    G_mask: Array,
    H_mask: Array,
) -> tuple[Array, ...]:
    """Apply genotype-variation masks to score-test outputs."""
    df_het = jnp.sum(G_mask)
    return (
        _masked_nan(chisq_hom, H_mask),
        _masked_nan(beta_hom, H_mask),
        _masked_nan(chisq_het, df_het != 0),
        _masked_nan(beta_het, G_mask),
        df_het,
        _masked_nan(chisq_anc, G_mask),
    )


def mask_wald(
    beta_het: Array,
    beta_hom: Array,
    chisq_anc: Array,
    chisq_het: Array,
    chisq_hom: Array,
    chisq_lrt: Array,
    G_mask: Array,
    H_mask: Array,
) -> tuple[Array, ...]:
    """Apply genotype-variation masks and degrees of freedom to Wald outputs."""
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


def make_blockwise(fun, inner_axes, outer_axes, out_axes=-1):
    return jit(
        vmap(
            vmap(fun, in_axes=inner_axes),
            in_axes=outer_axes,
            out_axes=out_axes,
        )
    )
