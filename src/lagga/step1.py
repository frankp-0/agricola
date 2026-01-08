# MIT License
# Copyright (c) 2025 Franklin Ockerman
# See LICENSE.txt file for full license text

import jax
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
from tqdm import tqdm
from typing import Optional
from ._utils import stdize, assert_covar_full_rank
from .models import ridge, logistic, logistic_ridge, logistic_ridge_loo

### ─────────────────────────────────────────────────────────────
### Helper Functions
### ─────────────────────────────────────────────────────────────


def validate_step1_inputs(
    Z: dict[str, np.ndarray],
    Y: Array,
    X: Optional[Array],
    train_mask: Optional[Array],
    test_mask: Optional[Array],
    h2_prior: Array,
):
    ## Array conversions
    Y = jnp.asarray(Y)
    train_mask = jnp.asarray(train_mask)
    test_mask = jnp.asarray(test_mask)
    h2_prior = jnp.asarray(h2_prior)
    if X is not None:
        X = jnp.asarray(X)

    ## Check array shapes
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N, P), got shape {Y.shape}")
    N = Y.shape[0]

    if not isinstance(Z, dict):
        raise TypeError(f"Z must be a dict, got {type(Z)}")

    if len(Z) == 0:
        raise ValueError("Z must not be empty")

    Z_N = None
    Z_P = None
    for chrom, Z_chrom in Z.items():
        if not isinstance(Z_chrom, (np.ndarray, jnp.ndarray)):
            raise TypeError(
                f"Z[{chrom}] must be a numpy/jax array, got {type(Z_chrom)}"
            )
        if Z_chrom.ndim != 3:
            raise ValueError(
                f"Z[{chrom}] must be 3D (N, P, N_blocks), got shape {Z_chrom.shape}"
            )

        n_chrom, p_chrom, _ = Z_chrom.shape

        if Z_N is None:
            Z_N = n_chrom
            Z_P = p_chrom
        else:
            if n_chrom != Z_N:
                raise ValueError(
                    f"All Z arrays must have same N; got {Z_N} vs {n_chrom} in Z[{chrom}]"
                )
            if p_chrom != Z_P:
                raise ValueError(
                    f"All Z arrays must have same P; got {Z_P} vs {p_chrom} in Z[{chrom}]"
                )

    if Z_N != N:
        raise ValueError(f"Z arrays have N={Z_N} but Y has N={N}")

    if X is not None:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N, C), got shape {X.shape}")
        if X.shape[0] != N:
            raise ValueError(
                f"X.shape[0] must match Y.shape[0], got {X.shape[0]} vs {N}"
            )

    if not (train_mask is None or test_mask is None):
        if (
            train_mask.ndim != 2
            or test_mask.ndim != 2
            or train_mask.shape != test_mask.shape
        ):
            raise ValueError(
                "train_mask and test_mask must be 2D (N, K) with the same shape"
            )
        if train_mask.shape[0] != N or test_mask.shape[0] != N:
            raise ValueError("train_mask/test_mask must match N of Y")

    if h2_prior.ndim != 1:
        raise ValueError(f"h2_prior must be 1D, got shape {h2_prior.shape}")
    if not jnp.all((0 < h2_prior) & (h2_prior < 1)):
        raise ValueError("h2_prior values must be in the open interval (0, 1)")

    return (Z, Y, X, train_mask, test_mask, h2_prior)


### ─────────────────────────────────────────────────────────────
### Quantitative Traits
### ─────────────────────────────────────────────────────────────


def _ridge_cv_qt(Z, Y, train_mask, test_mask, h2_prior):
    """Get level 1 predictions for a single chromosome for quantitative traits

    Args:
        Z: A (N, P, N_blocks) jax array of step 0 predictions (for a single chromosome)
        Y: A (N, P) jax array of (residualized, standardize) phenotypes
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D jax array of prior values for snp heritability

    Returns:
        Yhat: A (N, P) numpy array of step 0 predictions
    """
    ## Assign dimensions
    n, p = Y.shape
    _, k = train_mask.shape
    a = len(h2_prior)

    ## Calculate penalties based on prior heritability
    alphas = Z.shape[2] * (1 - h2_prior) / h2_prior

    ## Would love to vmap this but it uses way too much memory
    Yhat_alphas = np.zeros(shape=(n, p, a), dtype=np.float32)
    for fold in range(k):
        for pheno in range(p):
            ridge_fold = ridge(
                Z[:, pheno, :],
                Y[:, pheno],
                train_mask[:, fold],
                test_mask[:, fold],
                alphas,
            )
            Yhat_alphas[:, pheno, :] += ridge_fold[:, 0, :]

    # Get best CV alpha per-phenotype
    cv_errors = np.mean(
        (np.asarray(Y)[:, :, None] - Yhat_alphas) ** 2, axis=0
    )  # (p, a)
    alpha_idx = np.argmin(cv_errors, axis=1)  # (p,)

    # Get best CV prediction
    Yhat = np.take_along_axis(Yhat_alphas, alpha_idx[None, :, None], axis=2).squeeze(
        axis=2
    )
    return Yhat


def step1_qt(
    Z: dict[str, np.ndarray],
    Y: Array,
    X: Optional[Array],
    train_mask: Array,
    test_mask: Array,
    h2_prior: Array,
):
    """Perform level 1 ridge regressions for quantitative phenotypes

    Args:
        Z: A dict where keys are chromosomes and values are (N, P, N_blocks) jax arrays of step 0 predictions
        Y: A (N, P) jax array of phenotypes
        X: A (N, C) jax array of covariates (no intercept)
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D jax array of prior values for snp heritability

    Returns:
        step1_predictions: A dict where keys are chromosomes and values are (N, P) numpy arrays of step 0 predictions
    """

    n, p = Y.shape

    ## Residualize and standardize phenotypes
    if X is None:
        X = jnp.ones((Y.shape[0], 1), dtype=np.float32)
    else:
        X = jnp.concatenate([jnp.ones((Y.shape[0], 1), dtype=np.float32), X], axis=1)
    X = stdize(X)
    assert_covar_full_rank(X)
    Q, _ = jnp.linalg.qr(X, mode="reduced")
    Y = stdize(Y - (Q @ (Q.T @ Y)))

    ## Perform step 1 for each chromosome
    with tqdm(
        total=len(Z),
        desc="Getting step 1 predictions",
        unit="chromosomes",
    ) as pbar:
        step1_predictions = {}
        for chrom, Z_chrom in Z.items():
            Yhat_chrom = np.empty(shape=(n, p), dtype=np.float32)
            Yhat_chrom = _ridge_cv_qt(Z_chrom, Y, train_mask, test_mask, h2_prior)
            step1_predictions[chrom] = Yhat_chrom
            pbar.update(1)

    return step1_predictions


### ─────────────────────────────────────────────────────────────
### Binary Traits
### ─────────────────────────────────────────────────────────────


def _ridge_loocv_bt(Z, Y, offset, h2_prior):
    """Get level 1 LOOCV predictions for a single chromosome for binary traits

    Args:
        Z: A (N, P, N_blocks) jax array of step 0 predictions (for a single chromosome)
        Y: A (N, P) jax array of phenotypes
        offset: A (N, P) jax array of offsets from covariate model
        h2_prior: A 1D jax array of prior values for snp heritability

    """
    ## Assign dimensions
    n, p = Y.shape
    a = len(h2_prior)

    ## Calculate penalties based on prior heritability
    alphas = Z.shape[2] * (1 - h2_prior) / h2_prior

    ## Would love to vmap this but it uses way too much memory
    eta_alphas = np.zeros(shape=(n, p, a), dtype=np.float32)
    for pheno in range(p):
        loocv_model = jax.vmap(
            logistic_ridge_loo, in_axes=(None, None, None, 0), out_axes=1
        )
        eta_pheno = loocv_model(Z[:, pheno, :], Y[:, pheno], offset[:, pheno], alphas)
        eta_alphas[:, pheno, :] += eta_pheno

    # Get best CV alpha per-phenotype
    l_i_alphas = Y[:, :, None] * eta_alphas - jnp.log(1 + jnp.exp(eta_alphas))
    l_alphas = jnp.sum(l_i_alphas, axis=0)
    alpha_idx = np.argmax(l_alphas, axis=1)

    # Get best CV prediction
    eta_hat = np.take_along_axis(eta_alphas, alpha_idx[None, :, None], axis=2).squeeze(
        axis=2
    )
    return eta_hat


def _ridge_cv_bt(Z, Y, train_mask, test_mask, offset, h2_prior):
    """Get level 1 CV predictions for a single chromosome for binary traits

    Args:
        Z: A (N, P, N_blocks) jax array of step 0 predictions (for a single chromosome)
        Y: A (N, P) jax array of phenotypes
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        offset: A (N, P) jax array of offsets from covariate model
        h2_prior: A 1D jax array of prior values for snp heritability

    """
    ## Assign dimensions
    n, p = Y.shape
    _, k = train_mask.shape
    a = len(h2_prior)

    ## Calculate penalties based on prior heritability
    alphas = Z.shape[2] * (1 - h2_prior) / h2_prior

    ## Would love to vmap this but it uses way too much memory
    eta_alphas = np.zeros(shape=(n, p, a), dtype=np.float32)
    for fold in range(k):
        for pheno in range(p):
            cv_model = jax.vmap(
                logistic_ridge, in_axes=(None, None, None, None, 0), out_axes=1
            )
            eta_pheno_fold = cv_model(
                Z[:, pheno, :],
                Y[:, pheno],
                offset[:, pheno],
                train_mask[:, fold],
                alphas,
            )
            eta_pheno_fold = eta_pheno_fold * test_mask[:, fold][:, None]
            eta_alphas[:, pheno, :] += eta_pheno_fold

    # Get best CV alpha per-phenotype
    l_i_alphas = Y[:, :, None] * eta_alphas - jnp.log(1 + jnp.exp(eta_alphas))
    l_alphas = jnp.sum(l_i_alphas, axis=0)
    alpha_idx = np.argmax(l_alphas, axis=1)

    # Get best CV prediction
    eta_hat = np.take_along_axis(eta_alphas, alpha_idx[None, :, None], axis=2).squeeze(
        axis=2
    )
    return eta_hat


def step1_bt(
    Z: dict[str, np.ndarray],
    Y: Array,
    X: Optional[Array],
    loocv: bool,
    train_mask: Optional[Array],
    test_mask: Optional[Array],
    h2_prior: Array,
):
    """Perform level 1 ridge regressions for binary phenotypes

    Args:
        Z: A dict where keys are chromosomes and values are (N, P, N_blocks) jax arrays of step 0 predictions
        Y: A (N, P) jax array of phenotypes
        X: An optional (N, C) jax array of covariates (no intercept)
        loocv: A logical indicating whether to perform LOOCV. If False, train_mask, test_mask must be provided
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D jax array of prior values for snp heritability

    Returns:
        step1_predictions: A dict where keys are chromosomes and values are (N, P) numpy arrays of step 0 predictions
    """
    ## Validate inputs
    Z, Y, X, train_mask, test_mask, h2_prior = validate_step1_inputs(
        Z, Y, X, train_mask, test_mask, h2_prior
    )

    ## Covariate-only model
    if X is None:
        X = jnp.ones((Y.shape[0], 1), dtype=np.float32)
    else:
        X = jnp.concatenate([jnp.ones((Y.shape[0], 1), dtype=np.float32), X], axis=1)
    X = stdize(X)
    assert_covar_full_rank(X)

    covar_model = jax.vmap(logistic, in_axes=(None, 1, None), out_axes=1)
    offset = X @ covar_model(X, Y, jnp.zeros(X.shape[0]))

    ## Perform step 1 for each chromosome
    with tqdm(
        total=len(Z),
        desc="Getting step 1 predictions",
        unit="chromosomes",
    ) as pbar:
        step1_predictions = {}
        for chrom, Z_chrom in Z.items():
            if loocv:
                eta_hat_chrom = _ridge_loocv_bt(Z_chrom, Y, offset, h2_prior)
            else:
                eta_hat_chrom = _ridge_cv_bt(
                    Z_chrom, Y, train_mask, test_mask, offset, h2_prior
                )
            step1_predictions[chrom] = eta_hat_chrom
            pbar.update(1)

    return step1_predictions
