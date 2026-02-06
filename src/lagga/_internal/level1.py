# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Level-1 whole genome predictions.

This module performs "level 1" of lagga step1. It takes the block-wise predictions for
each trait and combines them into a single prediction per-chromosome per-trait
using ridge or logistic ridge regression with cross-validation. The entry-point
for this module is the `level1` function.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Optional
from .utils import stdize
from .models import (
    ridge,
    logistic_ridge,
    logistic_ridge_loo,
)
from .inputs import validate_level1_inputs


### ─────────────────────────────────────────────────────────────
### Computation
### ─────────────────────────────────────────────────────────────


def _ridge_cv_qt(
    Z: Array, Y: Array, train_mask: Array, test_mask: Array, h2_prior: Array
) -> NDArray:
    """Get level 1 predictions for a single chromosome for quantitative traits

    Args:
        Z: A (N, P, N_blocks) jax array of level 0 predictions (for a single chromosome)
        Y: A (N, P) jax array of (residualized, standardize) phenotypes
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D jax array of prior values for snp heritability

    Returns:
        Yhat: A (N, P) numpy array of level 1 predictions
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
            ridge_fold_beta = ridge(
                Z[:, pheno, :],
                Y[:, pheno],
                train_mask[:, fold],
                alphas,
            ).squeeze(axis=1)
            ridge_fold = (Z[:, pheno, :] @ ridge_fold_beta) * test_mask[:, fold][
                :, None
            ]
            Yhat_alphas[:, pheno, :] += ridge_fold

    # Get best CV alpha per-phenotype
    cv_errors = np.mean((np.asarray(Y)[:, :, None] - Yhat_alphas) ** 2, axis=0)
    alpha_idx = np.argmin(cv_errors, axis=1)

    # Get best CV prediction
    Yhat = np.take_along_axis(Yhat_alphas, alpha_idx[None, :, None], axis=2).squeeze(
        axis=2
    )
    return Yhat


def _ridge_loocv_bt(Z: Array, Y: Array, offset: Array, h2_prior: Array) -> NDArray:
    """Get level 1 LOOCV predictions for a single chromosome for binary traits

    Args:
        Z: A (N, P, N_blocks) jax array of level 0 predictions (for a single chromosome)
        Y: A (N, P) jax array of phenotypes
        offset: A (N, P) jax array of offsets from covariate model
        h2_prior: A 1D jax array of prior values for snp heritability

    Returns:
        An (N, P) numpy array of level 1 predictions (linear predictor, not
        including offset)
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
        beta_pheno = loocv_model(Z[:, pheno, :], Y[:, pheno], offset[:, pheno], alphas)
        eta_pheno = (
            jnp.sum(Z[:, pheno, :][:, None, :] * beta_pheno.T, axis=2)
            + offset[:, pheno][:, None]
        )
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


def _ridge_cv_bt(
    Z: Array,
    Y: Array,
    train_mask: Array,
    test_mask: Array,
    offset: Array,
    h2_prior: Array,
) -> NDArray:
    """Get level 1 CV predictions for a single chromosome for binary traits

    Args:
        Z: A (N, P, N_blocks) jax array of level 0 predictions (for a single chromosome)
        Y: A (N, P) jax array of phenotypes
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        offset: A (N, P) jax array of offsets from covariate model
        h2_prior: A 1D jax array of prior values for snp heritability

    Returns:
        An (N, P) numpy array of level 1 predictions (linear predictor, not
        including offset)
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
            beta_pheno_fold = cv_model(
                Z[:, pheno, :],
                Y[:, pheno],
                offset[:, pheno],
                train_mask[:, fold],
                alphas,
            )
            eta_pheno_fold = Z[:, pheno, :] @ beta_pheno_fold
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


### ─────────────────────────────────────────────────────────────
### Public-facing
### ─────────────────────────────────────────────────────────────


def level1(
    Z: dict[str, NDArray],
    Y: ArrayLike,
    X: Optional[ArrayLike],
    train_mask: Optional[ArrayLike],
    test_mask: Optional[ArrayLike],
    h2_prior: ArrayLike,
    trait_type: str,
    loocv: bool = False,
) -> dict[str, NDArray]:
    """Perform level 1 ridge regressions

    Args:
        Z: A dict where keys are chromosomes and values are (N, P, N_blocks)
            numpy arrays of level 0 predictions
        Y: A (N, P) ArrayLike of phenotypes
        X: An optional (N, C) ArrayLike of covariates (no intercept)
        train_mask: An optional (N, K) ArrayLike indicating training set status for each set k in 1, ..., K
        test_mask: An optional (N, K) ArrayLike indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D ArrayLike of prior values for snp heritability
        trait_type: Either "qt" or "bt"
        loocv: A boolean indicating whether to perform LOOCV instead of standard
            cross validation. Ignored for trait_type="qt".

    Returns:
        A dict where keys are chromosomes and values are (N, P) numpy arrays of level 1 predictions
    """
    ## Validate inputs
    Z, Y, X, train_mask, test_mask, h2_prior = validate_level1_inputs(
        Z, Y, X, train_mask, test_mask, h2_prior
    )

    offset: Optional[Array] = None
    if trait_type == "qt":
        Q, _ = jnp.linalg.qr(X, mode="reduced")
        Y = stdize(Y - (Q @ (Q.T @ Y)))
    elif trait_type == "bt":
        ## Covariate-only model
        covar_model = jax.vmap(
            logistic_ridge, in_axes=(None, 1, None, None, None), out_axes=1
        )
        offset = X @ covar_model(X, Y, jnp.zeros(X.shape[0]), jnp.ones(Y.shape[0]), 0)
    else:
        raise ValueError("trait_type must be qt or bt")

    ## Perform level 1 for each chromosome
    with tqdm(
        total=len(Z),
        desc="Getting level 1 predictions",
        unit="chromosomes",
    ) as pbar:
        level1_predictions = {}
        for chrom, Z_chrom in Z.items():
            if trait_type == "bt" and loocv:
                pred_chrom = _ridge_loocv_bt(
                    jnp.asarray(Z_chrom), Y, jnp.asarray(offset), h2_prior
                )
            elif trait_type == "bt":
                pred_chrom = _ridge_cv_bt(
                    jnp.asarray(Z_chrom),
                    Y,
                    jnp.asarray(train_mask),
                    jnp.asarray(test_mask),
                    jnp.asarray(offset),
                    h2_prior,
                )
            elif trait_type == "qt":
                pred_chrom = _ridge_cv_qt(
                    jnp.asarray(Z_chrom),
                    Y,
                    jnp.asarray(train_mask),
                    jnp.asarray(test_mask),
                    h2_prior,
                )
            level1_predictions[chrom] = pred_chrom
            pbar.update(1)

    return level1_predictions
