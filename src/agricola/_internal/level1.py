# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Level-1 whole genome predictions.

This module performs "level 1" of agricola step1. It takes the block-wise predictions for
each trait and combines them into a single prediction per-chromosome per-trait
using ridge or logistic ridge regression with cross-validation. The entry-point
for this module is the `level1` function.
"""

import logging
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Optional
from pandas import DataFrame, Index
from .utils import stdize, TraitType
from .models import (
    ridge,
    logistic_ridge,
    logistic_ridge_loo,
)
from .inputs import validate_level1_inputs

logger = logging.getLogger(__name__)

### ─────────────────────────────────────────────────────────────
### Computation
### ─────────────────────────────────────────────────────────────


_fit_ridge = jax.jit(ridge)
_fit_logistic_ridge = jax.jit(logistic_ridge)
_fit_logistic_ridge_loo = jax.jit(logistic_ridge_loo)


def _ridge_cv_qt(
    Z: Array,
    Y: Array,
    train_mask: Array,
    test_mask: Array,
    h2_prior: Array,
    n_blocks: list[int],
) -> NDArray:
    """Get level 1 LOCO predictions for a quantitative trait

    Args:
        Z: A (N, N_blocks) jax array of level 0 predictions
        Y: A (N,) jax array of phenotypes
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D jax array of prior values for snp heritability
        n_blocks: A list with number of blocks for each chromosome

    Returns:
        eta_loco: An (N, C) numpy array of LOCO level 1 predictions
    """
    N, B = Z.shape
    A = len(h2_prior)
    C = len(n_blocks)

    ## Calculate penalties based on prior heritability
    alphas = B * (1 - h2_prior) / h2_prior

    ## Would love to vmap this but it uses way too much memory
    eta = np.zeros(shape=(N, A, C))
    beta = _fit_ridge(Z, Y[:, None], train_mask, alphas)[:, :, :, 0]  # AKB
    beta_mask = np.zeros(shape=(B, C))
    col0 = 0
    for c in range(C):
        n_block = n_blocks[c]
        beta_mask[np.arange(col0, col0 + n_block), c] = 1
        col0 = col0 + n_block

    eta = (
        jnp.einsum(
            "nb,akbc->nack", Z, beta[:, :, :, None] * beta_mask[None, None, :, :]
        )
        * test_mask[:, None, None, :]
    )
    eta = jnp.sum(eta, axis=3)

    ## Get best CV alpha
    eta_all = np.sum(eta, axis=2)
    cv_errors = np.sum((np.asarray(Y)[:, None] - eta_all) ** 2, axis=0)
    idx_best = np.argmin(cv_errors)
    eta_loco = eta_all[:, idx_best][:, None] - eta[:, idx_best, :]

    return eta_loco


def _ridge_cv_bt(
    Z: Array,
    Y: Array,
    train_mask: Array,
    test_mask: Array,
    offset: Array,
    h2_prior: Array,
    n_blocks: list[int],
) -> NDArray:
    """Get level 1 LOCO predictions for a single binary trait

    Args:
        Z: A (N, N_blocks) jax array of level 0 predictions
        Y: A (N,) jax array of phenotypes
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        offset: A (N,) jax array of offsets from covariate model
        h2_prior: A 1D jax array of prior values for snp heritability
        n_blocks: A list with number of blocks for each chromosome

    Returns:
        eta_loco: An (N, C) numpy array of LOCO level 1 predictions (linear predictor)
    """
    ## Assign dimensions
    N, B = Z.shape
    K = train_mask.shape[1]
    A = len(h2_prior)
    C = len(n_blocks)

    ## Calculate penalties based on prior heritability
    alphas = B * (1 - h2_prior) / h2_prior

    eta = np.zeros(shape=(N, A, C))
    for fold in range(K):
        for a in range(A):
            beta = _fit_logistic_ridge(Z, Y, offset, train_mask[:, fold], alphas[a])
            beta_mask = np.zeros(shape=(B, C))
            col0 = 0
            for c in range(C):
                n_block = n_blocks[c]
                beta_mask[np.arange(col0, col0 + n_block), c] = 1
                col0 = col0 + n_block

            eta_chroms = Z @ (beta[:, None] * beta_mask) * test_mask[:, fold, None]
            eta[:, a, :] += eta_chroms

    ## Get best CV alpha
    eta_all = np.sum(eta, axis=2)
    l_i_alphas = Y[:, None] * (eta_all + offset[:, None]) - np.log(1 + jnp.exp(eta_all))
    l_alphas = np.sum(l_i_alphas, axis=0)
    idx_best = np.argmax(l_alphas)
    eta_loco = eta_all[:, idx_best][:, None] - eta[:, idx_best, :]

    return eta_loco


def _ridge_loocv_bt(
    Z: Array, Y: Array, offset: Array, h2_prior: Array, n_blocks: list[int]
) -> NDArray:
    """Get level 1 LOCO LOOCV predictions for a single binary trait

    Args:
        Z: A (N, N_blocks) jax array of level 0 predictions
        Y: A (N,) jax array of phenotypes
        offset: A (N,) jax array of offsets from covariate model
        h2_prior: A 1D jax array of prior values for snp heritability
        n_blocks: A list with number of blocks for each chromosome

    Returns:
        An (N, C) numpy array of LOCO level 1 predictions (linear predictor)
    """
    ## Assign dimensions
    N, B = Z.shape
    A = len(h2_prior)
    C = len(n_blocks)

    ## Calculate penalties based on prior heritability
    alphas = B * (1 - h2_prior) / h2_prior

    eta = np.zeros(shape=(N, A, C))
    for a in range(A):
        beta = _fit_logistic_ridge_loo(Z, Y, offset, alphas[a]).T
        col0 = 0
        for c in range(C):
            n_block = n_blocks[c]
            beta_mask = np.zeros(B)
            beta_mask[np.arange(col0, col0 + n_block)] = 1
            eta[:, a, c] += np.sum(Z * beta * beta_mask[None, :], axis=1)
            col0 = col0 + n_block

    # Get best CV alpha
    eta_all = np.sum(eta, axis=2)
    l_i_alphas = Y[:, None] * (eta_all + offset[:, None]) - np.log(1 + jnp.exp(eta_all))
    l_alphas = np.sum(l_i_alphas, axis=0)
    idx_best = np.argmax(l_alphas)
    eta_loco = eta_all[:, idx_best][:, None] - eta[:, idx_best, :]

    return eta_loco


### ─────────────────────────────────────────────────────────────
### Public-facing
### ─────────────────────────────────────────────────────────────


def level1(
    level0_files: dict[str, dict[str, str]],
    Y: ArrayLike,
    X: Optional[ArrayLike],
    phenotypes: list[str],
    train_mask: ArrayLike,
    test_mask: ArrayLike,
    h2_prior: ArrayLike,
    trait_type: str,
    loocv: bool = False,
) -> dict[str, DataFrame]:
    """Perform level 1 ridge regressions

    Args:
        level0_files: A two-level dict. The outer keys are phenotypes, the inner keys are chromosomes,
            and the values are paths to .npy files containing (N, n_blocks) level 0 predictions.
        Y: A (N, P) ArrayLike of phenotypes
        X: An optional (N, C) ArrayLike of covariates (no intercept)
        phenotypes: A list of phenotype names, ordered as the columns of Y
        train_mask: An (N, K) ArrayLike indicating training set status for each set k in 1, ..., K
        test_mask: An (N, K) ArrayLike indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D ArrayLike of prior values for snp heritability
        trait_type: Either "qt" or "bt"
        loocv: A boolean indicating whether to perform LOOCV instead of standard
            cross validation. Ignored for trait_type="qt".

    Returns:
        A dict where keys are chromosomes and values are (N, P) pandas DataFrames of level 1 predictions
    """
    ## Validate inputs
    Y, X, train_mask, test_mask, h2_prior, trait = validate_level1_inputs(
        Y, X, phenotypes, train_mask, test_mask, h2_prior, trait_type
    )
    N, P = Y.shape
    C = len(level0_files[phenotypes[0]])

    ## Standardize Y if QT
    if trait == TraitType.QT:
        Q, _ = jnp.linalg.qr(X, mode="reduced")
        Y = stdize(Y - (Q @ (Q.T @ Y)))

    loco_arr = np.zeros(shape=(N, P, C))
    logger.info("Getting level 1 predictions")
    with tqdm(total=Y.shape[1], unit="phenotypes") as pbar:
        for p in range(P):
            pheno: str = phenotypes[p]
            Zs = [np.load(v) for v in level0_files[pheno].values()]
            n_blocks = [z.shape[1] for z in Zs]
            Z = jnp.concatenate(Zs, axis=1)

            if trait == TraitType.BT:
                beta_covar = logistic_ridge(X, Y[:, p], jnp.zeros(N), jnp.ones(N), 0)
                offset = X @ beta_covar

                if not loocv:
                    loco_p = _ridge_cv_bt(
                        Z, Y[:, p], train_mask, test_mask, offset, h2_prior, n_blocks
                    )
                else:
                    loco_p = _ridge_loocv_bt(Z, Y[:, p], offset, h2_prior, n_blocks)

            else:
                loco_p = _ridge_cv_qt(
                    Z, Y[:, p], train_mask, test_mask, h2_prior, n_blocks
                )

            loco_arr[:, p, :] = loco_p
            pbar.update(1)

    logger.info("Finished getting level 1 predictions\n")
    level1_loco = {}
    for i, chrom in enumerate(level0_files[phenotypes[0]].keys()):
        level1_loco[chrom] = DataFrame(loco_arr[:, :, i], columns=Index(phenotypes))

    return level1_loco
