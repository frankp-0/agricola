# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Level-0 block-wise whole-genome ridge predictions.

This module performs "step 0" of lagga. It splits the genome into blocks and
performs a ridge regression within each block. It returns block-wise predictions
for each trait across a sequence of heritability priors. The entry-point for this
module is the `step0` function.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
import numpy as np
from tqdm import tqdm
from typing import Optional
from lanctools import LancData
from ._utils import stdize, assert_covar_full_rank, get_geno_lanc_deconv
from .models import ridge
from numpy.typing import NDArray


def validate_step0_inputs(
    datasets: list[LancData],
    Y: ArrayLike,
    X: Optional[ArrayLike],
    train_mask: ArrayLike,
    test_mask: ArrayLike,
    h2_prior: ArrayLike,
    B: int = 2000,
    idx_sample: Optional[ArrayLike] = None,
    variants: Optional[list[str]] = None,
) -> tuple[Array, Array, Array, Array, Array, Optional[Array]]:
    """Validate input data for step0"""
    ## genotype/lanc data
    if not isinstance(datasets, (list, tuple)):
        raise TypeError(f"datasets must be a list of LancData, got {type(datasets)}")
    for i, ds in enumerate(datasets):
        if not isinstance(ds, LancData):
            raise TypeError(f"datasets[{i}] must be LancData, got {type(ds)}")

    ## Y
    Y = jnp.asarray(Y)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N, P), got shape {Y.shape}")
    N = Y.shape[0]

    ## X
    if X is None:
        X = jnp.ones((Y.shape[0], 1), dtype=np.float32)
    else:
        X = jnp.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N, C), got shape {X.shape}")
        if X.shape[0] != N:
            raise ValueError(
                f"X.shape[0] must match Y.shape[0], got {X.shape[0]} vs {N}"
            )
        X = jnp.concatenate([jnp.ones((Y.shape[0], 1), dtype=np.float32), X], axis=1)
    X = stdize(X)
    assert_covar_full_rank(X)

    ## Masks
    train_mask = jnp.asarray(train_mask)
    test_mask = jnp.asarray(test_mask)
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

    ## H2
    h2_prior = jnp.asarray(h2_prior)
    if h2_prior.ndim != 1:
        raise ValueError(f"h2_prior must be 1D, got shape {h2_prior.shape}")
    if not jnp.all((0 < h2_prior) & (h2_prior < 1)):
        raise ValueError("h2_prior values must be in the open interval (0, 1)")

    ## B
    if not isinstance(B, int) or B <= 0:
        raise ValueError(f"B must be a positive integer, got {B}")

    ## variants
    if variants is not None:
        if not isinstance(variants, (list, tuple)) or not all(
            isinstance(v, str) for v in variants
        ):
            raise TypeError("variants must be a list of strings")

    ## samples
    N_pgen = datasets[0].pgen.get_raw_sample_ct()
    if idx_sample is not None:
        idx_sample = jnp.asarray(idx_sample)
        if idx_sample.ndim != 1:
            raise TypeError("idx_sample must be 1D")
        if idx_sample.dtype != np.uint32:
            raise TypeError("idx_sample must have dtype numpy.uint32")
        if not set(np.asarray(idx_sample)).issubset(np.arange(N_pgen, dtype=np.uint32)):
            raise ValueError("idx_sample outside range of N samples")

    return (Y, X, train_mask, test_mask, h2_prior, idx_sample)


def _step0_block(
    dataset: LancData,
    Y: Array,
    Q: Array,
    idx_sample: Optional[Array],
    train_mask: Array,
    test_mask: Array,
    block: NDArray,
    h2_prior: Array,
) -> NDArray:
    """Get level 0 predictions for a single block

    Args:
        dataset: A  LancData object
        Y: A (N, P) jax array of (residualized, standardized) phenotypes
        Q: A (N, C) jax array with the Q matrix in the QR decomposition of the covariates
        idx_sample: An optional (N_sub,) jax array with indices of samples to include
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        block: A (B,) ndarray with indices of variants in the block
        h2_prior: A 1D jax array of prior values for snp heritability

    Returns:
        Z_block: A (N, P, len(h2_prior)) numpy array of predictions
    """
    ## Standardize genotype block and residualize by covariates
    G, _ = get_geno_lanc_deconv(dataset, block)
    if idx_sample is not None:
        G = G[idx_sample]
    G = G[:, :, 0] + G[:, :, 1]
    G = stdize(G - (Q @ (Q.T @ G)))  # pyright: ignore

    ## Calculate penalties based on prior heritability
    B = G.shape[1]
    alphas = B * (1 - h2_prior) / h2_prior

    ## Perform ridge regression
    ridge_beta = jax.vmap(ridge, in_axes=(None, None, 1, None))(
        G, Y, train_mask, alphas
    )
    ridge_Z = jnp.einsum("nb,kbpa->nkpa", G, ridge_beta) * test_mask[:, :, None, None]
    Z_block = jnp.sum(ridge_Z, axis=1)
    Z_block = np.asarray(stdize(Z_block))
    return Z_block


def _step0_dataset(
    dataset: LancData,
    Y: Array,
    Q: Array,
    idx_sample: Optional[Array],
    train_mask: Array,
    test_mask: Array,
    B: int,
    variants: Optional[list[str]],
    h2_prior: Array,
    desc: str,
) -> dict[str, NDArray]:
    """Get level 0 predictions for a dataset

    Args:
        dataset: A  LancData object
        Y: A (N, P) jax array of (residualized, standardized) phenotypes
        Q: A (N, C) jax array with the Q matrix in the QR decomposition of the covariates
        idx_sample: An optional (N_sub,) jax array with indices of samples to include
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        B: The number of variants per block
        variants: A list of variant IDs to include in the analysis. If not provided, all variants are used
        h2_prior: A 1D jax array of prior values for snp heritability
        desc: A string describing the dataset, used for tracking progress

    Returns:
        Z_chroms: A dict where keys are chromosomes and values are (N, P, N_predictors) numpy arrays of step 0 predictions
    """

    ## Get variant indices
    if variants is None:
        idx_variant = np.arange(dataset.pvar.get_variant_ct(), dtype=np.uint32)
    else:
        dataset_ids = [
            dataset.pvar.get_variant_id(i).decode("utf8")
            for i in np.arange(dataset.pvar.get_variant_ct(), dtype=np.uint32)
        ]
        varset = set(variants)
        idx_variant = np.array(
            [i for i, x in enumerate(dataset_ids) if x in varset], dtype=np.uint32
        )

    ## Get chromosomes and number of blocks
    chromosomes = [
        dataset.pvar.get_variant_chrom(i).decode("utf8") for i in idx_variant
    ]
    chroms = list(set(chromosomes))  # unique chromosomes
    n_blocks = sum(
        (len([c for c in chromosomes if c == chrom]) + B - 1) // B for chrom in chroms
    )

    ## Perform level 0 ridge regression for all blocks
    Z_chroms = {}
    with tqdm(total=n_blocks, desc=desc, unit="block") as pbar:
        for chrom in chroms:
            idx_chrom = np.array(
                [i for i, c in enumerate(chromosomes) if c == chrom], dtype=np.uint32
            )
            blocks = [
                idx_variant[idx_chrom[i : i + B]] for i in range(0, len(idx_chrom), B)
            ]

            Z_blocks = []
            for block in blocks:
                Z_blocks.append(
                    _step0_block(
                        dataset,
                        Y,
                        Q,
                        idx_sample,
                        train_mask,
                        test_mask,
                        block,
                        h2_prior,
                    )
                )
                pbar.update(1)

            Z_chroms[chrom] = np.concatenate(Z_blocks, axis=2)

    return Z_chroms


def step0(
    datasets: list[LancData],
    Y: ArrayLike,
    X: Optional[ArrayLike],
    train_mask: ArrayLike,
    test_mask: ArrayLike,
    h2_prior: ArrayLike,
    B: int = 2000,
    idx_sample: Optional[ArrayLike] = None,
    variants: Optional[list[str]] = None,
) -> dict[str, NDArray]:
    """Perform level 0 ridge regressions

    Args:
        datasets: A list of LancData objects (likely one per chromosome)
        Y: A (N, P) jax array of phenotypes
        X: A (N, C) jax array of covariates (no intercept)
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D jax array of prior values for snp heritability
        B: The number of variants per block
        idx_sample: An optional (N_sub,) jax array with indices of samples to include
        variants: A list of variant IDs to include in the analysis. If not provided, all variants are used

    Returns:
        Z: A dict where keys are chromosomes and values are (N, P, N_blocks) numpy arrays of step 0 predictions
    """
    (Y, X, train_mask, test_mask, h2_prior, idx_sample) = validate_step0_inputs(
        datasets, Y, X, train_mask, test_mask, h2_prior, B, idx_sample, variants
    )

    Q, _ = jnp.linalg.qr(X, mode="reduced")
    Y = stdize(Y - (Q @ (Q.T @ Y)))

    ## Perform step 0 for each dataset
    Z_datasets = []
    for ds in datasets:
        pgen_path = ds.plink_prefix + ".pgen"
        desc = f"Getting step 0 predictions for file: {pgen_path}"
        Z_dataset = _step0_dataset(
            ds,
            Y,
            Q,
            idx_sample,
            train_mask,
            test_mask,
            B,
            variants,
            h2_prior,
            desc=desc,
        )
        Z_datasets.append(Z_dataset)

    chrom_keys = sorted({k for d in Z_datasets for k in d})
    Z = {
        chrom: np.concatenate([d[chrom] for d in Z_datasets if chrom in d], axis=2)
        for chrom in chrom_keys
    }

    return Z
