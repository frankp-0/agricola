# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Level-0 block-wise whole-genome ridge predictions.

This module performs "level 0" of lagga step1. It splits the genome into blocks and
performs a ridge regression within each block. It returns block-wise predictions
for each trait across a sequence of heritability priors. The entry-point for this
module is the `level0` function.
"""

import os
import tempfile
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
import numpy as np
from tqdm import tqdm
from typing import Optional
from lanctools import LancData
from .utils import stdize, get_geno_lanc_deconv
from .models import ridge
from .inputs import validate_level0_inputs
from numpy.typing import NDArray


def _level0_block(
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
    G = stdize(G - (Q @ (Q.T @ G)))

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


def level0(
    datasets: list[LancData],
    Y: ArrayLike,
    X: Optional[ArrayLike],
    train_mask: ArrayLike,
    test_mask: ArrayLike,
    h2_prior: ArrayLike,
    B: int = 2000,
    idx_sample: Optional[ArrayLike] = None,
    variants: Optional[list[str]] = None,
    level0_dir: Optional[str] = None,
) -> dict[str, str]:
    """Perform level 0 ridge regressions

    Args:
        datasets: A list of LancData objects (either single object or one
            per-chromosome)
        Y: A (N, P) jax array of phenotypes
        X: A (N, C) jax array of covariates (no intercept)
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D jax array of prior values for snp heritability
        B: The number of variants per block
        idx_sample: An optional (N_sub,) jax array with indices of samples to include
        variants: A list of variant IDs to include in the analysis. If not provided, all variants are used
        level0_dir: The directory where level 0 predictions are written

    Returns:
        level0_files: A dict where keys are chromosomes and values are files
            containing level 0 predictions
    """
    (Y, X, train_mask, test_mask, h2_prior, idx_sample) = validate_level0_inputs(
        datasets, Y, X, train_mask, test_mask, h2_prior, B, idx_sample, variants
    )

    Q, _ = jnp.linalg.qr(X, mode="reduced")
    Y = stdize(Y - (Q @ (Q.T @ Y)))

    N, P = Y.shape
    K = len(h2_prior)

    ## Perform level 0 for each dataset
    if level0_dir is None:
        tmp = tempfile.TemporaryDirectory()
        level0_dir = tmp.name

    os.makedirs(level0_dir, exist_ok=True)

    level0_files = {}
    for ds in datasets:
        pgen_path = ds.plink_prefix + ".pgen"
        desc = f"Getting level 0 predictions for file: {pgen_path}"
        if variants is None:
            idx_variant = np.arange(ds.pvar.get_variant_ct(), dtype=np.uint32)
        else:
            dataset_ids = [
                ds.pvar.get_variant_id(i).decode("utf8")
                for i in np.arange(ds.pvar.get_variant_ct(), dtype=np.uint32)
            ]
            varset = set(variants)
            idx_variant = np.array(
                [i for i, x in enumerate(dataset_ids) if x in varset], dtype=np.uint32
            )

        chromosomes = [ds.pvar.get_variant_chrom(i).decode("utf8") for i in idx_variant]
        chroms = list(set(chromosomes))
        for chrom in chroms:
            idx_chrom = np.array(
                [i for i, c in enumerate(chromosomes) if c == chrom], dtype=np.uint32
            )

            blocks = [
                idx_variant[idx_chrom[i : i + B]] for i in range(0, len(idx_chrom), B)
            ]

            n_blocks = len(blocks)

            fname = os.path.join(level0_dir, f"{chrom}.npy")
            Z = np.lib.format.open_memmap(
                fname, mode="w+", dtype=np.float32, shape=(N, P, n_blocks * K)
            )

            col0 = 0
            with tqdm(total=n_blocks, desc=f"{desc} chr{chrom}", unit="block") as pbar:
                for block in blocks:
                    Z_block = _level0_block(
                        ds, Y, Q, idx_sample, train_mask, test_mask, block, h2_prior
                    )

                    Z[:, :, col0 : col0 + K] = Z_block
                    col0 += K
                    pbar.update(1)

            Z.flush()

            level0_files[chrom] = fname

    return level0_files
