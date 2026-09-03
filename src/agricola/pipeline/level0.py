# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Level-0 block-wise whole-genome ridge predictions.

This module performs "level 0" of agricola step1. It splits the genome into blocks and
performs a ridge regression within each block. It returns block-wise predictions
for each trait across a sequence of heritability priors. The entry-point for this
module is the `level0` function.
"""

import logging
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from lanctools import LancData
from numpy.typing import NDArray
from tqdm import tqdm

from ..io.variants import get_variant_indices, group_variant_indices_by_chromosome
from ..models.ridge import ridge, ridge_lowmem, ridge_lowmem_folds
from ..numerical.linear_algebra import stdize
from ..validation.inputs import validate_level0_inputs

logger = logging.getLogger(__name__)

### ─────────────────────────────────────────────────────────────
### Helpers
### ─────────────────────────────────────────────────────────────


def _level0_ridge(G, Y, Q, train_mask, test_mask, M, h2_prior, memory_mode: str = "standard"):
    ## Standardize genotype block and residualize by covariates
    G = G[:, :, 0] + G[:, :, 1]
    G = stdize(G - (Q @ (Q.T @ G)))

    ## Calculate penalties based on prior heritability
    alphas = M * (1 - h2_prior) / h2_prior

    if memory_mode == "lowest":
        ridge_beta = ridge_lowmem_folds(G, Y, train_mask, alphas)
    elif memory_mode == "low":
        ridge_beta = ridge_lowmem(G, Y, train_mask, alphas)
    else:
        ridge_beta = ridge(G, Y, train_mask, alphas)

    ridge_Z = jnp.einsum("nb,akbp->nkpa", G, ridge_beta) * test_mask[:, :, None, None]
    Z_block = stdize(jnp.sum(ridge_Z, axis=1))
    return Z_block


_level0_ridge_jit = jax.jit(_level0_ridge, static_argnames=("memory_mode",))


def _level0_block(
    dataset: LancData,
    Y: Array,
    Q: Array,
    train_mask: Array,
    test_mask: Array,
    idx_sample: Array | None,
    block: NDArray,
    h2_prior: Array,
    M: int,
    B: int,
    memory_mode: str = "standard",
) -> NDArray:
    """Get level 0 predictions for a single block

    Args:
        dataset: A  LancData object
        Y: A (N, P) jax array of (residualized, standardized) phenotypes
        Q: A (N, C) jax array with the Q matrix in the QR decomposition of the covariates
        train_mask: An (N, K) ArrayLike indicating training set status for each set k in 1, ..., K
        test_mask: An (N, K) ArrayLike indicating test set status for each set k in 1, ..., K
        idx_sample: An optional (N_sub,) jax array with indices of samples to include
        block: A (B,) ndarray with indices of variants in the block
        h2_prior: A 1D jax array of prior values for snp heritability
        M: The total number of variants used in step 1

    Returns:
        Z_block: A (N, P, len(h2_prior)) numpy array of predictions
    """
    block_start = time.perf_counter()

    load_start = time.perf_counter()
    geno = dataset.get_geno(block)
    load_elapsed = time.perf_counter() - load_start

    transfer_start = time.perf_counter()
    G = jnp.asarray(geno)
    G.block_until_ready()
    transfer_elapsed = time.perf_counter() - transfer_start

    if idx_sample is not None:
        G = G[idx_sample]

    ridge_start = time.perf_counter()
    if block.shape[0] == B:
        ridge_name = "_level0_ridge_jit"
        Z_block = _level0_ridge_jit(G, Y, Q, train_mask, test_mask, M, h2_prior, memory_mode)
    else:
        ridge_name = "_level0_ridge"
        Z_block = _level0_ridge(G, Y, Q, train_mask, test_mask, M, h2_prior, memory_mode)
    Z_block.block_until_ready()
    ridge_elapsed = time.perf_counter() - ridge_start

    conversion_start = time.perf_counter()
    Z_block = np.asarray(Z_block)
    conversion_elapsed = time.perf_counter() - conversion_start
    block_elapsed = time.perf_counter() - block_start
    logger.debug(
        "Level 0 block (%d variants): total=%.6fs, get_geno=%.6fs, "
        "genotype transfer=%.6fs, %s execution=%.6fs, "
        "Z_block conversion to NumPy=%.6fs",
        block.shape[0],
        block_elapsed,
        load_elapsed,
        transfer_elapsed,
        ridge_name,
        ridge_elapsed,
        conversion_elapsed,
    )
    return Z_block


def level0(
    datasets: list[LancData],
    Y: ArrayLike,
    X: ArrayLike | None,
    phenotypes: list[str],
    train_mask: ArrayLike,
    test_mask: ArrayLike,
    h2_prior: ArrayLike,
    B: int = 1000,
    idx_sample: ArrayLike | None = None,
    variants: list[str] | None = None,
    level0_dir: str | None = None,
    prune_blocks: bool | None = True,
    key: ArrayLike | None = None,
    memory_mode: str = "standard",
) -> dict[str, dict[str, str]]:
    """Perform level 0 ridge regressions

    Args:
        datasets: A list of LancData objects (either single object or one
            per-chromosome)
        Y: A (N, P) jax array of phenotypes
        X: A (N, C) jax array of covariates (no intercept)
        phenotypes: A list of phenotype names, ordered as the columns of Y
        train_mask: An (N, K) ArrayLike indicating training set status for each set k in 1, ..., K
        test_mask: An (N, K) ArrayLike indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D jax array of prior values for snp heritability
        B: The number of variants per block
        idx_sample: An optional (N_sub,) jax array with indices of samples to include
        variants: A list of variant IDs to include in the analysis. If not provided,
            all variants are used
        level0_dir: The directory where level 0 predictions are written
        prune_blocks: Whether to sample variants in a dataset so that n_variants
            (mod B) = 0. This will improve speed with JIT compilations
        key: Optional JAX PRNG key used for variant subsampling
    Returns:
        level0_files: A two-level dict. The outer keys are phenotypes, the inner
            keys are chromosomes, and the values are paths to .npy files containing
            (N, n_blocks) level 0 predictions.
    """
    (Y, X, train_mask, test_mask, h2_prior, idx_sample) = validate_level0_inputs(
        datasets, Y, X, train_mask, test_mask, h2_prior, B, idx_sample, variants
    )

    Q, _ = jnp.linalg.qr(X, mode="reduced")
    Y = stdize(Y - (Q @ (Q.T @ Y)))

    N, _ = Y.shape
    K = len(h2_prior)

    ## Perform level 0 for each dataset
    output_dir = (
        Path(level0_dir)
        if level0_dir is not None
        else Path(tempfile.mkdtemp(prefix="agricola-level0-"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    level0_files_chrom = {}

    M = 0
    idx_variant_ds = {}
    keys = (
        jax.random.split(jax.random.key(0), len(datasets))
        if key is None
        else jax.random.split(jnp.asarray(key), len(datasets))
    )
    for i, ds in enumerate(datasets):
        idx_variant = get_variant_indices(ds, variants)
        if prune_blocks:
            M_ds = len(idx_variant)
            M_ds = M_ds - (M_ds % B)
            if not M_ds:
                M_ds = len(idx_variant)
            idx_variant = np.asarray(
                jax.random.choice(keys[i], idx_variant, shape=(M_ds,), replace=False)
            )

        idx_variant = np.sort(idx_variant)
        idx_variant_ds[i] = idx_variant
        M += idx_variant.shape[0]

    time_total_start = time.perf_counter()
    for i, ds in enumerate(datasets):
        pgen_path = ds.plink_prefix + ".pgen"
        time_ds_start = time.perf_counter()
        logger.info(f"Getting level 0 predictions for file: {pgen_path}")

        idx_variant = idx_variant_ds[i]
        variants_by_chromosome = group_variant_indices_by_chromosome(ds, idx_variant)
        for chrom, idx_chrom in variants_by_chromosome.items():
            blocks = [idx_chrom[i : i + B] for i in range(0, len(idx_chrom), B)]

            n_blocks = len(blocks)

            fnames = [str(output_dir / f"{pheno}_{chrom}.npy") for pheno in phenotypes]
            Zs = np.empty((N, len(phenotypes), n_blocks * K), dtype=float)

            col0 = 0
            with tqdm(total=n_blocks, desc=f"chr{chrom}", unit="block") as pbar:
                for block_index, block in enumerate(blocks):
                    Z_block = _level0_block(
                        ds,
                        Y,
                        Q,
                        train_mask,
                        test_mask,
                        idx_sample,
                        block,
                        h2_prior,
                        M,
                        B,
                        memory_mode,
                    )
                    update_start = time.perf_counter()
                    Zs[:, :, col0 : col0 + K] = Z_block
                    update_elapsed = time.perf_counter() - update_start
                    logger.debug(
                        "Level 0 block %d/%d: Zs update=%.6fs",
                        block_index + 1,
                        n_blocks,
                        update_elapsed,
                    )
                    col0 += K
                    pbar.update(1)

            for p in range(len(phenotypes)):
                np.save(fnames[p], Zs[:, p, :])

            level0_files_chrom[chrom] = fnames
        time_ds = str(timedelta(seconds=int(time.perf_counter() - time_ds_start)))
        logger.info(f"Elapsed time: {time_ds}\n")

    time_total = str(timedelta(seconds=int(time.perf_counter() - time_total_start)))
    logger.info(f"Step 1 Level 0 predictions completed in: {time_total}\n")
    level0_files = {
        k: {dk: dv[i] for dk, dv in level0_files_chrom.items()} for i, k in enumerate(phenotypes)
    }

    return level0_files
