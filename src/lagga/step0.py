import jax
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
from .data import GenoAncestryDataset
from ._utils import _stdize
from tqdm import tqdm
from .models import _ridge_masked
from typing import Optional


def _step0_block(
    dataset: GenoAncestryDataset,
    Y: Array,
    Q: Array,
    train_mask: Array,
    test_mask: Array,
    block: np.ndarray,
    h2_prior: Array,
):
    """Get level 0 predictions for a single block

    Args:
        dataset: A  GenoAncestryDataset object
        Y: A (N, P) jax array of (residualized, standardized) phenotypes
        Q: A (N, C) jax array with the Q matrix in the QR decomposition of the covariates
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        block: A (B,) ndarray with indices of variants in the block
        h2_prior: A 1D jax array of prior values for snp heritability

    Returns:
        Z_block: A (N, B * len(h2_prior), P) jax array of predictions
    """
    ## Standardize genotype block and residualize by covariates
    G = dataset.get_geno(block)
    G = G[:, :, 0] + G[:, :, 1]
    G = _stdize(G - (Q @ (Q.T @ G)))  # pyright: ignore

    ## Calculate penalties based on prior heritability
    B = G.shape[1]
    alphas = B * (1 - h2_prior) / h2_prior

    ## Perform ridge regression
    ridge = jax.vmap(_ridge_masked, in_axes=(None, None, 1, 1, None))
    Z_block = jnp.sum(ridge(G, Y, train_mask, test_mask, alphas), axis=0)
    Z_block = np.asarray(_stdize(Z_block))
    return Z_block


def _step0_dataset(
    dataset: GenoAncestryDataset,
    Y: Array,
    Q: Array,
    train_mask: Array,
    test_mask: Array,
    B: int,
    variants: Optional[list[str]],
    h2_prior: Array,
    desc: str,
):
    """Get level 0 predictions for a dataset

    Args:
        dataset: A  GenoAncestryDataset object
        Y: A (N, P) jax array of (residualized, standardized) phenotypes
        Q: A (N, C) jax array with the Q matrix in the QR decomposition of the covariates
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        B: The number of variants per block
        variants: A list of variant IDs to include in the analysis. If not provided, all variants are used
        h2_prior: A 1D jax array of prior values for snp heritability
        desc: A string with the description for printing

    Returns:
        Z_chroms: A dict where keys are chromosomes and values are (N, N_predictors, P) jax arrays of step 0 predictions
    """

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

    chromosomes = [
        dataset.pvar.get_variant_chrom(i).decode("utf8") for i in idx_variant
    ]

    chroms = list(set(chromosomes))  # unique chromosomes

    n_blocks = sum(
        (len([c for c in chromosomes if c == chrom]) + B - 1) // B for chrom in chroms
    )

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
                    _step0_block(dataset, Y, Q, train_mask, test_mask, block, h2_prior)
                )
                pbar.update(1)

            Z_chroms[chrom] = np.concatenate(Z_blocks, axis=2)

    return Z_chroms


def step0(
    datasets: list[GenoAncestryDataset],
    Y: Array,
    X: Array,
    train_mask: Array,
    test_mask: Array,
    h2_prior: Array,
    B: int = 2000,
    variants: Optional[list[str]] = None,
):
    """Perform level 0 ridge regressions

    Args:
        datasets: A list of GenoAncestryDataset objects (likely one per chromosome)
        Y: A (N, P) jax array of phenotypes
        X: A (N, C) jax array of covariates
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D jax array of prior values for snp heritability
        B: The number of variants per block
        variants: A list of variant IDs to include in the analysis. If not provided, all variants are used

    Returns:
        Z: A dict where keys are chromosomes and values are (N, N_blocks) jax arrays of step 0 predictions
    """
    Q, _ = jnp.linalg.qr(X, mode="reduced")  # pyright: ignore
    Y = _stdize(Y - (Q @ (Q.T @ Y)))

    Z_datasets = []
    for ds in datasets:
        pgen_path = ds.plink_prefix + ".pgen"
        desc = f"Getting step 0 predictions for file: {pgen_path}"
        Z_dataset = _step0_dataset(
            ds, Y, Q, train_mask, test_mask, B, variants, h2_prior, desc=desc
        )
        Z_datasets.append(Z_dataset)

    chrom_keys = sorted({k for d in Z_datasets for k in d})
    Z = {
        chrom: np.concatenate([d[chrom] for d in Z_datasets if chrom in d], axis=2)
        for chrom in chrom_keys
    }

    return Z
