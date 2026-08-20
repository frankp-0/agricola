# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Level-1 whole genome predictions.

This module performs "step 1" of agricola, combining the "level 0" and "level 1"
steps . The entry-point for this module is the `step1` function.
"""

import tempfile
from contextlib import nullcontext
from pathlib import Path
from jaxtyping import ArrayLike
from typing import Optional
from pandas import DataFrame
from .level0 import level0
from .level1 import level1
from lanctools import LancData


def step1(
    datasets: list[LancData],
    Y: ArrayLike,
    X: Optional[ArrayLike],
    phenotypes: list[str],
    train_mask: ArrayLike,
    test_mask: ArrayLike,
    h2_prior: ArrayLike,
    trait_type: str,
    loocv: bool = False,
    B: int = 1000,
    idx_sample: Optional[ArrayLike] = None,
    variants: Optional[list[str]] = None,
    level0_dir: Optional[str] = None,
    prune_blocks: bool = True,
) -> dict[str, DataFrame]:
    """Perform agricola step 1

    Args:
        datasets: A list of LancData objects (either single object or one
            per-chromosome)
        Y: A (N, P) jax array of phenotypes
        X: A (N, C) jax array of covariates (no intercept)
        phenotypes: A list of phenotype names, ordered as the columns of Y
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D jax array of prior values for snp heritability
        trait_type: Either "qt" or "bt"
        loocv: A boolean indicating whether to perform LOOCV instead of standard
            cross validation. Ignored for trait_type="qt".
        B: The number of variants per block
        idx_sample: An optional (N_sub,) jax array with indices of samples to include
        variants: A list of variant IDs to include in the analysis. If not provided, all variants are used
        level0_dir: The directory where level 0 predictions are written
        prune_blocks: Whether to sample variants in a dataset in level 0 so that
            n_variants (mod B) = 0. This will improve speed with JIT compilations

    Returns:
        A dict where keys are chromosomes and values are (N, P) pandas DataFrames of level 1 predictions
    """
    directory_context = (
        tempfile.TemporaryDirectory()
        if level0_dir is None
        else nullcontext(level0_dir)
    )
    with directory_context as working_dir:
        level0_path = Path(working_dir)
        level0_path.mkdir(parents=True, exist_ok=True)

        level0_files = level0(
            datasets,
            Y,
            X,
            phenotypes,
            train_mask,
            test_mask,
            h2_prior,
            B,
            idx_sample,
            variants,
            str(level0_path),
            prune_blocks,
        )

        return level1(
            level0_files,
            Y,
            X,
            phenotypes,
            train_mask,
            test_mask,
            h2_prior,
            trait_type,
            loocv,
        )
