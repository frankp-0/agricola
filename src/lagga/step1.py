# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Level-1 whole genome predictions.

This module performs "step 1" of lagga, combining the "level 0" and "level 1"
steps . The entry-point for this module is the `step1` function.
"""

from jaxtyping import ArrayLike
from numpy.typing import NDArray
from typing import Optional
from ._internal.level0 import level0
from ._internal.level1 import level1
from lanctools import LancData


def step1(
    datasets: list[LancData],
    Y: ArrayLike,
    X: Optional[ArrayLike],
    train_mask: ArrayLike,
    test_mask: ArrayLike,
    h2_prior: ArrayLike,
    trait_type: str,
    loocv: bool = False,
    B: int = 2000,
    idx_sample: Optional[ArrayLike] = None,
    variants: Optional[list[str]] = None,
) -> dict[str, NDArray]:
    """Perform lagga step 1

    Args:
        datasets: A list of LancData objects (either single object or one
            per-chromosome)
        Y: A (N, P) jax array of phenotypes
        X: A (N, C) jax array of covariates (no intercept)
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D jax array of prior values for snp heritability
        trait_type: Either "qt" or "bt"
        loocv: A boolean indicating whether to perform LOOCV instead of standard
            cross validation. Ignored for trait_type="qt".
        B: The number of variants per block
        idx_sample: An optional (N_sub,) jax array with indices of samples to include
        variants: A list of variant IDs to include in the analysis. If not provided, all variants are used

    Returns:
        A dict where keys are chromosomes and values are (N, P) numpy arrays of level 1 predictions
    """
    Z = level0(datasets, Y, X, train_mask, test_mask, h2_prior, B, idx_sample, variants)
    step1_predictions = level1(
        Z, Y, X, train_mask, test_mask, h2_prior, trait_type, loocv
    )
    return step1_predictions
