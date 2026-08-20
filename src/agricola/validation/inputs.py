# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Input validation for steps 1 and 2."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxtyping import Array, ArrayLike
from lanctools import LancData

from ..numerical.linear_algebra import assert_covar_full_rank, stdize
from ..types import TestType, TraitType


def _validate_datasets(datasets: list[LancData]) -> None:
    if not isinstance(datasets, (list, tuple)):
        raise TypeError(f"datasets must be a list of LancData, got {type(datasets)}")
    for i, ds in enumerate(datasets):
        if not isinstance(ds, LancData):
            raise TypeError(f"datasets[{i}] must be LancData, got {type(ds)}")


def _prepare_y(Y: ArrayLike) -> tuple[Array, int]:
    Y = jnp.asarray(Y)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N, P), got shape {Y.shape}")
    Y_means = jnp.nanmean(Y, axis=0)
    return jnp.where(jnp.isnan(Y), Y_means, Y), Y.shape[0]


def _prepare_x(X: ArrayLike | None, N: int) -> Array:
    if X is None:
        X = jnp.ones((N, 1), dtype=float)
    else:
        X = jnp.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N, C), got shape {X.shape}")
        if X.shape[0] != N:
            raise ValueError(f"X.shape[0] must match Y.shape[0], got {X.shape[0]} vs {N}")
        X = jnp.concatenate([jnp.ones((N, 1), dtype=float), X], axis=1)
    X = stdize(X)
    assert_covar_full_rank(X)
    return X


def _prepare_masks(train_mask: ArrayLike, test_mask: ArrayLike, N: int):
    if train_mask is not None:
        train_mask = jnp.asarray(train_mask)
    if test_mask is not None:
        test_mask = jnp.asarray(test_mask)
    if not (train_mask is None or test_mask is None):
        if train_mask.ndim != 2 or test_mask.ndim != 2 or train_mask.shape != test_mask.shape:
            raise ValueError("train_mask and test_mask must be 2D (N, K) with the same shape")
        if train_mask.shape[0] != N or test_mask.shape[0] != N:
            raise ValueError("train_mask/test_mask must match N of Y")
    return train_mask, test_mask


def _prepare_h2(h2_prior: ArrayLike) -> Array:
    h2_prior = jnp.asarray(h2_prior)
    if h2_prior.ndim != 1:
        raise ValueError(f"h2_prior must be 1D, got shape {h2_prior.shape}")
    if not jnp.all((0 < h2_prior) & (h2_prior < 1)):
        raise ValueError("h2_prior values must be in the open interval (0, 1)")
    return h2_prior


def _validate_b(B: int) -> None:
    if not isinstance(B, int) or B <= 0:
        raise ValueError(f"B must be a positive integer, got {B}")


def _validate_variants(variants: list[str] | None) -> None:
    if variants is not None and (
        not isinstance(variants, (list, tuple)) or not all(isinstance(v, str) for v in variants)
    ):
        raise TypeError("variants must be a list of strings")


def _prepare_idx_sample(idx_sample: ArrayLike | None, N_pgen: int) -> Array | None:
    if idx_sample is None:
        return None
    idx_sample = jnp.asarray(idx_sample)
    if idx_sample.ndim != 1:
        raise TypeError("idx_sample must be 1D")
    if idx_sample.dtype != np.uint32:
        raise TypeError("idx_sample must have dtype numpy.uint32")
    if not set(np.asarray(idx_sample)).issubset(np.arange(N_pgen, dtype=np.uint32)):
        raise ValueError("idx_sample outside range of N samples")
    return idx_sample


def validate_level0_inputs(
    datasets: list[LancData],
    Y: ArrayLike,
    X: ArrayLike | None,
    train_mask: ArrayLike,
    test_mask: ArrayLike,
    h2_prior: ArrayLike,
    B: int = 2000,
    idx_sample: ArrayLike | None = None,
    variants: list[str] | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array | None]:
    """Validate input data for level0"""
    ## genotype/lanc data
    _validate_datasets(datasets)

    ## Y
    Y, N = _prepare_y(Y)

    ## X
    X = _prepare_x(X, N)

    train_mask, test_mask = _prepare_masks(train_mask, test_mask, N)

    ## H2
    h2_prior = _prepare_h2(h2_prior)

    ## B
    _validate_b(B)

    ## variants
    _validate_variants(variants)

    ## samples
    N_pgen = datasets[0].pgen.get_raw_sample_ct()
    if idx_sample is not None:
        idx_sample = _prepare_idx_sample(idx_sample, N_pgen)

    return (Y, X, train_mask, test_mask, h2_prior, idx_sample)


def validate_level1_inputs(
    Y: ArrayLike,
    X: ArrayLike | None,
    phenotypes: list[str],
    train_mask: ArrayLike,
    test_mask: ArrayLike,
    h2_prior: ArrayLike,
    trait_type: str | TraitType,
) -> tuple[Array, Array, Array, Array, Array, TraitType]:
    """Validate input data for level1"""
    ## Y
    Y, N = _prepare_y(Y)

    if len(phenotypes) != Y.shape[1]:
        raise ValueError(f"phenotype has length {len(phenotypes)}, but Y has {Y.shape[1]} columns")

    ## X
    X = _prepare_x(X, N)

    train_mask, test_mask = _prepare_masks(train_mask, test_mask, N)

    ## H2
    h2_prior = _prepare_h2(h2_prior)

    return (Y, X, train_mask, test_mask, h2_prior, TraitType(trait_type))


def validate_step2_inputs(
    datasets: list[LancData],
    Y: ArrayLike,
    X: ArrayLike | None,
    phenotypes: list[str],
    step1_predictions: dict[str, pd.DataFrame] | None,
    B: int,
    idx_sample: ArrayLike | None,
    variants: list[str] | None,
    test_type: str | TestType,
    trait_type: str | TraitType,
) -> tuple[Array, Array, dict[str, np.ndarray] | None, Array | None, TestType, TraitType]:
    """Validate input data for step1"""

    ## Y
    Y, N = _prepare_y(Y)

    if len(phenotypes) != Y.shape[1]:
        raise ValueError(f"phenotype has length {len(phenotypes)}, but Y has {Y.shape[1]} columns")

    X = _prepare_x(X, N)

    # datasets
    _validate_datasets(datasets)
    N_pred = None
    P_pred = None
    step1_predictions_np = None
    if step1_predictions is not None:
        step1_predictions_np = {}
        for chrom, pred_chrom in step1_predictions.items():
            if not isinstance(pred_chrom, pd.DataFrame):
                raise TypeError(
                    f"step1_predictions[{chrom}] must be a pandas DataFrame, got {type(pred_chrom)}"
                )
            if pred_chrom.ndim != 2:
                raise ValueError(
                    f"step1_predictions[{chrom}] must be 2D (N, P), got shape {pred_chrom.shape}"
                )

            n_chrom, p_chrom = pred_chrom.shape

            if N_pred is None:
                N_pred = n_chrom
                P_pred = p_chrom
            else:
                if n_chrom != N_pred:
                    raise ValueError(
                        f"All step1_predictions arrays must have same N; got {N_pred} "
                        f"vs {n_chrom} in step1_predictions[{chrom}]"
                    )
                if p_chrom != P_pred:
                    raise ValueError(
                        f"All step1_predictions arrays must have same P; got {P_pred} "
                        f"vs {p_chrom} in step1_predictions[{chrom}]"
                    )
            step1_predictions_np[chrom] = step1_predictions[chrom][phenotypes].to_numpy()

        if N_pred != N:
            raise ValueError(f"step1_predictions arrays have N={N_pred} but Y has N={N}")

    ## B
    _validate_b(B)

    ## variants
    _validate_variants(variants)

    ## samples
    N_pgen = datasets[0].pgen.get_raw_sample_ct()
    if idx_sample is not None:
        idx_sample = _prepare_idx_sample(idx_sample, N_pgen)

    return (
        Y,
        X,
        step1_predictions_np,
        idx_sample,
        TestType(test_type),
        TraitType(trait_type),
    )
