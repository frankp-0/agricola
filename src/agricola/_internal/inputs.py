# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Input validation for steps 1 and 2."""

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
import numpy as np
from typing import Optional
from lanctools import LancData
from .utils import stdize, assert_covar_full_rank, TraitType, TestType


def validate_level0_inputs(
    datasets: list[LancData],
    Y: ArrayLike,
    X: Optional[ArrayLike],
    h2_prior: ArrayLike,
    B: int = 2000,
    idx_sample: Optional[ArrayLike] = None,
    variants: Optional[list[str]] = None,
) -> tuple[Array, Array, Array, Optional[Array]]:
    """Validate input data for level0"""
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

    Y_means = jnp.nanmean(Y, axis=0)
    Y = jnp.where(jnp.isnan(Y), Y_means, Y)
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

    return (Y, X, h2_prior, idx_sample)


def validate_level1_inputs(
    Y: ArrayLike,
    X: Optional[ArrayLike],
    train_mask: Optional[ArrayLike],
    test_mask: Optional[ArrayLike],
    h2_prior: ArrayLike,
    trait_type: str,
) -> tuple[Array, Array, Array, Array, Array, TraitType]:
    """Validate input data for level1"""
    ## Y
    Y = jnp.asarray(Y)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N, P), got shape {Y.shape}")

    Y_means = jnp.nanmean(Y, axis=0)
    Y = jnp.where(jnp.isnan(Y), Y_means, Y)
    N, _ = Y.shape

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
    if train_mask is not None:
        train_mask = jnp.asarray(train_mask)
    if test_mask is not None:
        test_mask = jnp.asarray(test_mask)

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
    if train_mask is None:
        train_mask = jnp.array([1])
    if test_mask is None:
        test_mask = jnp.array([1])

    ## H2
    h2_prior = jnp.asarray(h2_prior)
    if h2_prior.ndim != 1:
        raise ValueError(f"h2_prior must be 1D, got shape {h2_prior.shape}")
    if not jnp.all((0 < h2_prior) & (h2_prior < 1)):
        raise ValueError("h2_prior values must be in the open interval (0, 1)")

    return (Y, X, train_mask, test_mask, h2_prior, TraitType(trait_type))


def validate_step2_inputs(
    datasets: list[LancData],
    Y: ArrayLike,
    X: Optional[ArrayLike],
    step1_predictions: dict[str, np.ndarray],
    out_prefixes: list[str],
    B: int,
    idx_sample: Optional[ArrayLike],
    variants: Optional[list[str]],
    test_type: str,
    trait_type: str,
) -> tuple[Array, Array, Optional[Array], TestType, TraitType]:
    """Validate input data for step1"""

    ## Y
    Y = jnp.asarray(Y)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N, P), got shape {Y.shape}")

    Y_means = jnp.nanmean(Y, axis=0)
    Y = jnp.where(jnp.isnan(Y), Y_means, Y)
    N, P = Y.shape

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

    # datasets
    if not isinstance(datasets, (list, tuple)):
        raise TypeError(f"datasets must be a list of LancData, got {type(datasets)}")
    for i, ds in enumerate(datasets):
        if not isinstance(ds, LancData):
            raise TypeError(f"datasets[{i}] must be LancData, got {type(ds)}")
    N_pred = None
    P_pred = None
    for chrom, pred_chrom in step1_predictions.items():
        if not isinstance(pred_chrom, (np.ndarray, jnp.ndarray)):
            raise TypeError(
                f"step1_predictions[{chrom}] must be a numpy/jax array, got {type(pred_chrom)}"
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
                    f"All step1_predictions arrays must have same N; got {N_pred} vs {n_chrom} in step1_predictions[{chrom}]"
                )
            if p_chrom != P_pred:
                raise ValueError(
                    f"All step1_predictions arrays must have same P; got {P_pred} vs {p_chrom} in step1_predictions[{chrom}]"
                )
    if not len(out_prefixes) == len(datasets):
        raise ValueError("out_prefixes and datasets must have same number of elements")

    if N_pred != N:
        raise ValueError(f"step1_predictions arrays have N={N_pred} but Y has N={N}")

    if P_pred != P:
        raise ValueError(f"step1_predictions arrays have P={P_pred} but Y has P={P}")

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

    return (Y, X, idx_sample, TestType(test_type), TraitType(trait_type))
