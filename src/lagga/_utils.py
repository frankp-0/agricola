# MIT License
# Copyright (c) 2025 Franklin Ockerman
# See LICENSE.txt file for full license text

import jax
from jaxtyping import Array
import jax.numpy as jnp
from lanctools import LancData
import numpy as np
from numpy.typing import NDArray


def stdize(X: Array) -> Array:
    """Safely standardizes an ndarray along the first dimension.

    For columns with standard deviation of 0, this function does NOT subtract
    the mean. This behavior is intended to preserve an intercept, for example.
    """
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    mean = jnp.where(std == 0, 0, mean)
    std = jnp.where(std == 0, 1, std)
    result = (X - mean) / std
    return result


def get_cv_mask(n: int, k: int, key: Array) -> tuple[Array, Array]:
    """Generate boolean train/test masks for k-fold cross-validation."""
    idx = jax.random.permutation(key, n)
    fold_sizes = jnp.full(k, n // k)
    fold_sizes = fold_sizes.at[: n % k].add(1)
    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(idx[start : start + size])
        start += size

    train_mask = jnp.zeros((n, k), dtype=bool)
    test_mask = jnp.zeros((n, k), dtype=bool)

    for fold in range(k):
        test_idx = folds[fold]
        train_idx = jnp.concatenate([folds[i] for i in range(k) if i != fold])
        test_mask = test_mask.at[test_idx, fold].set(True)
        train_mask = train_mask.at[train_idx, fold].set(True)

    return train_mask, test_mask


def assert_covar_full_rank(X: jnp.ndarray, rtol: float = 1e-8):
    """Raises ValueError if X does not have full column rank.

    Assumes X is standardized (columns ~ unit variance).
    """
    S = jnp.linalg.svd(X, compute_uv=False)
    rank = jnp.sum(S > rtol * S[0])

    if rank < X.shape[1]:
        raise ValueError(
            f"Collinearity detected in : rank={rank}, n_features={X.shape[1]}"
        )


def get_geno_lanc_deconv(
    dataset: LancData, indices: NDArray[np.uint32]
) -> tuple[Array, Array]:
    geno = jnp.asarray(dataset.get_geno(indices))
    lanc = jnp.asarray(dataset.get_lanc(indices))
    ancestries = jnp.arange(len(dataset.ancestries))
    left_haps_mask = (lanc[:, :, 0:1] == ancestries[None, None, :]).astype(np.int32)
    right_haps_mask = (lanc[:, :, 1:2] == ancestries[None, None, :]).astype(np.int32)
    geno_masked = left_haps_mask * geno[:, :, 0:1] + right_haps_mask * geno[:, :, 1:2]
    lanc_masked = left_haps_mask + right_haps_mask
    return geno_masked, lanc_masked
