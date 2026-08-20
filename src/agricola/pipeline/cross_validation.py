# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Cross-validation split construction."""

import jax
import jax.numpy as jnp
from jaxtyping import Array


def get_cv_mask(n: int, k: int, key: Array) -> tuple[Array, Array]:
    """Generate boolean train/test masks for k-fold cross-validation."""
    indices = jax.random.permutation(key, n)
    fold_sizes = jnp.full(k, n // k).at[: n % k].add(1)
    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start : start + size])
        start += size

    train_mask = jnp.zeros((n, k), dtype=bool)
    test_mask = jnp.zeros((n, k), dtype=bool)
    for fold in range(k):
        test_indices = folds[fold]
        train_indices = jnp.concatenate([folds[i] for i in range(k) if i != fold])
        test_mask = test_mask.at[test_indices, fold].set(True)
        train_mask = train_mask.at[train_indices, fold].set(True)
    return train_mask, test_mask
