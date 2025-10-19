import jax
import jax.numpy as jnp
import numpy as np


def _stdize(X: jnp.ndarray) -> jnp.ndarray:
    """Safely standardizes an ndarray along the first dimension"""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = jnp.where(std == 0, 1, std)
    result = (X - mean) / std
    return result


def _get_cv_mask(n, k, key):
    idx = jax.random.permutation(key, n)
    fold_size = n // k
    folds = [idx[i * fold_size : (i + 1) * fold_size] for i in range(k)]

    train_mask = np.zeros((n, k), dtype=jnp.bool)
    test_mask = np.zeros((n, k), dtype=jnp.bool)
    for fold in range(k):
        idx_train = jnp.concatenate([folds[i] for i in range(k) if i != fold])
        test_mask[folds[fold], fold] = True
        train_mask[idx_train, fold] = True

    # (n, k) booleans
    return train_mask, test_mask
