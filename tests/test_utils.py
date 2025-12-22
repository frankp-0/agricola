import pytest
import jax.numpy as jnp
import jax
from lagga._utils import stdize, get_cv_mask, assert_covar_full_rank


def test_stdize_basic():
    X = jnp.array([[1, 2], [3, 4], [5, 6]])
    X_std = stdize(X)

    assert jnp.allclose(X_std.mean(axis=0), 0)
    assert jnp.allclose(X_std.std(axis=0), 1)


def test_stdize_zero_variance():
    X = jnp.array([[1, 2], [1, 3], [1, 4]])
    X_std = stdize(X)

    assert jnp.all(X_std[:, 0] == 1)
    assert jnp.allclose(X_std[:, 1].mean(), 0)
    assert jnp.allclose(X_std[:, 1].std(), 1)


def test_get_cv_mask():
    n = 12
    k = 5
    key = jax.random.PRNGKey(0)

    train_mask, test_mask = get_cv_mask(n, k, key)

    ## Shapes
    assert train_mask.shape == (n, k)
    assert test_mask.shape == (n, k)

    ## Train + test should cover all indices per fold
    for fold in range(k):
        combined = train_mask[:, fold] | test_mask[:, fold]
        assert combined.all()  # every index is either in train or test

    ## No overlap between train and test
    for fold in range(k):
        overlap = train_mask[:, fold] & test_mask[:, fold]
        assert not overlap.any()

    ## Each index appears in exactly one test fold
    test_counts = test_mask.sum(axis=1)
    assert jnp.all(test_counts == 1)


def test_assert_covar_full_rank_full_rank():
    X = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert_covar_full_rank(X)


def test_assert_covar_full_rank_rank_deficient():
    X = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        assert_covar_full_rank(X)
