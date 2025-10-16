import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
import numpy as np
from .data import GenoAncestryDataset
from ._utils import _stdize
from typing import Optional
from tqdm import tqdm

### ─────────────────────────────────────────────────────────────
### Math
### ─────────────────────────────────────────────────────────────


def _ridge_impl(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    alphas: jnp.ndarray,
    idx_train: jnp.ndarray | None = None,
):
    n, b = X.shape

    if idx_train is None:
        idx_train = jnp.arange(n, dtype=jnp.uint32)

    X_train = X[idx_train]
    Y_train = Y[idx_train]

    XtX = X_train.T @ X_train
    XtY = X_train.T @ Y_train
    I_ = jnp.eye(b, dtype=jnp.float32)

    def ridge_pred(alpha):
        A = XtX + alpha * I_
        beta = jnp.linalg.solve(A, XtY)
        return X @ beta

    preds = jax.vmap(ridge_pred)(alphas)
    preds = jnp.moveaxis(preds, 0, -1)

    return preds


def _ridge(
    G: Array,
    Y: Array,
    alphas: Array,
    idx_train: Optional[Array] = None,
    jit_enabled: bool = True,
) -> Array:
    """Get ridge regression predictions for multiple outcomes and penalties

    Call this function with jit_enabled=False for step 1 to avoid recompiling
    for each dataset/chromosome. Although training samples may be specified
    with idx_train, predictions will be returned for all samples

    Args:
        G: An (N, V) jax array of predictors
        Y: An (N, P) jax array of outcomes
        alphas: The ridge penalties
        idx_train: The indices of samples to use for training

    Returns:
        return_name: Return description

    Raises:
        exception name: Exception description
    """
    if jit_enabled:
        return jax.jit(_ridge_impl)(G, Y, alphas, idx_train)
    else:
        return _ridge_impl(G, Y, alphas, idx_train)


def _ridge_cv(G0: Array, Y: Array, alphas: Array, key: Array, k: int = 5) -> np.ndarray:
    """Get cross-validated ridge regression predictions

    Unlike _ridge, this assumes only a single outcome in Y. This is because the predictors
    X are outcome-specific.

    Args:
        G0: An (N, G) jax array of predictions from step 0
        Y: An (N, 1) jax array for a single outcome
        alphas: The ridge penalties
        key: Array representing PRNG key
        k: number of cross-validation folds

    Returns:
        An (N, 1) NumPy ndarray with the best cross-validated prediction
    """
    n_samples, n_pheno = Y.shape
    n_alpha = alphas.shape[0]

    idx = jax.random.permutation(key, n_samples)
    fold_size = n_samples // k
    folds = [idx[i * fold_size : (i + 1) * fold_size] for i in range(k)]

    cv_errors = jnp.zeros((n_pheno, n_alpha), dtype=jnp.float32)

    for fold in range(k):
        idx_test = folds[fold]
        idx_train = jnp.concatenate([folds[i] for i in range(k) if i != fold])
        fold_preds = _ridge(G0, Y, alphas, idx_train)[idx_test]
        errors = jnp.mean(
            (Y[idx_test, :, None] - fold_preds) ** 2, axis=0
        )  # shape (P, len(alphas))
        cv_errors += errors

    alpha_best_idx = jnp.argmin(cv_errors, axis=1)
    alpha_best = alphas[alpha_best_idx]

    preds_all = _ridge(G0, Y, alpha_best, jit_enabled=False)  # shape (N, P, 1)
    preds = preds_all[
        jnp.arange(n_samples)[:, None],
        jnp.arange(n_pheno)[None, :],
        alpha_best_idx[None, :],
    ]

    return np.array(preds)


### ─────────────────────────────────────────────────────────────
### Helper functions
### ─────────────────────────────────────────────────────────────


def _predict_block(
    dataset: GenoAncestryDataset,
    Y: Array,
    Q_covar: Array,
    block: np.ndarray,
    alphas: Array,
) -> np.ndarray:
    """Get ridge predictions for a block of variants

    Args:
        dataset: GenoAncestryDataset
        Y: (N, P) jax array of outcomes, already residualized by Q_covar
        Q_covar: (N, C) jax array. The orthogonal matrix Q in the QR
        decomposition of C covariates
        block: (V,) NumPy ndarray with indices for a block of variants
        alphas: The ridge regression penalties

    Returns:
        An (N, P,  len(alphas)) NumPy ndarray of ridge predictions
    """
    X_by_anc = dataset.get_lanc_geno(block)
    n_samples, n_variants, n_ancestries = X_by_anc.shape
    X = X_by_anc.reshape(n_samples, n_variants * n_ancestries)
    X_resid = _stdize(X - (Q_covar @ (Q_covar.T @ X)))
    preds = _stdize(_ridge(X_resid, Y, alphas, jit_enabled=True))
    return np.array(preds)


def _predict_dataset(
    dataset: GenoAncestryDataset,
    Y: Array,
    Q_covar: Array,
    B: int,
    alphas: Array,
    desc: str,
) -> dict[str, np.ndarray]:
    """Get ridge predictions for a dataset

    Args:
        dataset: GenoAncestryDataset
        Y: (N, P) jax array of outcomes, already residualized by Q_covar
        Q_covar: (N, C) jax array. The orthogonal matrix Q in the QR
        decomposition of the covariates
        alphas: The ridge regression penalties
        desc: A string describing the dataset, used for tracking progress

    Returns:
        A dict where keys are chromosomes and values are
        (N, P, len(alphas) * n_blocks) NumPy ndarrays with ridge predictions
    """
    idx_variant = np.arange(dataset.pvar.get_variant_ct(), dtype=np.uint32)

    chromosomes = [
        dataset.pvar.get_variant_chrom(i).decode("utf8") for i in idx_variant
    ]

    chroms = list(set(chromosomes))  # unique chromosomes

    n_blocks = sum(
        (len([c for c in chromosomes if c == chrom]) + B - 1) // B for chrom in chroms
    )

    predictions = {}
    with tqdm(total=n_blocks, desc=desc, unit="block") as pbar:
        for chrom in chroms:
            idx_chrom = np.array(
                [i for i, c in enumerate(chromosomes) if c == chrom], dtype=np.uint32
            )
            blocks = [idx_chrom[i : i + B] for i in range(0, len(idx_chrom), B)]

            preds = []
            for block in blocks:
                preds.append(_predict_block(dataset, Y, Q_covar, block, alphas))
                pbar.update(1)

            predictions[chrom] = np.concatenate(preds, axis=2)

    return predictions


def merge_predictions(
    predictions: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Merge step 0 predictions across datasets"""
    chrom_keys = sorted({k for d in predictions for k in d})
    return {
        chrom: np.concatenate([d[chrom] for d in predictions if chrom in d], axis=2)
        for chrom in chrom_keys
    }


### ─────────────────────────────────────────────────────────────
### Public API
### ─────────────────────────────────────────────────────────────


def step1(
    datasets: list[GenoAncestryDataset],
    Y: ArrayLike,
    covar: Optional[ArrayLike] = None,
    B: int = 2000,
    alphas0: ArrayLike = jnp.array([0.01, 0.25, 0.5, 0.75, 0.99], dtype=jnp.float32),
    alphas1: ArrayLike = jnp.array([0.01, 0.25, 0.5, 0.75, 0.99], dtype=jnp.float32),
    seed: int = 3432142,
) -> dict[str, np.ndarray]:
    """Run step 1 and return genomic predictions

    Args:
        datasets: A list of GenoAncestryDataset objects
        Y: An (N, P) array of outcomes
        covar: An (N, C) array of covariates. If not provided, defaults to intercept-only covariate
        B: The block size (max number of variants to read at once)
        alphas0: The ridge regression penalties for step 0
        alphas1: The ridge regression penalties for step 1
        seed: A seed for creating a PRNG key

    Returns:
        A dict where each key is a chromosome and each value is an (N, P)
        NumPy ndarray of genomic predictions
    """
    key = jax.random.key(seed)

    # Convert input to jax arrays
    Y = jnp.asarray(Y, dtype=jnp.float32)

    if covar is None:
        covar = jnp.ones((Y.shape[0], 1), dtype=jnp.float32)
    else:
        covar = jnp.hstack(
            [
                jnp.ones((Y.shape[0], 1), dtype=jnp.float32),
                jnp.asarray(covar, dtype=jnp.float32),
            ]
        )
    alphas0 = jnp.asarray(alphas0)
    alphas1 = jnp.asarray(alphas1)

    # Regress covariates from phenotypes
    Q_covar, _ = jnp.linalg.qr(covar, mode="reduced")
    Y_resid = _stdize(Y - (Q_covar @ (Q_covar.T @ Y)))

    dataset_predictions = []
    for ds in datasets:
        pgen_path = ds.plink_prefix + ".pgen"
        desc = f"Getting step 0 predictions for file: {pgen_path}"
        preds = _predict_dataset(ds, Y_resid, Q_covar, B, alphas0, desc=desc)
        dataset_predictions.append(preds)

    step0_predictions = merge_predictions(dataset_predictions)

    with tqdm(
        total=len(step0_predictions),
        desc="Getting step 1 predictions",
        unit="chromosomes",
    ) as pbar:
        step1_predictions = {}
        for chrom, G0 in step0_predictions.items():
            preds = np.empty((Y_resid.shape[0], Y_resid.shape[1]), dtype=np.float32)
            for j in range(Y.shape[1]):
                preds[:, j] = _ridge_cv(
                    jnp.asarray(G0[:, j, :]), Y_resid[:, j : j + 1], alphas1, key
                )[  # pyright: ignore
                    :, 0
                ]
            step1_predictions[chrom] = preds
            pbar.update(1)

    return step1_predictions
