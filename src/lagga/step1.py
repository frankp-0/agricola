import jax
import jax.numpy as jnp
from jaxtyping import Array, Arraylike
import numpy as np
from .data import GenoAncestryDataset
from ._utils import _stdize
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
    X: jnp.ndarray,
    Y: jnp.ndarray,
    alphas: jnp.ndarray,
    idx_train: jnp.ndarray | None = None,
    jit_enabled: bool = True,
):
    if jit_enabled:
        return jax.jit(_ridge_impl)(X, Y, alphas, idx_train)
    else:
        return _ridge_impl(X, Y, alphas, idx_train)


def _ridge_cv(
    X: jnp.ndarray, Y: jnp.ndarray, alphas: jnp.ndarray, k: int = 5, seed: int = 23891
) -> np.ndarray:
    n_samples, n_pheno = Y.shape
    n_alpha = alphas.shape[0]

    key = jax.random.PRNGKey(seed)
    idx = jax.random.permutation(key, n_samples)
    fold_size = n_samples // k
    folds = [idx[i * fold_size : (i + 1) * fold_size] for i in range(k)]

    cv_errors = jnp.zeros((n_pheno, n_alpha), dtype=jnp.float32)

    for fold in range(k):
        idx_test = folds[fold]
        idx_train = jnp.concatenate([folds[i] for i in range(k) if i != fold])
        fold_preds = _ridge(X, Y, alphas, idx_train)[idx_test]
        errors = jnp.mean(
            (Y[idx_test, :, None] - fold_preds) ** 2, axis=0
        )  # shape (P, len(alphas))
        cv_errors += errors

    alpha_best_idx = jnp.argmin(cv_errors, axis=1)
    alpha_best = alphas[alpha_best_idx]

    preds_all = _ridge(X, Y, alpha_best, jit_enabled=False)  # shape (N, P, 1)
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
    Y: jnp.ndarray,
    Q_covar: jnp.ndarray,
    block: np.ndarray,
    alphas: jnp.ndarray,
) -> np.ndarray:
    """Get ridge predictions for a block of variants

    Args:
        dataset: GenoAncestryDataset
        Y: (N, P) ndarray of outcomes
        Q_covar: (N, C) ndarray representing the orthogonal matrix Q in the QR
        decomposition of C covariates
        block: 1D ndarray with variant indices of block
        alphas: A 1D ndarray of ridge regression penalty values

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
    Y: jnp.ndarray,
    Q_covar: jnp.ndarray,
    B: int,
    alphas: jnp.ndarray,
    desc: str,
) -> dict[str, np.ndarray]:
    """Get ridge predictions for a dataset

    Args:
        dataset: GenoAncestryDataset
        Y: (N x P) ndarray of outcomes
        Q_covar: (N x C) ndarray representing the orthogonal matrix Q in the QR
        decomposition of C covariates
        B: An integer with the block size (max number of variants to read at once)
        alphas: A list of ridge regression penalty values
        desc: String describing the dataset, used for tracking progress

    Returns:
        A dict where keys are chromosomes and values are
        (N, P, len(alphas) * n_blocks) ndarrays with ridge predictions
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
    Y: jnp.ndarray | np.ndarray,
    covar: jnp.ndarray | np.ndarray | None = None,
    B: int = 2000,
    alphas0: jnp.ndarray | np.ndarray = jnp.array(
        [0.01, 0.25, 0.5, 0.75, 0.99], dtype=np.float32
    ),
    alphas1: jnp.ndarray | np.ndarray = jnp.array(
        [0.01, 0.25, 0.5, 0.75, 0.99], dtype=np.float32
    ),
) -> dict[str, np.ndarray]:
    """Run step 1 and return genomic predictions

    Args:
        datasets: list of GenoAncestryDataset
        Y: (N x P) ndarray of outcomes
        covar: (N x C) ndarray of covariates. If not provided, defaults to
        intercept-only covariate
        B: block size
        alphas0: ridge regression penalties for step 0
        alphas1: ridge regression penalties for step 1

    Returns:
        A dict where each key is a chromosome and each value is an (N x P)
        ndarray of genomic predictions
    """
    if covar is None:
        covar = jnp.ones((Y.shape[0], 1), dtype=jnp.float32)
    else:
        covar = jnp.hstack(
            [
                jnp.ones((Y.shape[0], 1), dtype=jnp.float32),
                jnp.asarray(covar, dtype=jnp.float32),
            ]
        )

    Q_covar, _ = jnp.linalg.qr(covar, mode="reduced")
    Y = jnp.asarray(Y, jnp.float32)
    Y_resid = _stdize(Y - (Q_covar @ (Q_covar.T @ Y)))

    alphas0 = jnp.asarray(alphas0)
    alphas1 = jnp.asarray(alphas1)

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
        for chrom, X in step0_predictions.items():
            preds = np.empty((Y_resid.shape[0], Y_resid.shape[1]), dtype=np.float32)
            for j in range(Y.shape[1]):
                preds[:, j] = _ridge_cv(X[:, j, :], Y_resid[:, j : j + 1], alphas1)[  # pyright: ignore
                    :, 0
                ]
            step1_predictions[chrom] = preds
            pbar.update(1)

    return step1_predictions
