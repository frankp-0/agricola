import jax
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
from ._utils import _stdize
from tqdm import tqdm
from .models import _ridge_masked


def _ridge_cv(Z, Y, train_mask, test_mask, h2_prior):
    """Get level 1 predictions for a single chromosome

    Args:
        Z: A (N, N_blocks, P) jax array of step 0 predictions (for a single chromosome)
        Y: A (N, P) jax array of (residualized, standardize) phenotypes
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D jax array of prior values for snp heritability

    Returns:
        Yhat: A (N, P) jax array of step 0 predictions
    """
    ## Assign dimensions
    n, p = Y.shape
    _, k = train_mask.shape
    a = len(h2_prior)

    ## Calculate penalties based on prior heritability
    alphas = Z.shape[2] * (1 - h2_prior) / h2_prior

    ## Would love to vmap this but it uses way too much memory
    Yhat_alphas = np.zeros(shape=(n, p, a), dtype=np.float32)
    for fold in range(k):
        for pheno in range(p):
            ridge_fold = _ridge_masked(
                Z[:, pheno, :],
                Y[:, pheno],
                train_mask[:, fold],
                test_mask[:, fold],
                alphas,
            )
            Yhat_alphas[:, pheno, :] += ridge_fold[:, 0, :]

    # Get best CV alpha per-phenotype
    cv_errors = np.mean(
        (np.asarray(Y)[:, :, None] - Yhat_alphas) ** 2, axis=0
    )  # (p, a)
    alpha_idx = np.argmin(cv_errors, axis=1)  # (p,)

    # Get best CV prediction
    Yhat = np.take_along_axis(Yhat_alphas, alpha_idx[None, :, None], axis=2).squeeze(
        axis=2
    )
    return Yhat


def step1_qt(
    Z: dict[str, np.ndarray],
    Y: Array,
    X: Array,
    train_mask: Array,
    test_mask: Array,
    h2_prior: Array,
):
    """Perform level 1 ridge regressions for quantitative phenotypes

    Args:
        Z: A dict where keys are chromosomes and values are (N, N_blocks, P) jax arrays of step 0 predictions
        Y: A (N, P) jax array of phenotypes
        X: A (N, C) jax array of covariates
        train_mask: A (N, K) jax array indicating training set status for each set k in 1, ..., K
        test_mask: A (N, K) jax array indicating test set status for each set k in 1, ..., K
        h2_prior: A 1D jax array of prior values for snp heritability

    Returns:
        step1_predictions: A dict where keys are chromosomes and values are (N, P) jax arrays of step 0 predictions
    """

    n, p = Y.shape

    ## Residualize and standardize phenotypes
    Q, _ = jnp.linalg.qr(X, mode="reduced")
    Y = _stdize(Y - (Q @ (Q.T @ Y)))

    ## Perform step 1 for each chromosome
    with tqdm(
        total=len(Z),
        desc="Getting step 1 predictions",
        unit="chromosomes",
    ) as pbar:
        step1_predictions = {}
        for chrom, Z_chrom in Z.items():
            Yhat_chrom = np.empty(shape=(n, p), dtype=np.float32)
            Yhat_chrom = _ridge_cv(Z_chrom, Y, train_mask, test_mask, h2_prior)
            step1_predictions[chrom] = Yhat_chrom
            pbar.update(1)

    return step1_predictions
