import jax
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
from ._utils import _stdize
from tqdm import tqdm
from .models import _ridge_masked


def _ridge_cv(Z, Y, train_mask, test_mask, h2_prior):
    n, p = Y.shape
    _, k = train_mask.shape
    a = len(h2_prior)
    preds = np.zeros(shape=(n, p, a), dtype=np.float32)
    alphas = Z.shape[2] * (1 - h2_prior) / h2_prior

    ## Would love to vmap this but it uses way too much memory
    for fold in range(k):
        for pheno in range(p):
            ridge_fold = _ridge_masked(
                Z[:, pheno, :],
                Y[:, pheno],
                train_mask[:, fold],
                test_mask[:, fold],
                alphas,
            )
            preds[:, pheno, :] += ridge_fold[:, 0, :]

    # Get best CV alpha per-phenotype
    cv_errors = np.mean((np.asarray(Y)[:, :, None] - preds) ** 2, axis=0)  # (p, a)
    alpha_idx = np.argmin(cv_errors, axis=1)  # (p,)

    # Get best CV prediction
    result = np.take_along_axis(preds, alpha_idx[None, :, None], axis=2).squeeze(axis=2)
    return result


def step1_qt(
    Z_dict: dict[str, np.ndarray],
    Y: Array,
    X: Array,
    train_mask: Array,
    test_mask: Array,
    h2_prior: Array,
):
    n, p = Y.shape
    Q, _ = jnp.linalg.qr(X, mode="reduced")
    Y_res = _stdize(Y - (Q @ (Q.T @ Y)))

    with tqdm(
        total=len(Z_dict),
        desc="Getting step 1 predictions",
        unit="chromosomes",
    ) as pbar:
        Yhat_dict = {}
        for chrom, Z in Z_dict.items():
            Yhat = np.empty(shape=(n, p), dtype=np.float32)
            Yhat = _ridge_cv(Z, Y_res, train_mask, test_mask, h2_prior)
            Yhat_dict[chrom] = Yhat
            pbar.update(1)

    return Yhat_dict
