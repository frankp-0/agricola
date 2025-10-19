import jax
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
from ._utils import _stdize
from tqdm import tqdm
from .models import _ridge_masked


def _ridge_cv(Z, Y, train_mask, test_mask, alphas):
    # vmap across CV folds for a single phenotype
    # (k, n, 1, a) array of masked predictions
    ridge_pheno = jax.vmap(_ridge_masked, in_axes=(None, None, 1, 1, None))

    # vmap the above across phenotypes,
    # (k, n, p, 1, a) masked predictions
    ridge_across_phenos = jax.vmap(
        ridge_pheno, in_axes=(1, 1, None, None, None), out_axes=2
    )
    result_masked = ridge_across_phenos(Z, Y, train_mask, test_mask, alphas)

    # Sum over masked k-fold predictions: (n, p, a)
    result = jnp.sum(jnp.squeeze(result_masked, axis=3), axis=0)

    # Get best CV alpha per-phenotype
    cv_errors = jnp.mean(result**2, axis=0)  # (p, a)
    alpha_idx = jnp.argmin(cv_errors, axis=1)  # (p,)

    # Get best CV prediction
    result = jnp.take_along_axis(result, alpha_idx[None, :, None], axis=2).squeeze(
        axis=2
    )
    return result


def step1_qt(
    Z_dict: dict[str, np.ndarray],
    Y: Array,
    X: Array,
    train_mask: Array,
    test_mask: Array,
    alphas: Array,
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
            Yhat = _ridge_cv(Z, Y_res, train_mask, test_mask, alphas)
            Yhat_dict[chrom] = Yhat
            pbar.update(1)

    return Yhat_dict
