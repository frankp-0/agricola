import jax
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
from .data import GenoAncestryDataset
from ._utils import _stdize
from tqdm import tqdm
from .models import _ridge_masked
from typing import Optional


def _step0_block(
    dataset: GenoAncestryDataset,
    Y_res: Array,
    Q: Array,
    train_mask: Array,
    test_mask: Array,
    block: np.ndarray,
    h2_prior: Array,
):
    n_samples = Y_res.shape[0]
    n_variants = len(block)
    n_ancestries = len(dataset.ancestries)

    G = dataset.get_lanc_geno(block).reshape(n_samples, n_variants * n_ancestries)
    G_res = _stdize(G - (Q @ (Q.T @ G)))
    M = G_res.shape[1]
    alphas = M * (1 - h2_prior) / h2_prior

    ridge = jax.vmap(_ridge_masked, in_axes=(None, None, 1, 1, None))
    result = jnp.sum(ridge(G_res, Y_res, train_mask, test_mask, alphas), axis=0)
    return np.asarray(_stdize(result))


def _step0_dataset(
    dataset: GenoAncestryDataset,
    Y_res: Array,
    Q: Array,
    train_mask: Array,
    test_mask: Array,
    B: int,
    variants: Optional[list[str]],
    h2_prior: Array,
    desc: str,
):
    if variants is None:
        idx_variant = np.arange(dataset.pvar.get_variant_ct(), dtype=np.uint32)
    else:
        dataset_ids = [
            dataset.pvar.get_variant_id(i).decode("utf8")
            for i in np.arange(dataset.pvar.get_variant_ct(), dtype=np.uint32)
        ]
        varset = set(variants)
        idx_variant = np.array(
            [i for i, x in enumerate(dataset_ids) if x in varset], dtype=np.uint32
        )

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
            blocks = [
                idx_variant[idx_chrom[i : i + B]] for i in range(0, len(idx_chrom), B)
            ]

            preds = []
            for block in blocks:
                preds.append(
                    _step0_block(
                        dataset, Y_res, Q, train_mask, test_mask, block, h2_prior
                    )
                )
                pbar.update(1)

            predictions[chrom] = np.concatenate(preds, axis=2)

    return predictions


def _merge_step0(
    predictions: list[dict[str, np.ndarray]],
):
    chrom_keys = sorted({k for d in predictions for k in d})
    return {
        chrom: np.concatenate([d[chrom] for d in predictions if chrom in d], axis=2)
        for chrom in chrom_keys
    }


def step0(
    datasets: list[GenoAncestryDataset],
    Y: Array,
    X: Array,
    train_mask: Array,
    test_mask: Array,
    h2_prior: Array,
    B: int = 2000,
    variants: Optional[list[str]] = None,
):
    Q, _ = jnp.linalg.qr(X, mode="reduced")  # pyright: ignore
    Y_res = _stdize(Y - (Q @ (Q.T @ Y)))

    dataset_predictions = []
    for ds in datasets:
        pgen_path = ds.plink_prefix + ".pgen"
        desc = f"Getting step 0 predictions for file: {pgen_path}"
        preds = _step0_dataset(
            ds, Y_res, Q, train_mask, test_mask, B, variants, h2_prior, desc=desc
        )
        dataset_predictions.append(preds)

    step0_predictions = _merge_step0(dataset_predictions)

    return step0_predictions
