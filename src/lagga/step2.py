import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
import numpy as np
from scipy.stats import chi2
from typing import List
import pandas as pd
from .data import GenoAncestryDataset
from tqdm import tqdm
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from ._utils import _stdize
from typing import Optional
from jax.scipy.linalg import solve


@jax.jit
def _step2_block_core(
    G: Array,
    L: Array,
    Y: Array,
    Q_covar: Array,
    eps=jnp.float32(1e-8),
) -> dict[str, Array]:
    k = G.shape[2]

    # Adjust X
    X = jnp.concatenate([G, L], axis=-1)
    QX = jnp.einsum("nc,nbk->cbk", Q_covar, X)
    X = X - jnp.einsum("nc,cbk->nbk", Q_covar, QX)

    # Fit regression
    XtX = jnp.einsum("nbc,nbd->bcd", X, X)
    I_ = jnp.identity(XtX.shape[1], XtX.dtype)
    XtX_inv = jnp.linalg.inv(XtX + eps * I_[None, :, :])
    XtY = jnp.einsum("nbc,np->bcp", X, Y)
    beta_hat = XtX_inv @ XtY
    beta_G = beta_hat[:, :k, :]
    sig_e = np.sum(Y**2, axis=0) / (Y.shape[0] - X.shape[2] - Q_covar.shape[1])
    W = jnp.einsum("bkp,bkp->bp", beta_G, solve(XtX_inv[:, :k, :k], beta_G)) / sig_e
    se_beta_G = jnp.sqrt(
        jnp.diagonal(XtX_inv[:, :k, :k], axis1=-2, axis2=-1)[:, :, None]
        * sig_e[None, None, :]
    )

    return {
        "beta_hat": beta_G,
        "se": se_beta_G,
        "wald": W,
    }


def _step2_block(
    dataset: GenoAncestryDataset,
    Y: Array,
    Q_covar: Array,
    block: np.ndarray,
    min_anc_ac: int = 1,
):
    """Perform step 2 for a block of variants"""
    G = dataset.get_lanc_geno(block)
    L = dataset.get_lanc_unphased(block)[:, :, 1:]
    anc_ac = G.sum(axis=0)

    anc_variant_mask = anc_ac.sum(axis=1) >= min_anc_ac

    res = _step2_block_core(G, L, Y, Q_covar)

    wald = np.array(res["wald"])
    se = np.array(res["se"])
    beta_hat = np.array(res["beta_hat"])

    log10p_overall = chi2.logsf(wald, G.shape[2]) / np.log(10)
    log10p_anc = chi2.logsf((beta_hat / se) ** 2, 1) / np.log(10)

    result_arr = np.concatenate(
        [
            beta_hat,
            log10p_overall[:, None, :],
            se,
            log10p_anc,
        ],
        axis=1,
    )

    # Variant info + DataFrames
    block_info = dataset.get_info(block)  # all variants
    block_info["N"] = Y.shape[0]

    ancs = dataset.ancestries
    colnames: List[str] = [
        *["beta_" + anc for anc in ancs],
        "log10p_overall",
        *["se_" + anc for anc in ancs],
        *["log10p_" + anc for anc in ancs],
    ]

    p = Y.shape[1]

    # Filter for reporting only variants that pass min_anc_ac
    valid_idx = np.array(anc_variant_mask)
    block_info_filtered = block_info[valid_idx]
    result_arr_filtered = result_arr[valid_idx, :, :]

    result_dfs = [
        pd.concat(
            [
                block_info_filtered,
                pd.DataFrame(data=result_arr_filtered[:, :, i], columns=colnames),  # pyright: ignore
            ],
            axis=1,
        )
        for i in range(p)
    ]

    return result_dfs


def _step2_dataset(
    dataset: GenoAncestryDataset,
    Y_loco: dict[str, np.ndarray],
    Q_covar: Array,
    out_prefix: str,
    phenotypes: list[str],
    desc: str,
    B: int = 2000,
    variants: Optional[list[str]] = None,
):
    """Perform GWAS for a single dataset

    Args:
        dataset: GenoAncestryDataset
        Y_loco: A dict where each value is a (N, P) NumPy array with LOCO residuals from step 1
        Q_covar: (N, C) jax array. The orthogonal matrix Q in the QR decomposition
        of the covariates
        out_prefix: Outputs will be written {output_prefix}_{phenotype}.parquet
        phenotypes: A list of phenotype names
        desc: A string describing the dataset, used for tracking progress
        B: The block size (max number of variants to read at once)
    """

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

    out_paths = [Path(f"{out_prefix}_{pheno}.parquet") for pheno in phenotypes]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)

    writers = [None] * len(phenotypes)
    schemas = [None] * len(phenotypes)

    with tqdm(total=n_blocks, desc=desc, unit="block") as pbar:
        for chrom in chroms:
            # Get indices for this chromosome
            idx_chrom = np.array(
                [i for i, c in enumerate(chromosomes) if c == chrom], dtype=np.uint32
            )
            # Split into blocks
            blocks = [
                idx_variant[idx_chrom[i : i + B]] for i in range(0, len(idx_chrom), B)
            ]

            for block in blocks:
                result_dfs = _step2_block(
                    dataset, jnp.asarray(Y_loco[chrom]), Q_covar, block
                )
                for i, df in enumerate(result_dfs):
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    if writers[i] is None:
                        writers[i] = pq.ParquetWriter(out_paths[i], table.schema)  # pyright: ignore
                        schemas[i] = table.schema
                    else:
                        table = pa.Table.from_pandas(
                            df, schema=schemas[i], preserve_index=False
                        )
                    writers[i].write_table(table)  # pyright: ignore
                pbar.update(1)


def step2_qt(
    datasets: list[GenoAncestryDataset],
    Y: Array,
    X: Array,
    step1_predictions: dict[str, np.ndarray],
    out_prefixes: list[str],
    phenotypes: list[str],
    B: int = 1000,
    variants: Optional[list[str]] = None,
):
    """Run step 2 for each dataset

    Args:
        datasets: A list of GenoAncestryDataset objects
        Y: An (N, P) array of outcomes
        step1_predictions: A dict with chromosome-specific predictions from step 1.
        The values are (N, P) NumPy arrays
        out_prefixes: A list of prefixes for each dataset. Outputs will be written
        to {output_prefix}_{phenotype}.parquet
        phenotypes: A list of phenotype names
        covar: An (N, C) array of covariates. If not provided, defaults to intercept-only covariates
        B: The block size (max number of variants to read at once)
    """
    Q, _ = jnp.linalg.qr(X, mode="reduced")
    Y = _stdize(Y - (Q @ (Q.T @ Y)))

    step1_prs = np.sum(np.stack(list(step1_predictions.values())), axis=0)
    Y_loco = {k: np.asarray(Y) - (step1_prs - v) for k, v in step1_predictions.items()}
    for i, ds in enumerate(datasets):
        pgen_path = ds.plink_prefix + ".pgen"
        desc = f"Getting step 2 results for file: {pgen_path}"
        _step2_dataset(ds, Y_loco, Q, out_prefixes[i], phenotypes, desc, B, variants)
