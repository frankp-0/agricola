import jax
import jax.numpy as jnp
from jaxtyping import Array
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
def _step2_qt_core(
    G: Array,
    L: Array,
    Y: Array,
    Q: Array,
    eps: np.float32 = jnp.float32(1e-8),
) -> dict[str, Array]:
    """Estimate coefficients and Wald statistic for quantitative traits

    Args:
        G: A (N, B, len(ancestries)) jax array of anc-deconvoluted genotypes
        L: A (N, B, len(ancestries) - 1) jax array of local ancestry
        Y: A (N, P) jax array of LOCO phenotypes
        Q: (N, C) jax array. The orthogonal matrix Q in the QR decomposition of the covariates
        eps: A tolerance term (for non-intervertible design matrix)
    """

    ## Create predictor array X
    X = jnp.concatenate([G, L], axis=-1)

    ## Adjust X for covariates
    QX = jnp.einsum("nc,nbk->cbk", Q, X)
    X = X - jnp.einsum("nc,cbk->nbk", Q, QX)

    # Fit regression
    XtX = jnp.einsum("nbc,nbd->bcd", X, X)
    I_ = jnp.identity(XtX.shape[1], XtX.dtype)
    XtX_inv = jnp.linalg.inv(XtX + eps * I_[None, :, :])
    XtY = jnp.einsum("nbc,np->bcp", X, Y)
    beta_hat = XtX_inv @ XtY

    ## Subset to anc-specific effects (ignore local ancestry terms)
    k = G.shape[2]
    beta_G = beta_hat[:, :k, :]

    ## Calculate Wald statistic and coefficient standard errors
    sig_e = np.sum(Y**2, axis=0) / (Y.shape[0] - X.shape[2] - Q.shape[1])
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


def _step2_qt_block(
    dataset: GenoAncestryDataset,
    Y: Array,
    Q: Array,
    block: np.ndarray,
    min_anc_ac: int = 1,
):
    """Perform GWAS for quantitative traits for a single block of variants

    Args:
        dataset: GenoAncestryDataset
        Y: A (N, P) jax array of LOCO phenotypes
        Q: (N, C) jax array. The orthogonal matrix Q in the QR decomposition of the covariates
        block: A (B,) ndarray with indices of variants in the block
        min_anc_ac: The minimum count of ancestry-deconvoluted alleles to test

    Returns:
        result_dfs: A list of pandas tables (per-phenotype) with GWAS results
    """

    ## Query local ancestry and anc-deconvoluted genotypes
    G = dataset.get_lanc_geno(block)
    L = dataset.get_lanc_unphased(block)[:, :, 1:]

    ## Estimate coefficients and Wald statistic
    res = _step2_qt_core(G, L, Y, Q)
    wald = np.array(res["wald"])
    se = np.array(res["se"])
    beta_hat = np.array(res["beta_hat"])

    ## Calculate p-values
    log10p_overall = chi2.logsf(wald, G.shape[2]) / np.log(10)
    log10p_anc = chi2.logsf((beta_hat / se) ** 2, 1) / np.log(10)

    ## Create array with results
    result_arr = np.concatenate(
        [
            beta_hat,
            log10p_overall[:, None, :],
            se,
            log10p_anc,
        ],
        axis=1,
    )

    ## Get column names for results
    ancs = dataset.ancestries
    colnames: List[str] = [
        *["beta_" + anc for anc in ancs],
        "log10p_overall",
        *["se_" + anc for anc in ancs],
        *["log10p_" + anc for anc in ancs],
    ]

    ## Get info on variants in block
    block_info = dataset.get_info(block)  # all variants
    block_info["N"] = Y.shape[0]

    ## Filter out variants that fail min_anc_ac
    anc_variant_mask = G.sum(axis=0).sum(axis=1) >= min_anc_ac
    valid_idx = np.array(anc_variant_mask)
    block_info_filtered = block_info[valid_idx]
    result_arr_filtered = result_arr[valid_idx, :, :]

    ## Format results into list of dataframes
    p = Y.shape[1]
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


def _step2_qt_dataset(
    dataset: GenoAncestryDataset,
    Y_loco: dict[str, np.ndarray],
    Q: Array,
    out_prefix: str,
    phenotypes: list[str],
    desc: str,
    B: int = 2000,
    variants: Optional[list[str]] = None,
):
    """Perform GWAS for quantitative traits for a single dataset

    Args:
        dataset: GenoAncestryDataset
        Y_loco: A dict where each value is a (N, P) NumPy array with LOCO residuals from step 1
        Q: (N, C) jax array. The orthogonal matrix Q in the QR decomposition of the covariates
        out_prefix: Outputs will be written {output_prefix}_{phenotype}.parquet
        phenotypes: A list of phenotype names
        desc: A string describing the dataset, used for tracking progress
        B: The block size (max number of variants to read at once)
        variants: A list of variant IDs to include in the analysis. If not provided, all variants are used
    """

    ## Get variant indices
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

    ## Get chromosomes and number of blocks
    chromosomes = [
        dataset.pvar.get_variant_chrom(i).decode("utf8") for i in idx_variant
    ]
    chroms = list(set(chromosomes))  # unique chromosomes
    n_blocks = sum(
        (len([c for c in chromosomes if c == chrom]) + B - 1) // B for chrom in chroms
    )

    ## Initialize output
    out_paths = [Path(f"{out_prefix}_{pheno}.parquet") for pheno in phenotypes]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)

    writers = [None] * len(phenotypes)
    schemas = [None] * len(phenotypes)

    ## Perform step 2 for each chromosome and block
    with tqdm(total=n_blocks, desc=desc, unit="block") as pbar:
        for chrom in chroms:
            ## Get indices for this chromosome
            idx_chrom = np.array(
                [i for i, c in enumerate(chromosomes) if c == chrom], dtype=np.uint32
            )
            ## Split into blocks
            blocks = [
                idx_variant[idx_chrom[i : i + B]] for i in range(0, len(idx_chrom), B)
            ]

            for block in blocks:
                result_dfs = _step2_qt_block(
                    dataset, jnp.asarray(Y_loco[chrom]), Q, block
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
    """Run step 2 for quantitative traits

    Args:
        datasets: A list of GenoAncestryDataset objects
        Y: An (N, P) jax array of outcomes
        step1_predictions: A dict with chromosome-specific predictions from step 1.
        The values are (N, P) NumPy arrays
        out_prefixes: A list of prefixes for each dataset. Outputs will be written
        to {output_prefix}_{phenotype}.parquet
        phenotypes: A list of phenotype names
        covar: An (N, C) array of covariates. If not provided, defaults to intercept-only covariates
        B: The block size (max number of variants to read at once)
    """

    ## Residualize and standardize phenotypes
    Q, _ = jnp.linalg.qr(X, mode="reduced")
    Y = _stdize(Y - (Q @ (Q.T @ Y)))

    ## Get LOCO predictions
    step1_prs = np.sum(np.stack(list(step1_predictions.values())), axis=0)
    Y_loco = {k: np.asarray(Y) - (step1_prs - v) for k, v in step1_predictions.items()}

    ## Step 2
    for i, ds in enumerate(datasets):
        pgen_path = ds.plink_prefix + ".pgen"
        desc = f"Getting step 2 results for file: {pgen_path}"
        _step2_qt_dataset(ds, Y_loco, Q, out_prefixes[i], phenotypes, desc, B, variants)
