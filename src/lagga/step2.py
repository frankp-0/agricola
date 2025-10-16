import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import f as f_dist, t as t_dist
from typing import List
import pandas as pd
from .data import GenoAncestryDataset
from tqdm import tqdm
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from ._utils import _stdize


@jax.jit
def _step2_block_core(
    X: jnp.ndarray, Y: jnp.ndarray, Q_covar: jnp.ndarray, anc_ac: jnp.ndarray, eps=1e-8
) -> dict[str, jnp.ndarray]:
    n, _ = Y.shape

    # Adjust X
    QX = jnp.einsum("nc,nbk->cbk", Q_covar, X)
    X_adj = X - jnp.einsum("nc,cbk->nbk", Q_covar, QX)

    # Fit regression
    XtX = jnp.einsum("nbk,nbl->bkl", X_adj, X_adj)
    XtY = jnp.einsum("nbk,np->bkp", X_adj, Y)
    I_ = jnp.identity(XtX.shape[1], XtX.dtype)
    beta_hat = jnp.linalg.solve(XtX + eps * I_[None, :, :], XtY)
    Y_hat = jnp.einsum("nbk,bkp->nbp", X_adj, beta_hat)

    # Compute useful quantities
    df = jnp.sum(anc_ac != 0, axis=1)
    dfd = n - df[:, None]
    dfn = df[:, None]
    mse = ((Y[:, None, :] - Y_hat) ** 2).sum(axis=0) / dfd

    # Global test
    f_stat = (Y_hat**2).sum(axis=0) / dfn / mse
    f_stat = jnp.where(df[:, None] == 0, jnp.nan, f_stat)

    # Ancestry-specific tests
    XtX_diag = jnp.diagonal(XtX, axis1=1, axis2=2)[:, :, None]
    XtX_diag_safe = jnp.where(XtX_diag == 0, 1.0, XtX_diag)
    se = jnp.where(XtX_diag == 0, jnp.nan, jnp.sqrt(mse[:, None, :] / XtX_diag_safe))  # pyright: ignore
    t_stat = jnp.where(XtX_diag == 0, jnp.nan, beta_hat / se)  # pyright: ignore

    return {
        "beta_hat": beta_hat,
        "t_stat": t_stat,  # pyright: ignore
        "f_stat": f_stat,
        "dfn": dfn,
        "dfd": dfd,
        "se": se,
    }


def _step2_block(
    dataset: GenoAncestryDataset,
    Y: jnp.ndarray,
    Q_covar: jnp.ndarray,
    block: np.ndarray,
    min_anc_ac: int = 1,
):
    X = dataset.get_lanc_geno(block)
    anc_ac = X.sum(axis=0)

    anc_variant_mask = anc_ac.sum(axis=1) >= min_anc_ac

    res = _step2_block_core(X, Y, Q_covar, anc_ac)

    f_stat = np.array(res["f_stat"])
    t_stat = np.array(res["t_stat"])
    dfn = np.array(res["dfn"])
    dfd = np.array(res["dfd"])
    se = np.array(res["se"])
    beta_hat = np.array(res["beta_hat"])

    log10p_overall = f_dist.logsf(f_stat, dfn=dfn, dfd=dfd) / np.log(10)
    log10p_anc = (np.log(2) + t_dist.logsf(np.abs(t_stat), dfd[:, :, None])) / np.log(
        10
    )

    result_arr = np.concatenate(
        [
            beta_hat,
            f_stat[:, None, :],
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
        "f_overall",
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
    Q_covar: jnp.ndarray,
    out_prefix: str,
    phenotypes: list[str],
    desc: str,
    B: int = 2000,
):
    idx_variant = np.arange(dataset.pvar.get_variant_ct(), dtype=np.uint32)

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
            blocks = [idx_chrom[i : i + B] for i in range(0, len(idx_chrom), B)]

            for block in blocks:
                result_dfs = _step2_block(
                    dataset, jnp.array(Y_loco[chrom]), Q_covar, block
                )
                for i, df in enumerate(result_dfs):
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    if writers[i] is None:
                        writers[i] = pq.ParquetWriter(out_paths[i], table.schema)
                        schemas[i] = table.schema
                    else:
                        table = pa.Table.from_pandas(
                            df, schema=schemas[i], preserve_index=False
                        )
                    writers[i].write_table(table)
                pbar.update(1)


def step2(
    datasets: list[GenoAncestryDataset],
    Y: jnp.ndarray | np.ndarray,
    step1_predictions: dict[str, np.ndarray],
    out_prefixes: list[str],
    phenotypes: list[str] | None = None,
    covar: jnp.ndarray | np.ndarray | None = None,
    B: int = 1000,
):
    if covar is None:
        covar = jnp.ones((Y.shape[0], 1), dtype=jnp.float32)
    else:
        covar = jnp.hstack(
            [jnp.ones((Y.shape[0], 1), dtype=jnp.float32), jnp.asarray(covar)]
        )

    if phenotypes is None:
        phenotypes = [str(i) for i in range(Y.shape[1])]

    Q_covar, _ = jnp.linalg.qr(covar, mode="reduced")
    Y = jnp.asarray(Y, dtype=jnp.float32)
    Y_resid = _stdize(Y - (Q_covar @ (Q_covar.T @ Y)))

    step1_prs = np.sum(np.stack(list(step1_predictions.values())), axis=0)
    Y_loco = {
        k: np.array(Y_resid) - (step1_prs - v) for k, v in step1_predictions.items()
    }
    for i, ds in enumerate(datasets):
        pgen_path = ds.plink_prefix + ".pgen"
        desc = f"Getting step 2 results for file: {pgen_path}"
        _step2_dataset(ds, Y_loco, Q_covar, out_prefixes[i], phenotypes, desc, B)
