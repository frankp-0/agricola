# MIT License
# Copyright (c) 2025 Franklin Ockerman
# See LICENSE.txt file for full license text

import jax
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
from scipy.stats import chi2
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from typing import Optional
from jax.scipy.special import expit
from lanctools import LancData
from ._utils import stdize, assert_covar_full_rank
from .models import logistic

### ─────────────────────────────────────────────────────────────
### Helper Functions
### ─────────────────────────────────────────────────────────────


def validate_step2_inputs(
    datasets: list[LancData],
    Y: Array,
    X: Optional[Array],
    step1_predictions: dict[str, np.ndarray],
    out_prefixes: list[str],
    phenotypes: list[str],
    B: int = 1000,
    variants: Optional[list[str]] = None,
) -> Tuple[
    list[LancData],
    Array,
    Array,
    dict[str, np.ndarray],
    list[str],
    list[str],
    int,
    Optional[list[str]],
]:
    ## Type and structure checks for genotype/lanc data
    if not isinstance(datasets, (list, tuple)):
        raise TypeError(f"datasets must be a list of LancData, got {type(datasets)}")
    for i, ds in enumerate(datasets):
        if not isinstance(ds, LancData):
            raise TypeError(f"datasets[{i}] must be LancData, got {type(ds)}")

    ## Array conversions
    Y = jnp.asarray(Y)
    if X is not None:
        X = jnp.asarray(X)

    ## Check array shapes
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N, P), got shape {Y.shape}")
    N, P = Y.shape

    if X is not None:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N, C), got shape {X.shape}")
        if X.shape[0] != N:
            raise ValueError(
                f"X.shape[0] must match Y.shape[0], got {X.shape[0]} vs {N}"
            )

    ## X
    if X is None:
        X = jnp.ones((Y.shape[0], 1), dtype=np.float32)
    else:
        X = jnp.concatenate([jnp.ones((Y.shape[0], 1), dtype=np.float32), X], axis=1)
    X = stdize(X)
    assert_covar_full_rank(X)

    ## Check step1 predictions
    N_pred = None
    P_pred = None
    for chrom, pred_chrom in step1_predictions.items():
        if not isinstance(pred_chrom, (np.ndarray, jnp.ndarray)):
            raise TypeError(
                f"step1_predictions[{chrom}] must be a numpy/jax array, got {type(pred_chrom)}"
            )
        if pred_chrom.ndim != 2:
            raise ValueError(
                f"step1_predictions[{chrom}] must be 2D (N, P), got shape {pred_chrom.shape}"
            )

        n_chrom, p_chrom = pred_chrom.shape

        if N_pred is None:
            N_pred = n_chrom
            P_pred = p_chrom
        else:
            if n_chrom != N_pred:
                raise ValueError(
                    f"All step1_predictions arrays must have same N; got {N_pred} vs {n_chrom} in step1_predictions[{chrom}]"
                )
            if p_chrom != P_pred:
                raise ValueError(
                    f"All step1_predictions arrays must have same P; got {P_pred} vs {p_chrom} in step1_predictions[{chrom}]"
                )

    if N_pred != N:
        raise ValueError(f"step1_predictions arrays have N={N_pred} but Y has N={N}")

    if P_pred != P:
        raise ValueError(f"step1_predictions arrays have P={P_pred} but Y has P={P}")

    ## Check B and variants
    if not isinstance(B, int) or B <= 0:
        raise ValueError(f"B must be a positive integer, got {B}")

    if variants is not None:
        if not isinstance(variants, (list, tuple)) or not all(
            isinstance(v, str) for v in variants
        ):
            raise TypeError("variants must be a list of strings")

    if not len(out_prefixes) == len(phenotypes):
        raise ValueError(
            "out_prefixes and phenotypes must have same number of elements"
        )
    return (datasets, Y, X, step1_predictions, out_prefixes, phenotypes, B, variants)


### ─────────────────────────────────────────────────────────────
### Quantitative Traits
### ─────────────────────────────────────────────────────────────


@jax.jit
def _step2_qt_core(G: Array, L: Array, Y: Array, Q: Array):
    """Estimate coefficients and Wald statistic for quantitative traits

    Args:
        G: A (N, B, len(ancestries)) jax array of anc-deconvoluted genotypes
        L: A (N, B, len(ancestries) - 1) jax array of local ancestry
        Y: A (N, P) jax array of LOCO phenotypes
        Q: (N, C) jax array. The orthogonal matrix Q in the QR decomposition of the covariates
    """
    ## Make homogeneous anc
    H = jnp.sum(G, axis=2)

    ## Adjust local ancestry and genotypes
    QG = jnp.einsum("nc,nbk->cbk", Q, G)
    QL = jnp.einsum("nc,nbk->cbk", Q, L)
    QH = jnp.einsum("nc,nb->cb", Q, H)
    G = G - jnp.einsum("nc,cbk->nbk", Q, QG)
    L = L - jnp.einsum("nc,cbk->nbk", Q, QL)
    H = H - jnp.einsum("nc,cb->nb", Q, QH)

    ## Fit null model
    LtL = jnp.einsum("nbc,nbd->bcd", L, L)
    I_ = jnp.identity(LtL.shape[1], LtL.dtype)
    LtL_inv = jnp.linalg.inv(LtL + 1e-8 * I_[None, :, :])
    LtY = jnp.einsum("nbc,np->bcp", L, Y)
    beta_L = LtL_inv @ LtY
    r_L = Y[:, None, :] - jnp.einsum("nbc,bcp->nbp", L, beta_L)

    ## MSE under null
    sig2 = jnp.sum(Y**2, axis=0) / (Y.shape[0] - L.shape[2])

    ## Get residualized genotypes
    GtL = jnp.einsum("nbk,nbc->bkc", G, L)
    HtL = jnp.einsum("nb,nbc->bc", H, L)
    G_res = G - jnp.einsum("nbc,bkc->nbk", jnp.einsum("nbc,bcd->nbc", L, LtL_inv), GtL)
    H_res = H - jnp.einsum("nbc,bc->nb", jnp.einsum("nbc,bcd->nbc", L, LtL_inv), HtL)

    ## Get masks based on variance
    var_G = jnp.var(G_res, axis=0)
    var_H = jnp.var(H_res, axis=0)
    mask_G = var_G > 1e-8
    mask_H = var_H > 1e-8

    ## Score for anc-deconvoluted genotypes
    U = jnp.einsum("nbk,nbp->bkp", G, r_L)
    U = U * mask_G[:, :, None]  # apply mask
    I22_inv = jax.scipy.linalg.inv(
        jnp.einsum("nbk,nbl->bkl", G_res, G_res)
        + 1e-10 * jnp.identity(G.shape[2])[None, :, :]
    )
    I22_inv = I22_inv * jnp.einsum("bk,bl->bkl", mask_G, mask_G)  # apply mask
    chisq_het = (
        jnp.einsum("bkp,bkp->bp", jnp.einsum("bkp,bkk->bkp", U, I22_inv), U)
        / sig2[None, :]
    )

    beta_anc = U * jnp.diagonal(I22_inv, axis1=-1, axis2=-2)[:, :, None]

    # Score for genotypes
    UH = jnp.einsum("nb,nbp->bp", H, r_L)
    UH = UH * mask_H[:, None]  # apply mask
    I22_inv_H = 1 / (jnp.einsum("nb,nb->b", H_res, H_res) + 1e-10)
    I22_inv_H = I22_inv_H * mask_H  # apply mask
    chisq_hom = (UH**2) * I22_inv_H[:, None]

    K = jnp.einsum("nbk,nbl->bkl", G_res, G_res)
    eigvals = jnp.linalg.eigvalsh(K)
    df_het = jnp.sum(eigvals > 1e-8, axis=1)

    return chisq_het, chisq_hom, beta_anc, df_het


def _step2_qt_block(
    dataset: LancData,
    Y: Array,
    Q: Array,
    block: np.ndarray,
    min_ac: int = 1,
):
    """Perform GWAS for quantitative traits for a single block of variants

    Args:
        dataset: LancData
        Y: A (N, P) jax array of LOCO phenotypes
        Q: (N, C) jax array. The orthogonal matrix Q in the QR decomposition of the covariates
        block: A (B,) ndarray with indices of variants in the block
        min_ac: The minimum count of alleles to test

    Returns:
        result_dfs: A list of pandas tables (per-phenotype) with GWAS results
    """

    ## Query local ancestry and anc-deconvoluted genotypes
    G = jnp.asarray(dataset.get_lanc_geno(block), dtype=jnp.float32)
    L = jnp.asarray(dataset.get_lanc_dosage(block)[:, :, 1:], dtype=jnp.float32)
    chisq_het, chisq_hom, beta_anc, df_het = _step2_qt_core(G, L, Y, Q)

    log10p_het = chi2.logsf(chisq_het, df_het[:, None]) / np.log(10)
    log10p_hom = chi2.logsf(chisq_hom, 1) / np.log(10)

    ## Create array with results
    result_arr = np.concatenate(
        [
            log10p_het[:, None, :],
            log10p_hom[:, None, :],
            beta_anc,
        ],
        axis=1,
    )

    ## Get column names for results
    ancs = dataset.ancestries
    colnames: List[str] = ["log10p_het", "log10p_hom", *["beta_" + anc for anc in ancs]]

    ## Get info on variants in block
    block_info = dataset.get_info(block)  # all variants
    block_info["N"] = Y.shape[0]

    ## Filter out variants that fail min_ac
    anc_variant_mask = G.sum(axis=0).sum(axis=1) >= min_ac
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
    dataset: LancData,
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
        dataset: LancData
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
    datasets: list[LancData],
    Y: Array,
    X: Optional[Array],
    step1_predictions: dict[str, np.ndarray],
    out_prefixes: list[str],
    phenotypes: list[str],
    B: int = 1000,
    variants: Optional[list[str]] = None,
):
    """Run step 2 for quantitative traits

    Args:
        datasets: A list of LancData objects
        Y: An (N, P) jax array of outcomes
        X: A (N, C) jax array of covariates
        step1_predictions: A dict with chromosome-specific predictions from step 1. The values are (N, P) NumPy arrays
        out_prefixes: A list of prefixes for each dataset. Outputs will be written to {output_prefix}_{phenotype}.parquet
        phenotypes: A list of phenotype names
        B: The block size (max number of variants to read at once)
    """

    ## Validate inputs
    datasets, Y, X, step1_predictions, out_prefixes, phenotypes, B, variants = (
        validate_step2_inputs(
            datasets, Y, X, step1_predictions, out_prefixes, phenotypes, B, variants
        )
    )

    ## Residualize and standardize phenotypes
    X = jnp.concatenate([jnp.ones((Y.shape[0], 1), dtype=np.float32), X], axis=1)
    Q, _ = jnp.linalg.qr(X, mode="reduced")
    Y = stdize(Y - (Q @ (Q.T @ Y)))

    ## Get LOCO predictions
    step1_prs = np.sum(np.stack(list(step1_predictions.values())), axis=0)
    Y_loco = {k: np.asarray(Y) - (step1_prs - v) for k, v in step1_predictions.items()}

    ## Step 2
    for i, ds in enumerate(datasets):
        pgen_path = ds.plink_prefix + ".pgen"
        desc = f"Getting step 2 results for file: {pgen_path}"
        _step2_qt_dataset(ds, Y_loco, Q, out_prefixes[i], phenotypes, desc, B, variants)


### ─────────────────────────────────────────────────────────────
### Binary Traits
### ─────────────────────────────────────────────────────────────


@jax.jit
def _step2_bt_core(
    G: Array,
    L: Array,
    Y: Array,
    Q_w: Array,
    W_sqrt: Array,
    O: Array,
):
    """
    Args:
        G: A (N, B, len(ancestries)) jax array of anc-deconvoluted genotypes
        L: A (N, B, len(ancestries) - 1) jax array of local ancestry
        Y: A (N, P) jax array of phenotypes
        Q_w:
        W_sqrt:
        O: A (N, P) jax array of offsets (from covariate-only model)
    """

    ## Residualize G, H, L by covariates
    H = jnp.sum(G, axis=2)
    H = H[:, :, None] - jnp.einsum(
        "ncp,cbp->nbp",
        Q_w,
        jnp.einsum("ncp,nbp->cbp", Q_w, H[:, :, None] * W_sqrt[:, None, :]),
    )

    G = G[:, :, :, None] - jnp.einsum(
        "ncp,cbkp->nbkp",
        Q_w,
        jnp.einsum("ncp,nbkp->cbkp", Q_w, G[:, :, :, None] * W_sqrt[:, None, None, :]),
    )

    L = L[:, :, :, None] - jnp.einsum(
        "ncp,cbap->nbap",
        Q_w,
        jnp.einsum("ncp,nbap->cbap", Q_w, L[:, :, :, None] * W_sqrt[:, None, None, :]),
    )

    ## Fit L + covariate offset null model
    logistic_model = jax.vmap(
        jax.vmap(logistic, in_axes=(2, 1, 1, None)), in_axes=(1, None, None, None)
    )
    beta = logistic_model(L, Y, O, 10)
    eta = jnp.einsum("nbap,bpa->nbp", L, beta) + O[:, None, :]
    mu = expit(eta)
    R = Y[:, None, :] - mu
    W_L_sqrt = jnp.sqrt(mu * (1 - mu))

    ## Residualize G, H by L
    QL, _ = jnp.linalg.qr(
        jnp.moveaxis(L * W_L_sqrt[:, :, None, :], (0, 1, 2, 3), (2, 0, 3, 1)),
        mode="reduced",
    )
    QL = QL.transpose((2, 0, 3, 1))
    G_res = G * W_L_sqrt[:, :, None, :] - jnp.einsum(
        "nbap,bakp->nbkp",
        QL,
        jnp.einsum("nbap,nbkp->bakp", QL, G * W_L_sqrt[:, :, None, :]),
    )
    H_res = H * W_L_sqrt - jnp.einsum(
        "nbap,bap->nbp", QL, jnp.einsum("nbap,nbp->bap", QL, H * W_L_sqrt)
    )

    ## Get masks based on variance
    var_G = jnp.var(G_res, axis=0)
    var_H = jnp.var(H_res, axis=0)
    mask_G = var_G > 1e-8
    mask_H = var_H > 1e-8

    ## Score for anc-deconvoluted genotypes
    U = jnp.einsum("nbkp,nbp->bkp", G, R)
    U = U * mask_G  # apply mask
    I22_inv = jnp.linalg.inv(
        jnp.einsum("nbkp,nblp->bpkl", G_res, G_res)
        + 1e-8 * jnp.eye(G_res.shape[2])[None, None, :, :],
    )
    I22_inv = I22_inv * jnp.einsum("bkp,blp->bpkl", mask_G, mask_G)  # apply mask
    chisq_het = jnp.einsum("bkp,bpkl,blp->bp", U, I22_inv, U)

    UH = jnp.einsum("nbp,nbp->bp", H, R)
    UH = UH * mask_H  # apply mask
    I22_inv_H = 1 / (jnp.einsum("nbp,nbp->bp", H_res, H_res) + 1e-8)
    I22_inv_H = I22_inv_H * mask_H
    chisq_hom = (UH**2) * I22_inv_H

    beta_anc = U * jnp.diagonal(I22_inv, axis1=-1, axis2=-2).transpose((0, 2, 1))

    K = jnp.einsum("nbkp,nblp->bpkl", G, G)
    eigvals = jnp.linalg.eigvalsh(K)
    df_het = jnp.sum(eigvals > 1e-8, axis=2)
    return chisq_hom, chisq_het, beta_anc, df_het


def _step2_bt_block(
    dataset: LancData,
    Y: Array,
    Q_w: Array,
    W_sqrt: Array,
    O: Array,
    block: np.ndarray,
    min_anc_ac: int = 1,
):
    G = dataset.get_lanc_geno(block)
    L = dataset.get_lanc_dosage(block)[:, :, 1:]

    chisq_hom, chisq_het, beta_anc, df_het = _step2_bt_core(G, L, Y, Q_w, W_sqrt, O)

    log10p_het = chi2.logsf(chisq_het, df_het) / np.log(10)
    log10p_hom = chi2.logsf(chisq_hom, 1) / np.log(10)

    ## Create array with results
    result_arr = np.concatenate(
        [log10p_het[:, None, :], log10p_hom[:, None, :], beta_anc],
        axis=1,
    )

    ## Get column names for results
    ancs = dataset.ancestries
    colnames: List[str] = ["log10p_het", "log10p_hom", *["beta_" + anc for anc in ancs]]

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


def _step2_bt_dataset(
    dataset: LancData,
    Y: Array,
    O_loco: dict[str, np.ndarray],
    X: Array,
    out_prefix: str,
    phenotypes: list[str],
    desc: str,
    B: int = 2000,
    variants: Optional[list[str]] = None,
):
    """Perform GWAS for quantitative traits for a single dataset

    Args:
        dataset: LancData
        Y: A (N, P) jax array of phenotypes
        O_loco: A dict where each value is a (N, P) NumPy array with LOCO offsets from step 1
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

            ## QR decomposition of weighted covariates
            mu = expit(O_loco[chrom])
            W_sqrt = jnp.sqrt(mu * (1 - mu))
            Q_w, _ = jnp.linalg.qr(
                jnp.moveaxis(X[:, :, None] * W_sqrt[:, None, :], (0, 1, 2), (1, 2, 0)),
                mode="reduced",
            )
            Q_w = Q_w.transpose((1, 2, 0))

            for block in blocks:
                result_dfs = _step2_bt_block(
                    dataset, Y, Q_w, W_sqrt, jnp.asarray(O_loco[chrom]), block
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


def step2_bt(
    datasets: list[LancData],
    Y: Array,
    X: Optional[Array],
    step1_predictions: dict[str, np.ndarray],
    out_prefixes: list[str],
    phenotypes: list[str],
    B: int = 1000,
    variants: Optional[list[str]] = None,
):
    """Run step 2 for quantitative traits

    Args:
        datasets: A list of LancData objects
        Y: An (N, P) jax array of outcomes
        X: A (N, C) jax array of covariates
        step1_predictions: A dict with chromosome-specific linear predictions from step 1. The values are (N, P) NumPy arrays
        out_prefixes: A list of prefixes for each dataset. Outputs will be written to {output_prefix}_{phenotype}.parquet
        phenotypes: A list of phenotype names
        B: The block size (max number of variants to read at once)
    """
    ## Validate inputs
    datasets, Y, X, step1_predictions, out_prefixes, phenotypes, B, variants = (
        validate_step2_inputs(
            datasets, Y, X, step1_predictions, out_prefixes, phenotypes, B, variants
        )
    )

    covar_model = jax.vmap(logistic, in_axes=(None, 1, None), out_axes=1)
    offset_covar = X @ covar_model(X, Y, jnp.zeros(X.shape[0]))

    ## LOCO offsets
    chromosome_offsets = np.stack(
        [v - offset_covar for v in step1_predictions.values()], axis=1
    )
    O_loco = {}
    for i, chrom in enumerate(step1_predictions):
        X_leave = np.delete(chromosome_offsets, i, axis=1)
        beta_chrom = jax.vmap(logistic, in_axes=(2, 1, 1), out_axes=1)(
            X_leave, Y, offset_covar
        )
        eta_chrom = np.einsum("ncp,cp->np", X_leave, beta_chrom) + offset_covar
        O_loco[chrom] = eta_chrom

    ## Step 2
    for i, ds in enumerate(datasets):
        pgen_path = ds.plink_prefix + ".pgen"
        desc = f"Getting step 2 results for file: {pgen_path}"
        _step2_bt_dataset(
            ds, Y, O_loco, X, out_prefixes[i], phenotypes, desc, B, variants
        )
