# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""agricola step 2 tests.

This module uses whole-genome predictions from steps 0/1 to adjust traits and
perform single variant association tests. The entry-point is the `step2` function.
"""

import logging
import shutil
import time
from datetime import timedelta
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow as pa
from jax import jit, vmap
from jax.scipy.linalg import qr
from jax.scipy.special import expit
from jaxtyping import Array, ArrayLike
from lanctools import LancData
from scipy.stats import chi2
from tqdm import tqdm

from ..io.genotypes import get_geno_lanc_deconv
from ..io.variants import get_variant_indices, group_variant_indices_by_chromosome
from ..models.logistic import logistic_ridge_with_convergence
from ..numerical.linear_algebra import stdize
from ..statistics.binary import (
    bt_score_lanc,
    bt_score_lanc_no_diff,
    bt_score_nolanc,
    bt_score_nolanc_no_diff,
    bt_wald_lanc,
    bt_wald_nolanc,
)
from ..statistics.quantitative import (
    qt_score_lanc,
    qt_score_lanc_impute,
    qt_score_lanc_impute_no_diff,
    qt_score_lanc_no_diff,
    qt_score_nolanc,
    qt_score_nolanc_impute,
    qt_score_nolanc_impute_no_diff,
    qt_score_nolanc_no_diff,
    qt_wald_lanc,
    qt_wald_lanc_impute,
    qt_wald_nolanc,
    qt_wald_nolanc_impute,
)
from ..types import TestType, TraitType
from ..validation.inputs import validate_step2_inputs
from .writer import ParquetRotatingWriter

logger = logging.getLogger(__name__)

_QT_FUNCTIONS = {
    (TestType.SCORE, True, False): qt_score_lanc,
    (TestType.SCORE, True, True): qt_score_lanc_impute,
    (TestType.WALD, True, False): qt_wald_lanc,
    (TestType.WALD, True, True): qt_wald_lanc_impute,
    (TestType.SCORE, False, False): qt_score_nolanc,
    (TestType.SCORE, False, True): qt_score_nolanc_impute,
    (TestType.WALD, False, False): qt_wald_nolanc,
    (TestType.WALD, False, True): qt_wald_nolanc_impute,
}

_BT_FUNCTIONS = {
    (TestType.SCORE, True): bt_score_lanc,
    (TestType.WALD, True): bt_wald_lanc,
    (TestType.SCORE, False): bt_score_nolanc,
    (TestType.WALD, False): bt_wald_nolanc,
}


def _selected_diff_args(
    G: Array,
    L: Array,
    Y: Array,
    Q: Array,
    N_eff: Array,
    offset: Array | None,
    M: Array | None,
    allele_G_mask: Array,
    allele_H_mask: Array,
    variant_idx: np.ndarray,
    phenotype_idx: int,
    trait_type: TraitType,
    adjust_lanc: bool,
    impute: bool,
) -> tuple[Array, ...]:
    """Build single-phenotype inputs for a selected variant subset."""
    phenotype = slice(phenotype_idx, phenotype_idx + 1)
    if trait_type == TraitType.QT:
        if impute:
            if adjust_lanc:
                return (
                    G[:, variant_idx, :],
                    L[:, variant_idx, :],
                    Y[:, phenotype],
                    Q,
                    N_eff,
                    allele_G_mask[variant_idx],
                    allele_H_mask[variant_idx],
                )
            return (
                G[:, variant_idx, :],
                Y[:, phenotype],
                Q,
                N_eff,
                allele_G_mask[variant_idx],
                allele_H_mask[variant_idx],
            )
        if adjust_lanc:
            return (
                G[:, variant_idx, :, phenotype],
                L[:, variant_idx, :, phenotype],
                Y[:, phenotype],
                Q[:, :, phenotype],
                N_eff[phenotype],
                allele_G_mask[variant_idx, :, phenotype],
                allele_H_mask[variant_idx, phenotype],
            )
        return (
            G[:, variant_idx, :, phenotype],
            Y[:, phenotype],
            Q[:, :, phenotype],
            N_eff[phenotype],
            allele_G_mask[variant_idx, :, phenotype],
            allele_H_mask[variant_idx, phenotype],
        )
    if adjust_lanc:
        return (
            G[:, variant_idx, :, phenotype],
            L[:, variant_idx, :, phenotype],
            Y[:, phenotype],
            Q[:, :, phenotype],
            offset[:, phenotype],
            M[:, phenotype],
            allele_G_mask[variant_idx, :, phenotype],
            allele_H_mask[variant_idx, phenotype],
        )
    return (
        G[:, variant_idx, :, phenotype],
        Y[:, phenotype],
        Q[:, :, phenotype],
        offset[:, phenotype],
        M[:, phenotype],
        allele_G_mask[variant_idx, :, phenotype],
        allele_H_mask[variant_idx, phenotype],
    )


### ─────────────────────────────────────────────────────────────
### Helpers
### ─────────────────────────────────────────────────────────────


@jit
def _prep_block(G, L, M, min_ac_g, min_ac_h):
    N_eff = jnp.sum(M, axis=0)
    GM = G[:, :, :, None] * M[:, None, None, :]
    LM = L[:, :, :, None] * M[:, None, None, :]
    ac = GM.sum(axis=0)
    lac = LM.sum(axis=0)
    af_lanc = ac / lac
    prop_lanc = lac / N_eff[None, None, :] / 2
    L = L[:, :, 1:]
    LM = LM[:, :, 1:, :]
    G = G[:, :, :, None] - GM.sum(axis=0) / N_eff
    L = L[:, :, :, None] - LM.sum(axis=0) / N_eff
    allele_G_mask = ac >= min_ac_g
    allele_H_mask = ac.sum(axis=1) >= min_ac_h
    return G, L, M, N_eff, af_lanc, prop_lanc, allele_G_mask, allele_H_mask


### ─────────────────────────────────────────────────────────────
### Orchestration
### ─────────────────────────────────────────────────────────────


def _step2_block(
    dataset: LancData,
    Y: Array,
    M: Array,
    Q: Array,
    phenotypes: list[str],
    trait_type: TraitType,
    test_type: TestType,
    block: np.ndarray,
    idx_sample: Array | None,
    min_ac_g: int,
    min_ac_h: int,
    extra_args: dict,
    adjust_lanc: bool,
    impute: bool,
    p_het_threshold: float,
) -> pa.Table:
    """Run step 2 for a single block of variants

    Args:
        dataset: A LancData object
        Y: A (N, P) jax array of outcomes
        M: A (N, P) mask for missing values in the phenotypes/covariates
        Q: A (N, C) jax array. The orthogonal matrix Q in the QR decomposition of
            the covariates. For trait_type="bt", this is weighted by estimated
            variance in the covariate-only model.
        block: A (B,) jax array of variant indices
        idx_sample: An optional numpy array with ordered indices of samples (in
            the psam file) to retain
        min_ac_g: minimum allele count for ancestry-specific genotype columns
        min_ac_h: minimum allele count for the homogeneous genotype column
        extra_args: A dict containing extra arguments needed for trait_type="bt"
        adjust_lanc: A boolean indicating whether to adjust tests for local ancestry
        impute: Whether to impute the phenotype. Much faster, but only available for qt traits
    """
    G, L = get_geno_lanc_deconv(dataset, block)
    if idx_sample is not None:
        G = G[idx_sample]
        L = L[idx_sample]

    if impute:
        M = M[:, 0][:, None]
    G, L, M, N_eff, af_lanc, prop_lanc, allele_G_mask, allele_H_mask = _prep_block(
        G, L, M, min_ac_g, min_ac_h
    )

    _, B, K, _ = G.shape
    _, P = Y.shape
    N_eff = jnp.broadcast_to(N_eff, (Y.shape[1]))
    valid_idx = np.broadcast_to(np.asarray(allele_G_mask.any(axis=1) | allele_H_mask), (B, P))

    if trait_type == TraitType.QT:
        if impute:
            G_qt = G[:, :, :, 0]
            L_qt = L[:, :, :, 0]
            Q_qt = Q[:, :, 0]
            N_eff_qt = N_eff[0]
            allele_G_mask_qt = allele_G_mask[:, :, 0]
            allele_H_mask_qt = allele_H_mask[:, 0]
        else:
            G_qt, L_qt, Q_qt, N_eff_qt = G, L, Q, N_eff
            allele_G_mask_qt, allele_H_mask_qt = allele_G_mask, allele_H_mask

        if adjust_lanc:
            test_args = (
                G_qt,
                L_qt,
                Y,
                Q_qt,
                N_eff_qt,
                allele_G_mask_qt,
                allele_H_mask_qt,
            )
        else:
            test_args = (G_qt, Y, Q_qt, N_eff_qt, allele_G_mask_qt, allele_H_mask_qt)
        test_func = _QT_FUNCTIONS[(test_type, adjust_lanc, impute)]
    else:
        if adjust_lanc:
            test_args = (G, L, Y, Q, extra_args["offset"], M, allele_G_mask, allele_H_mask)
        else:
            test_args = (G, Y, Q, extra_args["offset"], M, allele_G_mask, allele_H_mask)
        test_func = _BT_FUNCTIONS[(test_type, adjust_lanc)]

    selective_score_diff = test_type == TestType.SCORE and p_het_threshold < 1
    if selective_score_diff:
        if trait_type == TraitType.QT:
            test_func = {
                (True, False): qt_score_lanc_no_diff,
                (True, True): qt_score_lanc_impute_no_diff,
                (False, False): qt_score_nolanc_no_diff,
                (False, True): qt_score_nolanc_impute_no_diff,
            }[(adjust_lanc, impute)]
        else:
            test_func = {True: bt_score_lanc_no_diff, False: bt_score_nolanc_no_diff}[adjust_lanc]

    log10p_het_vs_hom: np.ndarray | None = None
    test_converged = None

    if test_type == TestType.WALD:
        if trait_type == TraitType.QT:
            (
                chisq_hom,
                beta_hom,
                chisq_het,
                beta_het,
                df_het,
                chisq_anc,
                chisq_diff,
                df_diff,
            ) = test_func(*test_args)
        else:
            (
                chisq_hom,
                beta_hom,
                chisq_het,
                beta_het,
                df_het,
                chisq_anc,
                chisq_diff,
                df_diff,
                test_converged,
            ) = test_func(*test_args)
        chisq_diff = jnp.reshape(chisq_diff, (B, P))
        if df_diff.ndim == 1:
            df_diff = df_diff[:, None]
        log10p_het_vs_hom = chi2.logsf(chisq_diff, df_diff) / np.log(10)
    else:
        if trait_type == TraitType.QT:
            (
                chisq_hom,
                beta_hom,
                chisq_het,
                beta_het,
                df_het,
                chisq_anc,
                chisq_diff,
                df_diff,
            ) = test_func(*test_args)
        else:
            (
                chisq_hom,
                beta_hom,
                chisq_het,
                beta_het,
                df_het,
                chisq_anc,
                chisq_diff,
                df_diff,
                test_converged,
            ) = test_func(*test_args)
        chisq_diff = jnp.reshape(chisq_diff, (B, P))
        if df_diff.ndim == 1:
            df_diff = df_diff[:, None]
        log10p_het_vs_hom = chi2.logsf(chisq_diff, df_diff) / np.log(10)
    if trait_type == TraitType.QT:
        test_converged = None

    chisq_hom = jnp.reshape(chisq_hom, (B, P))
    beta_hom = jnp.reshape(beta_hom, (B, P))
    beta_het = jnp.reshape(beta_het, (B, K, P))
    chisq_anc = jnp.reshape(chisq_anc, (B, K, P))
    if df_het.ndim < 2:
        df_het = jnp.broadcast_to(df_het[:, None], (B, P))
    df_het = jnp.reshape(df_het, (B, P))
    chisq_het = jnp.reshape(chisq_het, (B, P))

    log10p_anc = chi2.logsf(chisq_anc, 1) / np.log(10)
    log10p_het = chi2.logsf(chisq_het, df_het) / np.log(10)
    log10p_hom = chi2.logsf(chisq_hom, 1) / np.log(10)
    if selective_score_diff:
        selected = (
            np.zeros((B, P), dtype=bool)
            if p_het_threshold == 0
            else np.asarray(log10p_het <= np.log10(p_het_threshold))
        )
        if trait_type == TraitType.BT and adjust_lanc:
            selected &= np.asarray(test_converged, dtype=bool)
        log10p_het_vs_hom = np.full((B, P), np.nan)
        if trait_type == TraitType.BT:
            convergence = np.asarray(test_converged)
            test_converged = (
                np.full((B, P), np.nan)
                if convergence.ndim == 0
                else np.array(convergence, copy=True)
            )
        diff_func = (
            _QT_FUNCTIONS[(TestType.SCORE, adjust_lanc, impute)]
            if trait_type == TraitType.QT
            else _BT_FUNCTIONS[(TestType.SCORE, adjust_lanc)]
        )
        for phenotype_idx in range(P):
            variant_idx = np.flatnonzero(selected[:, phenotype_idx])
            if not len(variant_idx):
                continue
            if trait_type == TraitType.QT:
                diff_G, diff_L, diff_Q, diff_N_eff = G_qt, L_qt, Q_qt, N_eff_qt
                diff_offset, diff_M = None, None
                diff_allele_G_mask, diff_allele_H_mask = allele_G_mask_qt, allele_H_mask_qt
            else:
                diff_G, diff_L, diff_Q = G, L, Q
                diff_N_eff = N_eff
                diff_offset, diff_M = extra_args["offset"], M
                diff_allele_G_mask, diff_allele_H_mask = allele_G_mask, allele_H_mask
            diff_args = _selected_diff_args(
                diff_G,
                diff_L,
                Y,
                diff_Q,
                diff_N_eff,
                diff_offset,
                diff_M,
                diff_allele_G_mask,
                diff_allele_H_mask,
                variant_idx,
                phenotype_idx,
                trait_type,
                adjust_lanc,
                impute,
            )
            diff_result = diff_func(*diff_args)
            diff_chisq = np.asarray(diff_result[6]).reshape(-1)
            diff_df = np.asarray(diff_result[7]).reshape(-1)
            log10p_het_vs_hom[variant_idx, phenotype_idx] = chi2.logsf(
                diff_chisq, diff_df
            ) / np.log(10)
            if trait_type == TraitType.BT:
                test_converged[variant_idx, phenotype_idx] = np.asarray(diff_result[8]).reshape(-1)
    if p_het_threshold < 1 and not selective_score_diff:
        selected = (
            jnp.zeros((B, P), dtype=bool)
            if p_het_threshold == 0
            else log10p_het <= np.log10(p_het_threshold)
        )
        log10p_het_vs_hom = jnp.where(selected, log10p_het_vs_hom, jnp.nan)
    if test_converged is not None and np.issubdtype(np.asarray(test_converged).dtype, np.number):
        if np.isnan(np.asarray(test_converged)).all():
            test_converged = None

    p_het = 10**log10p_het
    p_hom = 10**log10p_hom

    T_cct = 0.5 * np.tan(np.pi * (0.5 - p_het)) + 0.5 * np.tan(np.pi * (0.5 - p_hom))

    p_cct = 0.5 - np.arctan(T_cct) / np.pi
    log10p_cct = np.log10(p_cct)

    ## Create array with results
    result_components = [
        np.broadcast_to(N_eff, (B, 1, P)),
        np.broadcast_to(af_lanc, (B, K, P)),
        np.broadcast_to(prop_lanc, (B, K, P)),
        beta_het,
        beta_hom[:, None, :],
        log10p_het[:, None, :],
        log10p_hom[:, None, :],
        log10p_cct[:, None, :],
        log10p_anc,
    ]

    ## Get column names for results
    ancs = dataset.ancestries
    colnames: list[str] = [
        "N",
        *["AF_" + anc for anc in ancs],
        *["LA_PROP_" + anc for anc in ancs],
        *["BETA_" + anc for anc in ancs],
        "BETA_HOM",
        "LOG10P_HET",
        "LOG10P_HOM",
        "LOG10P_CCT",
        *["LOG10P_" + anc for anc in ancs],
    ]

    if log10p_het_vs_hom is not None:
        result_components.append(log10p_het_vs_hom[:, None, :])
        result_components.append(log10p_het_vs_hom[:, None, :])
        colnames.extend(["LOG10P_HET_VS_HOM", "LOG10P_LRT"])

    result_arr = np.concatenate(result_components, axis=1)
    converged = None
    if test_converged is not None:
        converged = np.asarray(test_converged)

    ## Get info on variants in block
    block_info = dataset.get_info(block)  # all variants

    ## Create single DataFrame
    p = Y.shape[1]
    tables = []

    for i in range(p):
        idx = valid_idx[:, i]

        block_table = pa.Table.from_pandas(
            block_info[idx].reset_index(drop=True),
            preserve_index=False,
        )

        columns = {name: block_table[name] for name in block_table.column_names}

        for j, name in enumerate(colnames):
            columns[name] = pa.array(result_arr[idx, j, i])
        if converged is not None:
            columns["CONVERGED"] = pa.array(converged[idx, i])

        columns["phenotype"] = pa.array([phenotypes[i]] * idx.sum())

        tables.append(pa.table(columns))

    result_table = pa.concat_tables(tables)

    return result_table


def _step2_dataset(
    dataset: LancData,
    writer: ParquetRotatingWriter,
    Y: Array,
    M: Array,
    step1_predictions: dict[str, np.ndarray] | None,
    X: Array,
    idx_sample: Array | None,
    phenotypes: list[str],
    trait_type: TraitType,
    test_type: TestType,
    chrom: str | None,
    B: int = 500,
    min_ac_g: int = 1,
    min_ac_h: int = 1,
    variants: list[str] | None = None,
    adjust_lanc: bool = True,
    impute: bool = False,
    p_het_threshold: float = 1.0,
) -> None:
    """Run step 2 for a single dataset

    Args:
        dataset: A LancData object
        writer: A Parquet writer
        Y: A (N, P) jax array of outcomes
        M: A (N, P) mask for missing values in the phenotypes/covariates
        step1_predictions: A dict of (N,P) pandas DataFrames containing LOCO predictions
        X: A (N, C) jax array of covariates
        idx_sample: An optional numpy array with ordered indices of samples (in
            the psam file) to retain
        phenotypes: A list of phenotype names
        trait_type: either qt or bt
        test_type: either score or wald
        B: The block size (max number of variants to read at once)
        min_ac_g: minimum allele count for ancestry-specific genotype columns
        min_ac_h: minimum allele count for the homogeneous genotype column
        variants: An optional list of variant IDs to retain
        adjust_lanc: A boolean indicating whether to adjust tests for local ancestry
        impute: Whether to impute the phenotype. Much faster, but only available for qt traits
    """
    if not 0 <= p_het_threshold <= 1:
        raise ValueError("p_het_threshold must be in [0, 1].")

    idx_variant = get_variant_indices(dataset, variants)

    variants_by_chromosome = group_variant_indices_by_chromosome(dataset, idx_variant)
    if chrom is not None:
        chroms = [chrom] if chrom in variants_by_chromosome else []
    else:
        chroms = list(variants_by_chromosome)

    n_blocks = sum((len(variants_by_chromosome[c]) + B - 1) // B for c in chroms)

    ## Perform step 2 for each chromosome and block
    with tqdm(total=n_blocks, unit="block") as pbar:
        for chrom in chroms:
            if step1_predictions is not None:
                if chrom in step1_predictions.keys():
                    step1_pred_chr = step1_predictions[chrom]
                else:
                    step1_pred_chr = step1_predictions["all"]
            else:
                step1_pred_chr = np.zeros(Y.shape)

            ## Get indices for this chromosome
            idx_chrom = variants_by_chromosome[chrom]
            ## Split into blocks
            blocks = [idx_chrom[i : i + B] for i in range(0, len(idx_chrom), B)]

            extra_args = {}

            Yc = Y  # chromosome-specific Y
            if trait_type == TraitType.QT:
                Yc = Yc - step1_pred_chr
                Yc = Yc - jnp.sum(Yc * M, axis=0) / jnp.sum(M, axis=0)
            else:
                beta_offset, _ = vmap(
                    lambda X, y, offset, train_mask, alpha: logistic_ridge_with_convergence(
                        X, y, offset, train_mask, alpha, max_iter=50, tol=1e-5
                    ),
                    in_axes=(None, 1, 1, 1, None),
                )(X, Y, jnp.asarray(step1_pred_chr), M, 0)
                offset = X @ beta_offset.T + step1_pred_chr
                mu = expit(offset)
                W_sqrt = jnp.sqrt(mu * (1 - mu))
                extra_args["W_sqrt"] = W_sqrt
                extra_args["offset"] = offset

            Yc = Yc * M
            ## Adjust covariates for per-phenotype missingness
            Xm = X[:, :, None] - jnp.sum(X[:, :, None] * M[:, None, :], axis=0) / jnp.sum(M, axis=0)
            Xm = Xm * M[:, None, :]
            Q, R, _ = qr(Xm.transpose(2, 0, 1), mode="economic", pivoting=True)
            r_diag = jnp.abs(jnp.diagonal(R, axis1=1, axis2=2))
            rank_tol = jnp.finfo(R.dtype).eps * max(Xm.shape[1], Xm.shape[2])
            rank_mask = r_diag > rank_tol * jnp.max(r_diag, axis=1, keepdims=True)
            Q = Q * rank_mask[:, None, :]
            Q = Q.transpose(1, 2, 0)

            for block in blocks:
                result_table = _step2_block(
                    dataset,
                    Yc,
                    M,
                    Q,
                    phenotypes,
                    trait_type,
                    test_type,
                    block,
                    idx_sample,
                    min_ac_g,
                    min_ac_h,
                    extra_args,
                    adjust_lanc,
                    impute,
                    p_het_threshold,
                )

                writer.write(result_table)
                pbar.update(1)


def step2(
    datasets: list[LancData],
    Y: ArrayLike,
    X: ArrayLike | None,
    step1_predictions: dict[str, pd.DataFrame] | None,
    outdir: str | Path,
    phenotypes: list[str],
    trait_type: str | TraitType = TraitType.QT,
    test_type: str | TestType = TestType.SCORE,
    chrom: str | None = None,
    B: int = 1000,
    min_ac: int = 1,
    idx_sample: ArrayLike | None = None,
    variants: list[str] | None = None,
    adjust_lanc: bool = True,
    impute: bool = False,
    overwrite: bool = True,
    partition_phenotype: bool = True,
    max_rows: int | None = None,
    p_het_threshold: float = 1.0,
    min_ac_g: int | None = None,
    min_ac_h: int | None = None,
) -> None:
    """Perform agricola step 2

    Args:
        datasets: A list of LancData objects (either single object or one
            per-chromosome)
        Y: A (N, P) jax array of outcomes
        X: A (N, C) jax array of covariates
        step1_predictions: An optional dict with LOCO linear predictions from step 1.
            The values are (N, P) NumPy arrays
        outdir: Outputs will be written to {output_prefix}_{phenotype}.parquet
        phenotypes: A list of phenotype names
        trait_type: either "qt" or "bt"
        test_type: Either "score" or "wald"
        B: The block size (max number of variants to read at once)
        min_ac: legacy minimum allele count used for both models when the
            model-specific thresholds are not supplied
        min_ac_g: minimum allele count for ancestry-specific genotype columns.
            Defaults to min_ac.
        min_ac_h: minimum allele count for the homogeneous genotype column.
            Defaults to min_ac.
        idx_sample: An optional numpy array with ordered indices of samples (in
            the psam file) to retain
        variants: An optional list of variant IDs to retain
        adjust_lanc: A boolean indicating whether to adjust tests for local ancestry
        impute: Whether to impute the phenotype. Much faster, but only available
            for qt traits. If all phenotypes are non-missing, this is ignored.
        p_het_threshold: Only report the heterogeneous-versus-homogeneous
            difference test for pairs with P_HET at or below this threshold.
            Set to zero to never report the test.
        overwrite: Whether to overwrite the outdir if it already exists
        partition_phenotype: Whether to partition output parquet files by phenotype
        max_rows: Max number of rows/variants per phenotype to keep in memory
            before writing an output file. Defaults to 5000000 / len(phenotypes)
    """
    if not 0 <= p_het_threshold <= 1:
        raise ValueError("p_het_threshold must be in [0, 1].")
    if min_ac_g is None:
        min_ac_g = min_ac
    if min_ac_h is None:
        min_ac_h = min_ac
    if min_ac_g < 0 or min_ac_h < 0:
        raise ValueError("min_ac_g and min_ac_h must be non-negative.")

    ## Create writer
    outdir_path = Path(outdir)
    if overwrite and outdir_path.exists():
        shutil.rmtree(outdir_path)

    if max_rows:
        max_rows_total = max_rows * len(phenotypes)
    else:
        max_rows_total = 5000000

    with ParquetRotatingWriter(outdir_path, partition_phenotype, max_rows_total) as writer:
        if impute:
            M = jnp.ones(shape=jnp.asarray(Y).shape)
        else:
            M = (~jnp.isnan(jnp.asarray(Y))).astype(float)

        Y, X, step1_predictions_np, idx_sample, test_type_enum, trait_type_enum = (
            validate_step2_inputs(
                datasets,
                Y,
                X,
                phenotypes,
                step1_predictions,
                B,
                idx_sample,
                variants,
                test_type,
                trait_type,
            )
        )

        ## Adjust phenotype for covariates to match step 1
        if trait_type_enum == TraitType.QT:
            Q, _ = jnp.linalg.qr(X, mode="reduced")
            Y = stdize(Y - (Q @ (Q.T @ Y)))
            if (M == 1).all():
                impute = True
        elif impute:
            raise ValueError("impute must be False for binary traits")

        time_total_start = time.perf_counter()
        for dataset in datasets:
            pgen_path = dataset.plink_prefix + ".pgen"
            logger.info("Testing associations for file: %s", pgen_path)

            time_ds_start = time.perf_counter()
            _step2_dataset(
                dataset,
                writer,
                Y,
                M,
                step1_predictions_np,
                X,
                idx_sample,
                phenotypes,
                trait_type_enum,
                test_type_enum,
                chrom,
                B,
                min_ac_g,
                min_ac_h,
                variants,
                adjust_lanc,
                impute,
                p_het_threshold,
            )
            time_ds = str(timedelta(seconds=int(time.perf_counter() - time_ds_start)))
            logger.info("Elapsed time: %s", time_ds)

        time_total = str(timedelta(seconds=int(time.perf_counter() - time_total_start)))
        logger.info("Step 2 completed in: %s", time_total)
