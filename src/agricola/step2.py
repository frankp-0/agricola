# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""agricola step 2 tests.

This module uses whole-genome predictions from steps 0/1 to adjust traits and
perform single variant association tests. The entry-point is the `step2` function.
"""

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, ArrayLike
import numpy as np
from scipy.stats import chi2
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from typing import Optional
from jax.scipy.special import expit
from lanctools import LancData
from ._internal.utils import (
    stdize,
    get_geno_lanc_deconv,
    get_geno_deconv,
    TestType,
    TraitType,
)
from ._internal.step2_stats import (
    qt_score_lanc,
    qt_score_nolanc,
    bt_score_lanc,
    bt_score_nolanc,
    qt_wald_lanc,
    qt_wald_nolanc,
    bt_wald_lanc,
    bt_wald_nolanc,
)
from ._internal.inputs import validate_step2_inputs

### ─────────────────────────────────────────────────────────────
### Orchestration
### ─────────────────────────────────────────────────────────────


def _step2_block(
    dataset: LancData,
    Y: Array,
    M: Array,
    Q: Array,
    trait_type: TraitType,
    test_type: TestType,
    block: np.ndarray,
    idx_sample: Optional[Array],
    min_ac: int,
    extra_args: dict,
    adjust_lanc: bool,
) -> list[pd.DataFrame]:
    """Run step 2 for a single block of variants

    Args:
        dataset: A LancData object
        Y: A (N, P) jax array of outcomes
        Q: A (N, C) jax array. The orthogonal matrix Q in the QR decomposition of
            the covariates. For trait_type="bt", this is weighted by estimated
            variance in the covariate-only model.
        block: A (B,) jax array of variant indices
        idx_sample: An optional numpy array with ordered indices of samples (in
            the psam file) to retain
        min_ac: the minimum allele count threshold
        extra_args: A dict containing extra arguments needed for trait_type="bt"
        adjust_lanc: A boolean indicating whether to adjust tests for local ancestry
    """

    @jit
    def adjust_G(G, M, N_eff):
        G = (
            G[:, :, :, None]
            - jnp.sum(G[:, :, :, None] * M[:, None, None, :], axis=0) / N_eff
        )
        G = G * M[:, None, None, :]
        return G

    N_eff = jnp.sum(M, axis=0)
    G, L = get_geno_lanc_deconv(dataset, block)
    if idx_sample is not None:
        G = G[idx_sample]
        L = L[idx_sample]
    ac = (G[:, :, :, None] * M[:, None, None, :]).sum(axis=0)
    lac = (L[:, :, :, None] * M[:, None, None, :]).sum(axis=0)
    af_lanc = ac / lac
    prop_lanc = jnp.sum(L[:, :, :, None], axis=0) / N_eff[None, None, :] / 2
    L = L[:, :, 1:]
    G = adjust_G(G, M, N_eff)
    L = adjust_G(L, M, N_eff)
    func_map = {
        (TraitType.QT, TestType.SCORE, True): (
            qt_score_lanc,
            lambda: (G, L, Y, Q, N_eff),
        ),
        (TraitType.QT, TestType.WALD, True): (
            qt_wald_lanc,
            lambda: (G, L, Y, Q, N_eff),
        ),
        (TraitType.BT, TestType.SCORE, True): (
            bt_score_lanc,
            lambda: (G, L, Y, Q, extra_args["O"], M, N_eff),
        ),
        (TraitType.BT, TestType.WALD, True): (
            bt_wald_lanc,
            lambda: (G, L, Y, Q, extra_args["O"], M, N_eff),
        ),
        (TraitType.QT, TestType.SCORE, False): (
            qt_score_nolanc,
            lambda: (G, Y, Q, N_eff),
        ),
        (TraitType.QT, TestType.WALD, False): (
            qt_wald_nolanc,
            lambda: (G, Y, Q, N_eff),
        ),
        (TraitType.BT, TestType.SCORE, False): (
            bt_score_nolanc,
            lambda: (G, Y, Q, extra_args["O"], M, N_eff),
        ),
        (TraitType.BT, TestType.WALD, False): (
            bt_wald_nolanc,
            lambda: (G, Y, Q, extra_args["O"], M, N_eff),
        ),
    }

    ## Filter variants with low ac
    ac_variant_mask = ac.sum(axis=1) >= min_ac
    valid_idx = np.array(ac_variant_mask)

    test_func, arg_fn = func_map[(trait_type, test_type, adjust_lanc)]
    chisq_hom, beta_hom, chisq_het, beta_het, df_het = test_func(*arg_fn())

    log10p_het = chi2.logsf(chisq_het, df_het) / np.log(10)
    log10p_hom = chi2.logsf(chisq_hom, 1) / np.log(10)

    ## Create array with results
    B, P = log10p_hom.shape
    result_arr = np.concatenate(
        [
            log10p_het[:, None, :],
            beta_het,
            log10p_hom[:, None, :],
            beta_hom[:, None, :],
            np.broadcast_to(N_eff, (B, 1, P)),
            af_lanc,
            prop_lanc,
        ],
        axis=1,
    )

    ## Get column names for results
    ancs = dataset.ancestries
    colnames: list[str] = [
        "log10p_het",
        *["beta_" + anc for anc in ancs],
        "log10p_hom",
        "beta_hom",
        "N",
        *["AF_" + anc for anc in ancs],
        *["LAprop_" + anc for anc in ancs],
    ]

    ## Get info on variants in block
    block_info = dataset.get_info(block)  # all variants

    ## Format results into list of dataframes
    p = Y.shape[1]
    result_dfs = [
        pd.concat(
            [
                block_info[valid_idx[:, i]].reset_index(drop=True),
                pd.DataFrame(
                    data=result_arr[valid_idx[:, i], :, i], columns=pd.Index(colnames)
                ),
            ],
            axis=1,
        )
        for i in range(p)
    ]

    return result_dfs


def _step2_dataset(
    dataset: LancData,
    Y: Array,
    M: Array,
    step1_predictions: dict[str, np.ndarray],
    X: Array,
    idx_sample: Optional[Array],
    out_prefix: str,
    phenotypes: list[str],
    trait_type: TraitType,
    test_type: TestType,
    desc: str,
    B: int = 2000,
    min_ac: int = 1,
    variants: Optional[list[str]] = None,
    adjust_lanc: bool = True,
) -> None:
    """Run step 2 for a single dataset

    Args:
        dataset: A LancData object
        Y: A (N, P) jax array of outcomes
        step1_predictions: A dict of (N,P) numpy arrays containing LOCO predictions
        X: A (N, C) jax array of covariates
        idx_sample: An optional numpy array with ordered indices of samples (in
            the psam file) to retain
        out_prefix: Outputs will be written to {output_prefix}_{phenotype}.parquet
        phenotypes: A list of phenotype names
        trait_type: either "qt" or "bt"
        desc: A string with the description to print for the progress bar
        B: The block size (max number of variants to read at once)
        min_ac: the minimum allele count threshold
        variants: An optional list of variant IDs to retain
        adjust_lanc: A boolean indicating whether to adjust tests for local ancestry
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
    chr_seen = set()
    chroms = [
        chrom for chrom in chromosomes if not (chrom in chr_seen or chr_seen.add(chrom))
    ]
    n_blocks = sum(
        (len([c for c in chromosomes if c == chrom]) + B - 1) // B for chrom in chroms
    )

    ## Initialize output
    out_paths = [Path(f"{out_prefix}_{pheno}.parquet") for pheno in phenotypes]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)

    writers: list[Optional[pq.ParquetWriter]] = [None] * len(phenotypes)
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

            extra_args = {}

            Q = jnp.linalg.qr(X.transpose(2, 0, 1), mode="reduced")[0].transpose(
                1, 2, 0
            )
            if trait_type == TraitType.QT:
                Y = Y - step1_predictions[chrom]
                Y = Y - jnp.sum(Y * M, axis=0) / jnp.sum(M, axis=0)
            else:
                mu = expit(step1_predictions[chrom])
                W_sqrt = jnp.sqrt(mu * (1 - mu))
                extra_args["W_sqrt"] = W_sqrt

                O = jnp.asarray(step1_predictions[chrom])
                extra_args["O"] = O

            Y = Y * M

            for block in blocks:
                result_dfs = _step2_block(
                    dataset,
                    Y,
                    M,
                    Q,
                    trait_type,
                    test_type,
                    block,
                    idx_sample,
                    min_ac,
                    extra_args,
                    adjust_lanc,
                )

                for i, df in enumerate(result_dfs):
                    table = pa.Table.from_pandas(df, preserve_index=False)

                    writer = writers[i]

                    if writer is None:
                        writer = pq.ParquetWriter(out_paths[i], table.schema)
                        writers[i] = writer
                        schemas[i] = table.schema
                    else:
                        table = pa.Table.from_pandas(
                            df, schema=schemas[i], preserve_index=False
                        )
                    writer.write_table(table)
                pbar.update(1)


def step2(
    datasets: list[LancData],
    Y: ArrayLike,
    X: Optional[ArrayLike],
    step1_predictions: dict[str, np.ndarray],
    out_prefixes: list[str],
    phenotypes: list[str],
    trait_type: str = "qt",
    test_type: str = "score",
    B: int = 1000,
    min_ac: int = 1,
    idx_sample: Optional[ArrayLike] = None,
    variants: Optional[list[str]] = None,
    adjust_lanc: bool = True,
) -> None:
    """Perform agricola step 2

    Args:
        datasets: A list of LancData objects (either single object or one
            per-chromosome)
        Y: A (N, P) jax array of outcomes
        X: A (N, C) jax array of covariates
        step1_predictions: A dict with LOCO linear predictions from step 1. The values are (N, P) NumPy arrays
        out_prefixes: A list of prefixes for each dataset. Outputs will be written to {output_prefix}_{phenotype}.parquet
        phenotypes: A list of phenotype names
        trait_type: either "qt" or "bt"
        B: The block size (max number of variants to read at once)
        min_ac: the minimum allele count threshold
        idx_sample: An optional numpy array with ordered indices of samples (in
            the psam file) to retain
        variants: An optional list of variant IDs to retain
        adjust_lanc: A boolean indicating whether to adjust tests for local ancestry
        test_type: Either "score" or "wald"
    """
    M = (~jnp.isnan(Y)).astype(jnp.float32)

    Y, X, idx_sample, test_type_enum, trait_type_enum = validate_step2_inputs(
        datasets,
        Y,
        X,
        step1_predictions,
        out_prefixes,
        B,
        idx_sample,
        variants,
        test_type,
        trait_type,
    )

    ## Adjust phenotype for covariates to match step 1
    if trait_type_enum == TraitType.QT:
        Q, _ = jnp.linalg.qr(X, mode="reduced")
        Y = stdize(Y - (Q @ (Q.T @ Y)))

    ## Adjust covariates for per-phenotype missingness
    X = X[:, :, None] - jnp.sum(X[:, :, None] * M[:, None, :], axis=0) / jnp.sum(
        M, axis=0
    )
    X = X * M[:, None, :]

    for i, dataset in enumerate(datasets):
        pgen_path = dataset.plink_prefix + ".pgen"
        desc = f"Getting step 2 results for file: {pgen_path}"
        _step2_dataset(
            dataset,
            Y,
            M,
            step1_predictions,
            X,
            idx_sample,
            out_prefixes[i],
            phenotypes,
            trait_type_enum,
            test_type_enum,
            desc,
            B,
            min_ac,
            variants,
            adjust_lanc,
        )
