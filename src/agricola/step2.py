# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""agricola step 2 tests.

This module uses whole-genome predictions from steps 0/1 to adjust traits and
perform single variant association tests. The entry-point is the `step2` function.
"""

from pathlib import Path, PurePath
import shutil
import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, ArrayLike
import numpy as np
from scipy.stats import chi2
import pandas as pd
from tqdm import tqdm
from typing import Optional
from jax.scipy.special import expit
from jax import vmap
from lanctools import LancData
from ._internal.utils import (
    stdize,
    get_geno_lanc_deconv,
    TestType,
    TraitType,
)
from ._internal.step2_stats import (
    qt_score_lanc,
    qt_score_lanc_impute,
    qt_score_nolanc,
    qt_score_nolanc_impute,
    bt_score_lanc,
    bt_score_nolanc,
    qt_wald_lanc,
    qt_wald_lanc_impute,
    qt_wald_nolanc,
    qt_wald_nolanc_impute,
    bt_wald_lanc,
    bt_wald_nolanc,
)
from ._internal.inputs import validate_step2_inputs
from ._internal.models import logistic_ridge

### ─────────────────────────────────────────────────────────────
### Helpers
### ─────────────────────────────────────────────────────────────


@jit
def _prep_block(G, L, M, min_ac):
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
    ac_variant_mask = ac.sum(axis=1) >= min_ac
    return G, L, M, N_eff, af_lanc, prop_lanc, ac_variant_mask


class ParquetRotatingWriter:
    def __init__(self, output_dir, max_size_mb=1000):
        self.output_dir = output_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.buffer = []
        self.buffer_size = 0
        self.file_idx = 0

        # TODO: document that output dir cannot exist
        Path.mkdir(output_dir, parents=True, exist_ok=False)

    def write(self, df):
        self.buffer.append(df)
        self.buffer_size += df.memory_usage(deep=True).sum()

        if self.buffer_size >= self.max_size_bytes:
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        df = pd.concat(self.buffer, ignore_index=True)

        path = self.output_dir / f"part-{self.file_idx:05d}.parquet"

        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)

        print(f"Wrote {path}: {len(df):,} rows")

        self.file_idx += 1
        self.buffer.clear()
        self.buffer_size = 0

    def close(self):
        self.flush()


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
    idx_sample: Optional[Array],
    min_ac: int,
    extra_args: dict,
    adjust_lanc: bool,
    impute: bool,
) -> pd.DataFrame:
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
        impute: Whether to impute the phenotype. Much faster, but only available for qt traits
    """
    G, L = get_geno_lanc_deconv(dataset, block)
    if idx_sample is not None:
        G = G[idx_sample]
        L = L[idx_sample]

    G, L, M, N_eff, af_lanc, prop_lanc, ac_variant_mask = _prep_block(G, L, M, min_ac)
    valid_idx = np.asarray(ac_variant_mask)

    func_map = {
        (TraitType.QT, TestType.SCORE, True, False): (
            qt_score_lanc,
            lambda: (G, L, Y, Q, N_eff),
        ),
        (TraitType.QT, TestType.SCORE, True, True): (
            qt_score_lanc_impute,
            lambda: (G[:, :, :, 0], L[:, :, :, 0], Y, Q[:, :, 0], N_eff[0]),
        ),
        (TraitType.QT, TestType.WALD, True, False): (
            qt_wald_lanc,
            lambda: (G, L, Y, Q, N_eff),
        ),
        (TraitType.QT, TestType.WALD, True, True): (
            qt_wald_lanc_impute,
            lambda: (G[:, :, :, 0], L[:, :, :, 0], Y, Q[:, :, 0], N_eff[0]),
        ),
        (TraitType.BT, TestType.SCORE, True, False): (
            bt_score_lanc,
            lambda: (G, L, Y, Q, extra_args["O"], M, N_eff),
        ),
        (TraitType.BT, TestType.WALD, True, False): (
            bt_wald_lanc,
            lambda: (G, L, Y, Q, extra_args["O"], M, N_eff),
        ),
        (TraitType.QT, TestType.SCORE, False, False): (
            qt_score_nolanc,
            lambda: (G, Y, Q, N_eff),
        ),
        (TraitType.QT, TestType.SCORE, False, True): (
            qt_score_nolanc_impute,
            lambda: (G[:, :, :, 0], Y, Q[:, :, 0], N_eff[0]),
        ),
        (TraitType.QT, TestType.WALD, False, False): (
            qt_wald_nolanc,
            lambda: (G, Y, Q, N_eff),
        ),
        (TraitType.QT, TestType.WALD, False, True): (
            qt_wald_nolanc_impute,
            lambda: (G[:, :, :, 0], Y, Q[:, :, 0], N_eff[0]),
        ),
        (TraitType.BT, TestType.SCORE, False, False): (
            bt_score_nolanc,
            lambda: (G, Y, Q, extra_args["O"], M, N_eff),
        ),
        (TraitType.BT, TestType.WALD, False, False): (
            bt_wald_nolanc,
            lambda: (G, Y, Q, extra_args["O"], M, N_eff),
        ),
    }

    test_func, arg_fn = func_map[(trait_type, test_type, adjust_lanc, impute)]

    log10p_lrt: np.ndarray | None = None

    _, B, K, P = G.shape
    if test_type == TestType.WALD:
        chisq_hom, beta_hom, chisq_het, beta_het, df_het, chisq_lrt, df_lrt = test_func(
            *arg_fn()
        )
        chisq_lrt = jnp.reshape(chisq_lrt, (B, P))
        if df_lrt.ndim == 1:
            df_lrt = df_lrt[:, None]
        log10p_lrt = chi2.logsf(chisq_lrt, df_lrt) / np.log(10)
    else:
        chisq_hom, beta_hom, chisq_het, beta_het, df_het = test_func(*arg_fn())

    chisq_hom = jnp.reshape(chisq_hom, (B, P))
    beta_hom = jnp.reshape(beta_hom, (B, P))
    beta_het = jnp.reshape(beta_het, (B, K, P))
    if df_het.ndim < 2:
        df_het = jnp.broadcast_to(df_het[:, None], (B, P))
    df_het = jnp.reshape(df_het, (B, P))
    chisq_het = jnp.reshape(chisq_het, (B, P))

    log10p_het = chi2.logsf(chisq_het, df_het) / np.log(10)
    log10p_hom = chi2.logsf(chisq_hom, 1) / np.log(10)

    ## Create array with results
    result_components = [
        log10p_het[:, None, :],
        beta_het,
        log10p_hom[:, None, :],
        beta_hom[:, None, :],
        np.broadcast_to(N_eff, (B, 1, P)),
        af_lanc,
        prop_lanc,
    ]

    ## Get column names for results
    ancs = dataset.ancestries
    colnames: list[str] = [
        "LOG10P_HET",
        *["BETA_" + anc for anc in ancs],
        "LOG10P_HOM",
        "BETA_HOM",
        "N",
        *["AF_" + anc for anc in ancs],
        *["LA_PROP_" + anc for anc in ancs],
    ]

    if log10p_lrt is not None:
        result_components.append(log10p_lrt[:, None, :])
        colnames.append("LOG10P_LRT")

    result_arr = np.concatenate(result_components, axis=1)

    ## Get info on variants in block
    block_info = dataset.get_info(block)  # all variants

    ## Create single DataFrame
    p = Y.shape[1]
    result_df = pd.concat(
        [
            pd.concat(
                [
                    block_info[valid_idx[:, i]].reset_index(drop=True),
                    pd.DataFrame(
                        data=result_arr[valid_idx[:, i], :, i],
                        columns=pd.Index(colnames),
                    ),
                    pd.DataFrame(
                        {"phenotype": [phenotypes[i]] * np.sum(valid_idx[:, i])}
                    ),
                ],
                axis=1,
            )
            for i in range(p)
        ],
        axis=0,
    )

    return result_df


def _step2_dataset(
    dataset: LancData,
    writer: ParquetRotatingWriter,
    Y: Array,
    M: Array,
    step1_predictions: dict[str, np.ndarray],
    X: Array,
    idx_sample: Optional[Array],
    phenotypes: list[str],
    trait_type: TraitType,
    test_type: TestType,
    desc: str,
    chrom: Optional[str],
    B: int = 500,
    min_ac: int = 1,
    variants: Optional[list[str]] = None,
    adjust_lanc: bool = True,
    impute: bool = False,
) -> None:
    """Run step 2 for a single dataset

    Args:
        dataset: A LancData object
        Y: A (N, P) jax array of outcomes
        step1_predictions: A dict of (N,P) pandas DataFrames containing LOCO predictions
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
        impute: Whether to impute the phenotype. Much faster, but only available for qt traits
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
    if chrom is not None:
        chroms = [chrom]
    else:
        chr_seen = set()
        chroms = [
            chrom
            for chrom in chromosomes
            if not (chrom in chr_seen or chr_seen.add(chrom))
        ]

    n_blocks = sum(
        (len([c for c in chromosomes if c == chrom]) + B - 1) // B for chrom in chroms
    )

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
            Yc = Y  # chromosome-specific Y
            if trait_type == TraitType.QT:
                Yc = Yc - step1_predictions[chrom]
                Yc = Yc - jnp.sum(Yc * M, axis=0) / jnp.sum(M, axis=0)
            else:
                mu = expit(step1_predictions[chrom])
                W_sqrt = jnp.sqrt(mu * (1 - mu))
                extra_args["W_sqrt"] = W_sqrt

                O = jnp.asarray(step1_predictions[chrom])
                extra_args["O"] = O

            Yc = Yc * M

            for block in blocks:
                result_df = _step2_block(
                    dataset,
                    Yc,
                    M,
                    Q,
                    phenotypes,
                    trait_type,
                    test_type,
                    block,
                    idx_sample,
                    min_ac,
                    extra_args,
                    adjust_lanc,
                    impute,
                )

                writer.write(result_df)
                pbar.update(1)


def step2(
    datasets: list[LancData],
    Y: ArrayLike,
    X: Optional[ArrayLike],
    step1_predictions: dict[str, pd.DataFrame],
    outdir: str | Path,
    phenotypes: list[str],
    trait_type: str = "qt",
    test_type: str = "score",
    chrom: Optional[str] = None,
    B: int = 1000,
    min_ac: int = 1,
    idx_sample: Optional[ArrayLike] = None,
    variants: Optional[list[str]] = None,
    adjust_lanc: bool = True,
    impute: bool = False,
    overwrite: bool = True,
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
        test_type: Either "score" or "wald"
        B: The block size (max number of variants to read at once)
        min_ac: the minimum allele count threshold
        idx_sample: An optional numpy array with ordered indices of samples (in
            the psam file) to retain
        variants: An optional list of variant IDs to retain
        adjust_lanc: A boolean indicating whether to adjust tests for local ancestry
        impute: Whether to impute the phenotype. Much faster, but only available
            for qt traits. If all phenotypes are non-missing, this is ignored.
        overwrite: Whether to overwrite the outdir if it already exists
    """
    ## Create writer
    outdir_path = Path(outdir)
    if outdir_path.exists():
        shutil.rmtree(outdir_path)

    writer = ParquetRotatingWriter(output_dir=outdir_path)

    if impute:
        M = jnp.ones(shape=jnp.asarray(Y).shape)
    else:
        M = (~jnp.isnan(jnp.asarray(Y))).astype(jnp.float32)

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
    else:
        if impute:
            raise ValueError("impute must be False for binary traits")
        beta_covar = vmap(logistic_ridge, in_axes=(None, 1, None, None, None))(
            X, Y, jnp.zeros(Y.shape[0]), jnp.ones(Y.shape[0]), 0
        )
        offset_covar = X @ beta_covar.T
        for k in step1_predictions_np.keys():
            step1_predictions_np[k] = step1_predictions_np[k] + offset_covar

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
            writer,
            Y,
            M,
            step1_predictions_np,
            X,
            idx_sample,
            phenotypes,
            trait_type_enum,
            test_type_enum,
            desc,
            chrom,
            B,
            min_ac,
            variants,
            adjust_lanc,
            impute,
        )
    writer.close()
