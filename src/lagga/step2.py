# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Single variant associations

This module uses whole-genome predictions from steps 0/1 to adjust traits and
perform single variant association tests. The entry-point is the `step2` function.
"""

import jax
import jax.numpy as jnp
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
from ._utils import stdize, assert_covar_full_rank, get_geno_lanc_deconv
from .models import logistic_ridge

### ─────────────────────────────────────────────────────────────
### Helper Functions
### ─────────────────────────────────────────────────────────────


def validate_step2_inputs(
    datasets: list[LancData],
    Y: ArrayLike,
    X: Optional[ArrayLike],
    step1_predictions: dict[str, np.ndarray],
    out_prefixes: list[str],
    B: int = 1000,
    idx_sample: Optional[ArrayLike] = None,
    variants: Optional[list[str]] = None,
) -> tuple[Array, Array, Optional[Array]]:
    """Validate input data for step1"""

    ## Y
    Y = jnp.asarray(Y)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N, P), got shape {Y.shape}")
    N, P = Y.shape

    if X is None:
        X = jnp.ones((Y.shape[0], 1), dtype=np.float32)
    else:
        X = jnp.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N, C), got shape {X.shape}")
        if X.shape[0] != N:
            raise ValueError(
                f"X.shape[0] must match Y.shape[0], got {X.shape[0]} vs {N}"
            )
        X = jnp.concatenate([jnp.ones((Y.shape[0], 1), dtype=np.float32), X], axis=1)
    X = stdize(X)
    assert_covar_full_rank(X)

    # datasets
    if not isinstance(datasets, (list, tuple)):
        raise TypeError(f"datasets must be a list of LancData, got {type(datasets)}")
    for i, ds in enumerate(datasets):
        if not isinstance(ds, LancData):
            raise TypeError(f"datasets[{i}] must be LancData, got {type(ds)}")
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
    if not len(out_prefixes) == len(datasets):
        raise ValueError("out_prefixes and datasets must have same number of elements")

    if N_pred != N:
        raise ValueError(f"step1_predictions arrays have N={N_pred} but Y has N={N}")

    if P_pred != P:
        raise ValueError(f"step1_predictions arrays have P={P_pred} but Y has P={P}")

    ## B
    if not isinstance(B, int) or B <= 0:
        raise ValueError(f"B must be a positive integer, got {B}")

    ## variants
    if variants is not None:
        if not isinstance(variants, (list, tuple)) or not all(
            isinstance(v, str) for v in variants
        ):
            raise TypeError("variants must be a list of strings")

    ## samples
    if idx_sample is not None:
        idx_sample = jnp.asarray(idx_sample)
        if idx_sample.ndim != 1:
            raise TypeError("idx_sample must be 1D")
        if idx_sample.dtype != np.uint32:
            raise TypeError("idx_sample must have dtype numpy.uint32")
        if not set(np.asarray(idx_sample)).issubset(np.arange(N, dtype=np.uint32)):
            raise ValueError("idx_sample outside range of N samples")

    return (Y, X, idx_sample)


### ─────────────────────────────────────────────────────────────
### Kernels
### ─────────────────────────────────────────────────────────────


@jax.jit
def _step2_qt_core(
    G: Array, L: Array, Y: Array, Q: Array
) -> tuple[Array, Array, Array, Array]:
    """Estimate coefficients and Wald statistic for quantitative traits

    Args:
        G: A (N, B, len(ancestries)) jax array of anc-deconvoluted genotypes
        L: A (N, B, len(ancestries) - 1) jax array of local ancestry
        Y: A (N, P) jax array of LOCO phenotypes
        Q: (N, C) jax array. The orthogonal matrix Q in the QR decomposition of the covariates

    Returns:
        A tuple with:
            1) chisq_hom: a (B, P) jax array with the chi-squared statistics for
                the homogeneous test
            2) chisq_het: a (B, P) jax array with the chi-squared statistics for
                the heterogeneous test
            3) beta_anc: a (B, len(ancestries), P) jax array with the estimated
                (heterogeneous) effect sizes
            4) df_het: a (B,) jax array with the degrees of freedom for the het test
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

    return chisq_hom, chisq_het, beta_anc, df_het


@jax.jit
def _step2_bt_core(
    G: Array,
    L: Array,
    Y: Array,
    Q_w: Array,
    W_sqrt: Array,
    O: Array,
) -> tuple[Array, Array, Array, Array]:
    """Estimate coefficients and Wald statistic for binary traits
    Args:
        G: A (N, B, len(ancestries)) jax array of anc-deconvoluted genotypes
        L: A (N, B, len(ancestries) - 1) jax array of local ancestry
        Y: A (N, P) jax array of phenotypes
        Q: (N, C) jax array. The orthogonal matrix Q in the QR decomposition of
        the covariates, weighted by estimated variance in the covariate-only model
        W_sqrt: (N, P) The square root of the estimated variance in the
        covariate-only model
        O: A (N, P) jax array of offsets (from covariate-only model)

    Returns:
        A tuple with:
            1) chisq_hom: a (B, P) jax array with the chi-squared statistics for
                the homogeneous test
            2) chisq_het: a (B, P) jax array with the chi-squared statistics for
                the heterogeneous test
            3) beta_anc: a (B, len(ancestries), P) jax array with the estimated
                (heterogeneous) effect sizes
            4) df_het: a (B,) jax array with the degrees of freedom for the het test
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
        jax.vmap(logistic_ridge, in_axes=(2, 1, 1, None, None, None)),
        in_axes=(1, None, None, None, None, None),
    )
    beta = logistic_model(L, Y, O, jnp.ones(L.shape[0]), 0, 10)
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


@jax.jit
def _step2_nolanc_qt_core(
    G: Array, Y: Array, Q: Array
) -> tuple[Array, Array, Array, Array]:
    """Estimate coefficients and Wald statistic for quantitative traits with no local-ancestry adjustment.

    Args:
        G: A (N, B, len(ancestries)) jax array of anc-deconvoluted genotypes
        L: A (N, B, len(ancestries) - 1) jax array of local ancestry
        Y: A (N, P) jax array of LOCO phenotypes
        Q: (N, C) jax array. The orthogonal matrix Q in the QR decomposition of the covariates

    Returns:
        A tuple with:
            1) chisq_hom: a (B, P) jax array with the chi-squared statistics for
                the homogeneous test
            2) chisq_het: a (B, P) jax array with the chi-squared statistics for
                the heterogeneous test
            3) beta_anc: a (B, len(ancestries), P) jax array with the estimated
                (heterogeneous) effect sizes
            4) df_het: a (B,) jax array with the degrees of freedom for the het test
    """
    ## Make homogeneous anc
    H = jnp.sum(G, axis=2)

    ## Adjust genotypes
    QG = jnp.einsum("nc,nbk->cbk", Q, G)
    QH = jnp.einsum("nc,nb->cb", Q, H)
    G = G - jnp.einsum("nc,cbk->nbk", Q, QG)
    H = H - jnp.einsum("nc,cb->nb", Q, QH)

    ## MSE under null
    sig2 = jnp.sum(Y**2, axis=0) / Y.shape[0]

    ## Get masks based on variance
    var_G = jnp.var(G, axis=0)
    var_H = jnp.var(H, axis=0)
    mask_G = var_G > 1e-8
    mask_H = var_H > 1e-8

    ## Score for anc-deconvoluted genotypes
    U = jnp.einsum("nbk,np->bkp", G, Y)
    U = U * mask_G[:, :, None]  # apply mask
    I22_inv = jax.scipy.linalg.inv(
        jnp.einsum("nbk,nbl->bkl", G, G) + 1e-10 * jnp.identity(G.shape[2])[None, :, :]
    )
    I22_inv = I22_inv * jnp.einsum("bk,bl->bkl", mask_G, mask_G)  # apply mask
    chisq_het = (
        jnp.einsum("bkp,bkp->bp", jnp.einsum("bkp,bkl->blp", U, I22_inv), U)
        / sig2[None, :]
    )

    beta_anc = U * jnp.diagonal(I22_inv, axis1=-1, axis2=-2)[:, :, None]

    # Score for genotypes
    UH = jnp.einsum("nb,np->bp", H, Y)
    UH = UH * mask_H[:, None]  # apply mask
    I22_inv_H = 1 / (jnp.einsum("nb,nb->b", H, H) + 1e-10)
    I22_inv_H = I22_inv_H * mask_H  # apply mask
    chisq_hom = (UH**2) * I22_inv_H[:, None]

    K = jnp.einsum("nbk,nbl->bkl", G, G)
    eigvals = jnp.linalg.eigvalsh(K)
    df_het = jnp.sum(eigvals > 1e-8, axis=1)

    return chisq_hom, chisq_het, beta_anc, df_het


@jax.jit
def _step2_nolanc_bt_core(
    G: Array,
    Y: Array,
    Q_w: Array,
    W_sqrt: Array,
    O: Array,
) -> tuple[Array, Array, Array, Array]:
    """Estimate coefficients and Wald statistic for binary traits without local-ancestry adjustment
    Args:
        G: A (N, B, len(ancestries)) jax array of anc-deconvoluted genotypes
        L: A (N, B, len(ancestries) - 1) jax array of local ancestry
        Y: A (N, P) jax array of phenotypes
        Q: (N, C) jax array. The orthogonal matrix Q in the QR decomposition of
        the covariates, weighted by estimated variance in the covariate-only model
        W_sqrt: (N, P) The square root of the estimated variance in the
        covariate-only model
        O: A (N, P) jax array of offsets (from covariate-only model)

    Returns:
        A tuple with:
            1) chisq_hom: a (B, P) jax array with the chi-squared statistics for
                the homogeneous test
            2) chisq_het: a (B, P) jax array with the chi-squared statistics for
                the heterogeneous test
            3) beta_anc: a (B, len(ancestries), P) jax array with the estimated
                (heterogeneous) effect sizes
            4) df_het: a (B,) jax array with the degrees of freedom for the het test
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

    ## Get masks based on variance
    var_G = jnp.var(G, axis=0)
    var_H = jnp.var(H, axis=0)
    mask_G = var_G > 1e-8
    mask_H = var_H > 1e-8

    ## Get null model
    mu = expit(O)
    R = Y - mu

    ## Score for anc-deconvoluted genotypes
    U = jnp.einsum("nbkp,np->bkp", G, R)
    U = U * mask_G  # apply mask
    I22_inv = jnp.linalg.inv(
        jnp.einsum("nbkp,nblp->bpkl", G, G)
        + 1e-8 * jnp.eye(G.shape[2])[None, None, :, :],
    )
    I22_inv = I22_inv * jnp.einsum("bkp,blp->bpkl", mask_G, mask_G)  # apply mask
    chisq_het = jnp.einsum("bkp,bpkl,blp->bp", U, I22_inv, U)

    UH = jnp.einsum("nbp,np->bp", H, R)
    UH = UH * mask_H  # apply mask
    I22_inv_H = 1 / (jnp.einsum("nbp,nbp->bp", H, H) + 1e-8)
    I22_inv_H = I22_inv_H * mask_H
    chisq_hom = (UH**2) * I22_inv_H

    beta_anc = U * jnp.diagonal(I22_inv, axis1=-1, axis2=-2).transpose((0, 2, 1))

    K = jnp.einsum("nbkp,nblp->bpkl", G, G)
    eigvals = jnp.linalg.eigvalsh(K)
    df_het = jnp.sum(eigvals > 1e-8, axis=2)
    return chisq_hom, chisq_het, beta_anc, df_het


### ─────────────────────────────────────────────────────────────
### Orchestration
### ─────────────────────────────────────────────────────────────


def _step2_block(
    dataset: LancData,
    Y: Array,
    Q: Array,
    trait_type: str,
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
    G, L = get_geno_lanc_deconv(dataset, block)
    L = L[:, :, 1:]

    if idx_sample is not None:
        G = G[idx_sample]
        L = L[idx_sample]

    if trait_type == "qt":
        if adjust_lanc:
            chisq_hom, chisq_het, beta_anc, df_het = _step2_qt_core(G, L, Y, Q)
        else:
            chisq_hom, chisq_het, beta_anc, df_het = _step2_nolanc_qt_core(G, Y, Q)
    elif trait_type == "bt":
        if adjust_lanc:
            chisq_hom, chisq_het, beta_anc, df_het = _step2_bt_core(
                G, L, Y, Q, extra_args["W_sqrt"], extra_args["O"]
            )
        else:
            chisq_hom, chisq_het, beta_anc, df_het = _step2_nolanc_bt_core(
                G, Y, Q, extra_args["W_sqrt"], extra_args["O"]
            )
    else:
        raise ValueError("trait_type must be qt or bt")

    if trait_type == "qt":
        log10p_het = chi2.logsf(chisq_het, df_het[:, None]) / np.log(10)
    else:
        log10p_het = chi2.logsf(chisq_het, df_het) / np.log(10)

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
    colnames: list[str] = ["log10p_het", "log10p_hom", *["beta_" + anc for anc in ancs]]

    ## Get info on variants in block
    block_info = dataset.get_info(block)  # all variants
    block_info["N"] = Y.shape[0]

    ## Filter out variants that fail min_ac
    anc_variant_mask = G.sum(axis=0).sum(axis=1) >= min_ac
    valid_idx = np.array(anc_variant_mask)
    block_info_filtered = block_info[valid_idx].reset_index(drop=True)
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


def _step2_dataset(
    dataset: LancData,
    Y: Array,
    pred_loco: dict[str, np.ndarray],
    X: Array,
    idx_sample: Optional[Array],
    out_prefix: str,
    phenotypes: list[str],
    trait_type: str,
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
        pred_loco: A dict of (N,P) numpy arrays containing LOCO predictions
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

            extra_args = {}
            if trait_type == "qt":
                Q, _ = jnp.linalg.qr(X, mode="reduced")
                Y = Y - pred_loco[chrom]
            elif trait_type == "bt":
                ## QR decomposition of weighted covariates
                mu = expit(pred_loco[chrom])
                W_sqrt = jnp.sqrt(mu * (1 - mu))
                Q, _ = jnp.linalg.qr(
                    jnp.moveaxis(
                        X[:, :, None] * W_sqrt[:, None, :], (0, 1, 2), (1, 2, 0)
                    ),
                    mode="reduced",
                )
                Q = Q.transpose((1, 2, 0))
                extra_args["W_sqrt"] = W_sqrt
                extra_args["O"] = pred_loco[chrom]
            else:
                raise ValueError("trait_type must be qt or bt")

            for block in blocks:
                result_dfs = _step2_block(
                    dataset,
                    Y,
                    Q,
                    trait_type,
                    block,
                    idx_sample,
                    min_ac,
                    extra_args,
                    adjust_lanc,
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


### ─────────────────────────────────────────────────────────────
### Public-facing
### ─────────────────────────────────────────────────────────────


def step2(
    datasets: list[LancData],
    Y: ArrayLike,
    X: Optional[ArrayLike],
    step1_predictions: dict[str, np.ndarray],
    out_prefixes: list[str],
    phenotypes: list[str],
    trait_type: "str",
    B: int = 1000,
    min_ac: int = 1,
    idx_sample: Optional[ArrayLike] = None,
    variants: Optional[list[str]] = None,
    adjust_lanc: bool = True,
) -> None:
    """Run step 2

    Args:
        datasets: A list of LancData objects
        Y: A (N, P) jax array of outcomes
        X: A (N, C) jax array of covariates
        step1_predictions: A dict with chromosome-specific linear predictions from step 1. The values are (N, P) NumPy arrays
        out_prefixes: A list of prefixes for each dataset. Outputs will be written to {output_prefix}_{phenotype}.parquet
        phenotypes: A list of phenotype names
        trait_type: either "qt" or "bt"
        B: The block size (max number of variants to read at once)
        min_ac: the minimum allele count threshold
        idx_sample: An optional numpy array with ordered indices of samples (in
            the psam file) to retain
        variants: An optional list of variant IDs to retain
        adjust_lanc: A boolean indicating whether to adjust tests for local ancestry
    """
    Y, X, idx_sample = validate_step2_inputs(
        datasets, Y, X, step1_predictions, out_prefixes, B, idx_sample, variants
    )

    if trait_type == "qt":
        X = jnp.concatenate([jnp.ones((Y.shape[0], 1), dtype=np.float32), X], axis=1)
        Q, _ = jnp.linalg.qr(X, mode="reduced")
        Y = stdize(Y - (Q @ (Q.T @ Y)))
        step1_prs = np.sum(np.stack(list(step1_predictions.values())), axis=0)
        pred_loco = {k: step1_prs - v for k, v in step1_predictions.items()}
    elif trait_type == "bt":
        covar_model = jax.vmap(
            logistic_ridge, in_axes=(None, 1, None, None, None), out_axes=1
        )
        offset_covar = X @ covar_model(
            X, Y, jnp.zeros(X.shape[0]), jnp.ones(Y.shape[0]), 0
        )
        chromosome_offsets = np.stack(
            [v - offset_covar for v in step1_predictions.values()], axis=1
        )
        pred_loco = {}
        for i, chrom in enumerate(step1_predictions):
            X_leave = np.delete(chromosome_offsets, i, axis=1)
            beta_chrom = jax.vmap(
                logistic_ridge, in_axes=(2, 1, 1, None, None), out_axes=1
            )(jnp.asarray(X_leave), Y, offset_covar, jnp.ones(Y.shape[0]), 0)
            eta_chrom = np.einsum("ncp,cp->np", X_leave, beta_chrom) + offset_covar
            pred_loco[chrom] = eta_chrom
    else:
        raise ValueError("trait_type must be qt or bt")

    for i, dataset in enumerate(datasets):
        pgen_path = dataset.plink_prefix + ".pgen"
        desc = f"Getting step 2 results for file: {pgen_path}"
        _step2_dataset(
            dataset,
            Y,
            pred_loco,
            X,
            idx_sample,
            out_prefixes[i],
            phenotypes,
            trait_type,
            desc,
            B,
            min_ac,
            variants,
            adjust_lanc,
        )
