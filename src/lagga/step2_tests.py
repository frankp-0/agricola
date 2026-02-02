# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text


"""Single variant associations

This module contains functions for calculating the tests statistics used in lagga step 2
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jax.scipy.special import expit
from .models import logistic_ridge


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
    LtL_inv = jnp.linalg.pinv(LtL)
    LtY = jnp.einsum("nbc,np->bcp", L, Y)
    beta_L = LtL_inv @ LtY
    r_L = Y[:, None, :] - jnp.einsum("nbc,bcp->nbp", L, beta_L)

    ## MSE under null
    sig2 = jnp.sum(r_L**2, axis=0) / (Y.shape[0] - L.shape[2])

    ## Get residualized genotypes
    GtL = jnp.einsum("nbk,nbc->bkc", G, L)
    HtL = jnp.einsum("nb,nbc->bc", H, L)
    G_res = G - jnp.einsum("nbc,bcd,bkd->nbk", L, LtL_inv, GtL)
    H_res = H - jnp.einsum("nbc,bcd,bd->nb", L, LtL_inv, HtL)

    ## Score for anc-deconvoluted genotypes
    U = jnp.einsum("nbk,nbp->bkp", G, r_L)
    I22_inv = jnp.linalg.pinv(jnp.einsum("nbk,nbl->bkl", G_res, G_res))
    chisq_het = jnp.einsum("bkp,bkl,blp->bp", U, I22_inv, U) / sig2
    beta_anc = U * jnp.diagonal(I22_inv, axis1=-1, axis2=-2)[:, :, None]
    df_het = jnp.linalg.matrix_rank(I22_inv)

    # Score for genotypes
    UH = jnp.einsum("nb,nbp->bp", H, r_L)
    I22_inv_H = (
        jnp.linalg.pinv(jnp.sum(H_res**2, axis=0).reshape((H_res.shape[1], 1, 1)))
        .squeeze(1)
        .squeeze(1)
    )
    chisq_hom = (UH**2) * I22_inv_H[:, None] / sig2

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
        "nbap,mbap,mbkp->nbkp", QL, QL, G * W_L_sqrt[:, :, None, :]
    )
    H_res = H * W_L_sqrt - jnp.einsum("nbap,mbap,mbp->nbp", QL, QL, H * W_L_sqrt)

    ## Score for anc-deconvoluted genotypes
    U = jnp.einsum("nbkp,nbp->bkp", G, R)
    I22_inv = jnp.linalg.pinv(jnp.einsum("nbkp,nblp->bpkl", G_res, G_res))
    chisq_het = jnp.einsum("bkp,bpkl,blp->bp", U, I22_inv, U)
    beta_anc = U * jnp.diagonal(I22_inv, axis1=-1, axis2=-2).transpose((0, 2, 1))
    df_het = jnp.linalg.matrix_rank(I22_inv)

    ## Score for genotypes
    UH = jnp.einsum("nbp,nbp->bp", H, R)
    I22_inv_H = (
        jnp.linalg.pinv(
            jnp.sum(H_res**2, axis=0).reshape((H_res.shape[1], H_res.shape[2], 1, 1))
        )
        .squeeze(2)
        .squeeze(2)
    )
    chisq_hom = (UH**2) * I22_inv_H

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

    ## Score for anc-deconvoluted genotypes
    U = jnp.einsum("nbk,np->bkp", G, Y)
    I22_inv = jnp.linalg.pinv(jnp.einsum("nbk,nbl->bkl", G, G))
    chisq_het = jnp.einsum("bkp,bkl,blp->bp", U, I22_inv, U) / sig2[None, :]
    beta_anc = U * jnp.diagonal(I22_inv, axis1=-1, axis2=-2)[:, :, None]
    df_het = jnp.linalg.matrix_rank(I22_inv)

    # Score for genotypes
    UH = jnp.einsum("nb,np->bp", H, Y)
    I22_inv_H = (
        jnp.linalg.pinv(jnp.einsum("nb,nb->b", H, H).reshape((H.shape[1], 1, 1)))
        .squeeze(1)
        .squeeze(1)
    )
    chisq_hom = (UH**2) * I22_inv_H[:, None]

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

    ## Get null model
    mu = expit(O)
    R = Y - mu

    ## Score for anc-deconvoluted genotypes
    U = jnp.einsum("nbkp,np->bkp", G, R)
    I22_inv = jnp.linalg.pinv(jnp.einsum("nbkp,nblp->bpkl", G, G))
    chisq_het = jnp.einsum("bkp,bpkl,blp->bp", U, I22_inv, U)
    beta_anc = U * jnp.diagonal(I22_inv, axis1=-1, axis2=-2).transpose((0, 2, 1))
    df_het = jnp.linalg.matrix_rank(I22_inv)

    ## Score for genotype
    UH = jnp.einsum("nbp,np->bp", H, R)
    I22_inv_H = (
        jnp.linalg.pinv(jnp.sum(H**2, axis=0).reshape((H.shape[1], H.shape[2], 1, 1)))
        .squeeze(2)
        .squeeze(2)
    )
    chisq_hom = (UH**2) * I22_inv_H

    return chisq_hom, chisq_het, beta_anc, df_het
