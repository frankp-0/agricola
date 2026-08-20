# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Genotype and local-ancestry access helpers."""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from lanctools import LancData
from numpy.typing import NDArray


def _get_lanc_masks(dataset: LancData, indices: NDArray[np.uint32]) -> tuple[Array, Array]:
    lanc = jnp.asarray(dataset.get_lanc(indices))
    ancestries = jnp.arange(len(dataset.ancestries))
    return (
        (lanc[:, :, 0:1] == ancestries[None, None, :]).astype(int),
        (lanc[:, :, 1:2] == ancestries[None, None, :]).astype(int),
    )


def get_geno_lanc_deconv(dataset: LancData, indices: NDArray[np.uint32]) -> tuple[Array, Array]:
    """Return ancestry-deconvoluted genotypes and local ancestries."""
    geno = jnp.asarray(dataset.get_geno(indices))
    left_mask, right_mask = _get_lanc_masks(dataset, indices)
    geno_masked = left_mask * geno[:, :, 0:1] + right_mask * geno[:, :, 1:2]
    return geno_masked, left_mask + right_mask


def get_geno_deconv(dataset: LancData, indices: NDArray[np.uint32]) -> Array:
    """Return ancestry-deconvoluted genotypes."""
    geno = jnp.asarray(dataset.get_geno(indices))
    left_mask, right_mask = _get_lanc_masks(dataset, indices)
    return left_mask * geno[:, :, 0:1] + right_mask * geno[:, :, 1:2]
