# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Genotype and local-ancestry access helpers."""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from lanctools import LancData
from numpy.typing import NDArray
from pgenlib import PgenReader, PvarReader


class PgenData:
    """Genotype and variant data backed directly by a PLINK2 fileset."""

    def __init__(self, plink_prefix: str):
        self.pgen = PgenReader(bytes(plink_prefix + ".pgen", "utf8"))
        self.pvar = PvarReader(bytes(plink_prefix + ".pvar", "utf8"))
        self.plink_prefix = plink_prefix
        self._closed = False

    def close(self) -> None:
        """Release the underlying PLINK readers."""
        if not self._closed:
            self.pgen.close()
            self.pvar.close()
            self._closed = True

    def get_geno(self, indices: NDArray[np.uint32]) -> NDArray[np.int32]:
        """Query phased genotypes for a set of variants."""
        if self._closed:
            raise RuntimeError("PgenData is closed")
        n_samples = self.pgen.get_raw_sample_ct()
        alleles = np.empty((len(indices), 2 * n_samples), dtype=np.int32)
        self.pgen.read_alleles_list(indices, alleles)
        return alleles.reshape(len(indices), n_samples, 2).transpose(1, 0, 2)


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
