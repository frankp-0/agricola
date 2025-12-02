from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from pgenlib import PgenReader, PvarReader
from pandas import DataFrame
import numpy as np
from numpy.typing import NDArray
import numba as nb
import jax
from jax import Array, numpy as jnp
from typing import Optional


### ─────────────────────────────────────────────────────────────
### Data structures
### ─────────────────────────────────────────────────────────────


@dataclass
class FlatLanc:
    """
    Stores .lanc file ancestry data in a flattened structure

    Attributes:
        left_haps: concatenated left haplotypes for all samples
        right_haps: concatenated right haplotypes for all samples
        breakpoints: concatenated breakpoints for all samples
        offsets: cumulative end indices separating samples
    """

    left_haps: NDArray[np.uint8]
    right_haps: NDArray[np.uint8]
    breakpoints: NDArray[np.uint32]
    offsets: NDArray[np.uint32]


### ─────────────────────────────────────────────────────────────
### I/O
### ─────────────────────────────────────────────────────────────


def _parse_lanc_line(
    line: str,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint32]]:
    """Parse a single line of .lanc file into a tuple of ancestries and breakpoints"""
    fields = line.strip().split()
    breakpoints, left_haps, right_haps = [], [], []
    for field in fields:
        breakpoint, hap_pair = field.split(":")
        breakpoints.append(int(breakpoint))
        left_haps.append(int(hap_pair[0]))
        right_haps.append(int(hap_pair[1]))
    return (
        np.array(left_haps, np.uint8),
        np.array(right_haps, np.uint8),
        np.array(breakpoints, np.uint32),
    )


def _read_lanc(path: str | Path) -> FlatLanc:
    """Read a .lanc file into a FlatLanc object"""
    left_haps, right_haps, breakpoints, offsets = [], [], [], [0]
    with open(path, "r") as f:
        next(f)
        for line in f:
            left_hap, right_hap, end = _parse_lanc_line(line)
            left_haps.append(left_hap)
            right_haps.append(right_hap)
            breakpoints.append(end)
            offsets.append(offsets[-1] + len(end))

    left_haps_all = np.concatenate(left_haps)
    right_haps_all = np.concatenate(right_haps)
    breakpoints_all = np.concatenate(breakpoints)
    return FlatLanc(
        left_haps_all,
        right_haps_all,
        breakpoints_all,
        np.array(offsets, dtype=np.uint32),
    )


def _get_info(pvar: PvarReader, indices: NDArray[np.unsignedinteger]) -> DataFrame:
    """Query variant information from pvar file

    Args:
        indices: A (V,) ndarray with indices of variants to query

    Returns:
        A (V, 6) pandas dataframe which information for each variant
    """
    chrom = [pvar.get_variant_chrom(i).decode("utf8") for i in indices]
    pos = [pvar.get_variant_pos(i) for i in indices]
    ref = [pvar.get_allele_code(i, 0).decode("utf8") for i in indices]
    alt = [pvar.get_allele_code(i, 1).decode("utf8") for i in indices]
    rsid = [pvar.get_variant_id(i).decode("utf8") for i in indices]
    df = DataFrame({"chrom": chrom, "pos": pos, "ref": ref, "alt": alt, "rsid": rsid})
    df["pos"] = df["pos"].astype("uint32")
    return df


### ─────────────────────────────────────────────────────────────
### Core
### ─────────────────────────────────────────────────────────────


@nb.njit(parallel=True)
def _get_lanc(
    left_haps: NDArray[np.uint8],
    right_haps: NDArray[np.uint8],
    breakpoints: NDArray[np.uint32],
    offsets: NDArray[np.uint32],
    indices: NDArray[np.unsignedinteger],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Query local ancestry"""
    n_samples = len(offsets) - 1
    n_variants = len(indices)
    left_out = np.empty((n_samples, n_variants), dtype=np.uint8)
    right_out = np.empty((n_samples, n_variants), dtype=np.uint8)

    for i in nb.prange(n_samples):
        start = offsets[i]
        end = offsets[i + 1]
        end_i = breakpoints[start:end]
        left_i = left_haps[start:end]
        right_i = right_haps[start:end]

        j = 0
        end_len = len(end_i)
        for q in range(n_variants):
            idx = indices[q]
            while j < end_len and idx >= end_i[j]:
                j += 1
            left_out[i, q] = left_i[j]
            right_out[i, q] = right_i[j]
    return left_out, right_out


def _get_geno(
    pgen: PgenReader, indices: NDArray[np.unsignedinteger]
) -> NDArray[np.int32]:
    """Query genotypes"""
    n = pgen.get_raw_sample_ct()
    v = len(indices)
    alleles = np.empty((v, 2 * n), dtype=np.int32)
    pgen.read_alleles_list(indices, alleles)
    return alleles.reshape(v, n, 2).transpose(1, 0, 2)


@jax.jit
def _deconv_geno(geno: Array, lanc: Array, ancestries: Array):
    """Get ancestry deconvoluted/masked genotypes"""
    left_haps_mask = (lanc[:, :, 0:1] == ancestries[None, None, :]).astype(jnp.float32)
    right_haps_mask = (lanc[:, :, 1:2] == ancestries[None, None, :]).astype(jnp.float32)
    geno_masked = left_haps_mask * geno[:, :, 0:1] + right_haps_mask * geno[:, :, 1:2]
    return geno_masked


### ─────────────────────────────────────────────────────────────
### GenoAncestryDataset
### ─────────────────────────────────────────────────────────────


@dataclass
class GenoAncestryDataset:
    """The genotype and local ancestry data for a single chromosome/dataset

    Attributes:
        pgen: A pgenlib PgenReader object
        pvar: A pgenlib PvarReader object
        lanc: A FlatLanc object containing local ancestry data
        ancestries: An ordered list of ancestry names
        plink_prefix: The prefix for the corresponding plink2 fileset
    """

    pgen: PgenReader
    pvar: PvarReader
    lanc: FlatLanc
    ancestries: list[str]
    plink_prefix: str

    @classmethod
    def from_plink(
        cls,
        plink_prefix: str,
        lanc_file: str | Path,
        ancestries: Optional[list[str]] = None,
    ) -> GenoAncestryDataset:
        """Constructs a GenoAncestryDataset from plink2 files

        Args:
            plink_prefix: A string with the prefix for a plink2 fileset
            lanc_file: A string or path for a .lanc file
            ancestries: An optional list of ordered ancestry names
            corresponding to the .lanc file

        Returns:
            A GenoAncestryDataset
        """
        pgen = PgenReader(bytes(plink_prefix + ".pgen", "utf8"))
        pvar = PvarReader(bytes(plink_prefix + ".pvar", "utf8"))
        lanc = _read_lanc(lanc_file)

        if ancestries is None:
            all_values = np.concatenate([lanc.left_haps, lanc.right_haps])
            ancestries = [str(i) for i in np.unique(all_values)]

        return cls(
            pgen=pgen,
            pvar=pvar,
            lanc=lanc,
            ancestries=ancestries,
            plink_prefix=plink_prefix,
        )

    def get_info(self, indices: NDArray[np.unsignedinteger]) -> DataFrame:
        return _get_info(self.pvar, indices)

    def get_lanc(self, indices: NDArray[np.unsignedinteger]) -> NDArray[np.uint8]:
        """Query local ancestries

        Args:
            indices: A (V,) ndarray with indices of variants to query

        Returns:
            An (N, V, 2) ndarray of local ancestries
        """
        left, right = _get_lanc(
            self.lanc.left_haps,
            self.lanc.right_haps,
            self.lanc.breakpoints,
            self.lanc.offsets,
            indices,
        )
        return np.stack((left, right), axis=-1)

    def get_lanc_unphased(
        self, indices: NDArray[np.unsignedinteger]
    ) -> NDArray[np.uint8]:
        """Query unphased local ancestry

        Args:
            indices: A (V,) ndarray with indices of variants to query

        Returns:
            An (N, V, len(self.ancestries) jax array of unphased local ancestries
        """
        lanc = jnp.asarray(self.get_lanc(indices), dtype=jnp.uint8)
        ancestries = jnp.arange(len(self.ancestries), dtype=jnp.uint8)
        left_haps_mask = (lanc[:, :, 0:1] == ancestries[None, None, :]).astype(
            jnp.float32
        )
        right_haps_mask = (lanc[:, :, 1:2] == ancestries[None, None, :]).astype(
            jnp.float32
        )
        return left_haps_mask + right_haps_mask

    def get_geno(self, indices: NDArray[np.unsignedinteger]) -> NDArray[np.int32]:
        """Query phased genotypes
        Args:
            indices: A (V,) ndarray with indices of variants to query
        Returns:
            An (N, V, 2) ndarray of phased genotypes
        """
        return _get_geno(self.pgen, indices)

    def get_lanc_geno(self, indices: NDArray[np.unsignedinteger]) -> Array:
        """Query genotypes deconvoluted/masked by ancestry

        Args:
            indices: A (V,) ndarray with indices of variants to query

        Returns:
            An (N, V, len(self.ancestries)) jax array of genotypes masked by ancestry
        """
        geno = jnp.asarray(self.get_geno(indices), dtype=jnp.int32)
        lanc = jnp.asarray(self.get_lanc(indices), dtype=jnp.uint8)
        ancestries = jnp.arange(len(self.ancestries), dtype=jnp.uint8)
        return _deconv_geno(geno, lanc, ancestries)
