# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Variant selection and chromosome grouping helpers."""

import numpy as np
from lanctools import LancData
from numpy.typing import NDArray


def get_variant_indices(
    dataset: LancData, variants: list[str] | None = None
) -> NDArray[np.uint32]:
    """Return dataset variant indices, optionally restricted to variant IDs."""
    indices = np.arange(dataset.pvar.get_variant_ct(), dtype=np.uint32)
    if variants is None:
        return indices

    requested = set(variants)
    return np.asarray(
        [
            index
            for index in indices
            if dataset.pvar.get_variant_id(index).decode("utf8") in requested
        ],
        dtype=np.uint32,
    )


def group_variant_indices_by_chromosome(
    dataset: LancData, indices: NDArray[np.uint32]
) -> dict[str, NDArray[np.uint32]]:
    """Group variant indices by chromosome while preserving dataset order."""
    grouped: dict[str, list[np.uint32]] = {}
    for index in indices:
        chromosome = dataset.pvar.get_variant_chrom(index).decode("utf8")
        grouped.setdefault(chromosome, []).append(index)
    return {
        chromosome: np.asarray(chromosome_indices, dtype=np.uint32)
        for chromosome, chromosome_indices in grouped.items()
    }
