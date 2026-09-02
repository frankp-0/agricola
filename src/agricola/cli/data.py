# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Input-file loading helpers for the agricola command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from jaxtyping import Array
    from lanctools import LancData
    from pandas import DataFrame


def load_variants(path: str | None) -> list[str] | None:
    if path is None:
        return None
    with Path(path).open() as file:
        return file.read().splitlines()


def _read_psam(path: str | Path) -> DataFrame:
    import pandas as pd

    with Path(path).open() as file:
        lines = file.readlines()

    header_line = next((line for line in reversed(lines) if line.startswith("#")), None)
    if header_line is not None:
        cols = header_line.lstrip("#").strip().split()
        df = pd.read_csv(path, sep=r"[ \t]", comment="#", names=cols, engine="python")
    else:
        data_line = next(line for line in lines if not line.startswith("#")).rstrip("\n")
        ncols = len(data_line.split())
        base = ["FID", "IID", "PAT", "MAT", "SEX"]
        if ncols <= len(base):
            names = base[:ncols]
        else:
            names = base + [f"PHENO{i}" for i in range(1, ncols - len(base) + 1)]
        df = pd.read_csv(path, sep=r"[ \t]", header=None, names=names, engine="python")

    if "IID" not in df.columns:
        raise ValueError("IID column not found in psam")
    return df


def load_samples(plinks: list[str], samples_file: str | None) -> tuple[list[str], list[str]]:
    df_psam = _read_psam(plinks[0] + ".psam")
    samples_psam = df_psam["IID"].astype(str).to_list()
    if samples_file is not None:
        with Path(samples_file).open() as file:
            samples_keep = [line.strip() for line in file.readlines()]
        samples = [sample for sample in samples_psam if sample in samples_keep]
    else:
        samples = samples_psam
    return samples, samples_psam


def _read_pheno_covar(path: str | Path) -> DataFrame:
    import pandas as pd

    with Path(path).open() as file:
        lines = file.readlines()

    header_line = next((line for line in reversed(lines) if line.startswith("#")), None)
    if header_line is not None:
        cols = header_line.lstrip("#").strip().split()
        df = pd.read_csv(path, sep=r"[ \t]", comment="#", names=cols, engine="python")
        if set(df.columns).issubset({"FID", "IID"}):
            raise ValueError("No phenotype columns found")
    else:
        data_line = next(line for line in lines if not line.startswith("#")).rstrip("\n")
        ncols = len(data_line.split())
        if ncols <= 2:
            raise ValueError("No phenotype columns found")
        names = ["FID", "IID"] + [f"PHENO{i}" for i in range(1, ncols - 1)]
        df = pd.read_csv(path, sep=r"[ \t]", header=None, names=names, engine="python")

    if "IID" not in df.columns:
        raise ValueError("IID column not found in psam")
    return df


def load_pheno_and_covars(
    pheno_file: str,
    covar_file: str | None,
    pheno: list[str] | None,
    pheno_list: str | None,
    covar: list[str] | None,
    covar_list: str | None,
    catcovar: list[str] | None,
    catcovar_list: str | None,
    samples_sub: list[str],
) -> tuple[Array, Array | None, list[str], list[str]]:
    import jax.numpy as jnp
    import pandas as pd

    df_pheno = _read_pheno_covar(pheno_file)
    sample_intersection = set(samples_sub) & set(df_pheno["IID"].astype(str))
    samples = [sample for sample in samples_sub if sample in sample_intersection]

    if pheno_list is not None and pheno is not None:
        raise ValueError("Only one of pheno and pheno_list may be provided")
    if covar_list is not None and covar is not None:
        raise ValueError("Only one of covar and covar_list may be provided")

    if pheno_list is not None:
        with Path(pheno_list).open() as file:
            phenotypes = [p.strip() for p in file]
    elif pheno is not None:
        phenotypes = pheno
    else:
        phenotypes = None

    if covar_list is not None:
        with Path(covar_list).open() as file:
            covariates = [p.strip() for p in file]
    elif covar is not None:
        covariates = covar
    else:
        covariates = None

    if catcovar_list is not None:
        with Path(catcovar_list).open() as file:
            catcovariates = [p.strip() for p in file]
    elif catcovar is not None:
        catcovariates = catcovar
    else:
        catcovariates = None

    if covar_file:
        df_covar = _read_pheno_covar(covar_file)
        covar_cols = ["IID"]
        if "FID" in df_covar.columns:
            covar_cols.append("FID")
        if covariates is not None:
            missing = [cov for cov in covariates if cov not in df_covar.columns]
            if missing:
                raise ValueError(f"Requested covariates missing from covar file: {missing}")
            covar_cols.extend(covariates)
        else:
            covar_cols.extend(col for col in df_covar.columns if col not in {"IID", "FID"})
        df_covar = df_covar[covar_cols]

        sample_intersection = set(samples) & set(df_covar["IID"].astype(str))
        samples = [sample for sample in samples if sample in sample_intersection]
        df_covar = df_covar[df_covar["IID"].astype(str).isin(samples)]  # pyright: ignore
        df_covar = df_covar.dropna()  # pyright: ignore

        sample_intersection = set(df_covar["IID"].astype(str))
        samples = [sample for sample in samples if sample in sample_intersection]

        df_covar["IID"] = pd.Categorical(
            df_covar["IID"].astype(str), categories=samples, ordered=True
        )
        df_covar = df_covar.sort_values(by="IID").reset_index(drop=True)  # pyright: ignore[reportCallIssue]
        df_covar_noid = df_covar.drop("IID", axis=1).drop("FID", axis=1, errors="ignore")
        X = jnp.asarray(
            pd.get_dummies(df_covar_noid, columns=catcovariates, dtype=float).to_numpy()
        )
    else:
        X = None

    df_pheno = df_pheno[df_pheno["IID"].astype(str).isin(samples)]
    df_pheno["IID"] = pd.Categorical(df_pheno["IID"].astype(str), categories=samples, ordered=True)
    df_pheno = df_pheno.sort_values("IID").reset_index(drop=True)  # pyright: ignore[reportCallIssue]
    if phenotypes is not None:
        df_pheno = df_pheno[["IID", *phenotypes]]
    Y = jnp.asarray(df_pheno.drop("IID", axis=1).drop("FID", axis=1, errors="ignore").to_numpy())
    phenotypes = df_pheno.drop("IID", axis=1).drop("FID", axis=1, errors="ignore").columns.to_list()
    return Y, X, phenotypes, samples


def load_lanc_data(
    plink_prefix: list[str] | None,
    plink_list: str | None,
    lanc_file: list[str] | None,
    lanc_list: str | None,
    ancestries: list[str] | None,
) -> tuple[list[LancData], list[str], list[str]]:
    from lanctools import LancData

    if plink_prefix and plink_list:
        raise typer.BadParameter("Specify either --plink OR --plink-list, not both")
    if lanc_file and lanc_list:
        raise typer.BadParameter("Specify either --lanc OR --lanc-list, not both")
    if plink_prefix is None:
        if plink_list is None:
            raise typer.BadParameter("Specify one of --plink or --plink-list")
        with Path(plink_list).open() as file:
            plinks = [line.strip() for line in file if line.strip()]
    else:
        plinks = plink_prefix
    if lanc_file is None:
        if lanc_list is None:
            raise typer.BadParameter("Specify one of --lanc or --lanc-list")
        with Path(lanc_list).open() as file:
            lancs = [line.strip() for line in file if line.strip()]
    else:
        lancs = lanc_file

    import logging

    logger = logging.getLogger("agricola")
    logger.info("Loading local ancestry data")
    datasets = [
        LancData(plink_prefix=plinks[i], lanc_file=lancs[i], ancestries=ancestries)
        for i in range(len(plinks))
    ]
    logger.info("Local ancestry data loaded\n")
    return datasets, plinks, lancs
