# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""The command line interface for agricola."""

from __future__ import annotations
import os
import pickle
import typer
from typing import Optional, TYPE_CHECKING
from importlib.metadata import version, PackageNotFoundError
import logging
import sys

if TYPE_CHECKING:
    from pandas import DataFrame
    from jaxtyping import Array
    from lanctools import LancData

DEFAULT_H2_PRIORS = "0.01,0.255,0.5,0.745,0.99"

app = typer.Typer(help="agricola CLI")
logger = logging.getLogger("agricola")
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)

### ─────────────────────────────────────────────────────────────
### Helpers
### ─────────────────────────────────────────────────────────────


def _list_from_csv(arg: Optional[str]) -> Optional[list[str]]:
    return None if arg is None else [x.strip() for x in arg.split(",")]


def _load_variants(path: Optional[str]) -> Optional[list[str]]:
    return None if path is None else open(path).read().splitlines()


def _read_psam(path) -> DataFrame:
    import pandas as pd

    with open(path) as f:
        lines = f.readlines()

    header_line = None
    for line in reversed(lines):
        if line.startswith("#"):
            header_line = line
            break

    if header_line is not None:
        cols = header_line.lstrip("#").strip().split()
        df = pd.read_csv(path, sep="\\s+", comment="#", names=cols)

    else:
        data_line = next(l for l in lines if not l.startswith("#")).rstrip("\n")
        ncols = len(data_line.split())

        base = ["FID", "IID", "PAT", "MAT", "SEX"]
        if ncols <= len(base):
            names = base[:ncols]
        else:
            extra = [f"PHENO{i}" for i in range(1, ncols - len(base) + 1)]
            names = base + extra

        df = pd.read_csv(path, sep="\\s+", header=None, names=names)

    if "IID" not in df.columns:
        raise ValueError("IID column not found in psam")

    return df


def _load_samples(plinks: list[str], samples_file: Optional[str]):
    df_psam = _read_psam(plinks[0] + ".psam")
    samples_psam = df_psam["IID"].astype(str).to_list()

    if samples_file is not None:
        with open(samples_file, "r") as f:
            samples_keep = [line.strip() for line in f.readlines()]
        samples = [sample for sample in samples_psam if sample in samples_keep]
    else:
        samples = samples_psam
    return samples, samples_psam


def _read_pheno_covar(path) -> DataFrame:
    import pandas as pd

    with open(path) as f:
        lines = f.readlines()

    header_line = None
    for line in reversed(lines):
        if line.startswith("#"):
            header_line = line
            break

    if header_line is not None:
        cols = header_line.lstrip("#").strip().split()
        df = pd.read_csv(path, sep="\\s+", comment="#", names=cols)
        if set(df.columns).issubset(set(["FID", "IID"])):
            raise ValueError("No phenotype columns found")
    else:
        data_line = next(l for l in lines if not l.startswith("#")).rstrip("\n")
        ncols = len(data_line.split())

        if ncols <= 2:
            raise ValueError("No phenotype columns found")
        else:
            extra = [f"PHENO{i}" for i in range(1, ncols - 1)]
            names = ["FID", "IID"] + extra

        df = pd.read_csv(path, sep="\\s+", header=None, names=names)

    if "IID" not in df.columns:
        raise ValueError("IID column not found in psam")

    return df


def _load_pheno_and_covars(
    pheno_file: str,
    covar_file: Optional[str],
    pheno: Optional[list[str]],
    pheno_list: Optional[str],
    covar: Optional[list[str]],
    covar_list: Optional[str],
    catcovar: Optional[list[str]],
    catcovar_list: Optional[str],
    samples_sub: list[str],
) -> tuple[Array, Optional[Array], list[str], list[str]]:
    import pandas as pd
    import jax.numpy as jnp

    df_pheno = _read_pheno_covar(pheno_file)
    samples_pheno = df_pheno["IID"].astype(str).to_list()

    samples: list[str] = [sample for sample in samples_sub if sample in samples_pheno]

    if pheno_list is not None and pheno is not None:
        raise ValueError("Only one of pheno and pheno_list may be provided")
    if covar_list is not None and covar is not None:
        raise ValueError("Only one of covar and covar_list may be provided")

    if pheno_list is not None:
        with open(pheno_list, "r") as f:
            phenotypes = [p.strip() for p in f]
    elif pheno is not None:
        phenotypes = pheno
    else:
        phenotypes = None

    if covar_list is not None:
        with open(covar_list, "r") as f:
            covariates = [p.strip() for p in f]
    elif covar is not None:
        covariates = covar
    else:
        covariates = None

    if catcovar_list is not None:
        with open(catcovar_list, "r") as f:
            catcovariates = [p.strip() for p in f]
    elif catcovar is not None:
        catcovariates = catcovar
    else:
        catcovariates = None

    if covar_file:
        df_covar = _read_pheno_covar(covar_file).dropna()
        samples_covar = df_covar["IID"].astype(str).to_list()
        samples = [sample for sample in samples if sample in samples_covar]
        df_covar = df_covar[df_covar["IID"].astype(str).isin(samples)]
        df_covar["IID"] = df_covar["IID"].astype(str)
        df_covar["IID"] = pd.Categorical(
            df_covar["IID"], categories=samples, ordered=True
        )
        df_covar = df_covar.sort_values(by="IID").reset_index(drop=True)  # pyright: ignore
        if covariates is not None:
            df_covar = df_covar[["IID"] + covariates]

        df_covar_noid = df_covar.drop("IID", axis=1).drop(
            "FID", axis=1, errors="ignore"
        )
        X = jnp.asarray(
            pd.get_dummies(df_covar_noid, columns=catcovariates, dtype=float).to_numpy()
        )
    else:
        X = None

    df_pheno = df_pheno[df_pheno["IID"].astype(str).isin(samples)]
    df_pheno["IID"] = df_pheno["IID"].astype(str)
    df_pheno["IID"] = pd.Categorical(df_pheno["IID"], categories=samples, ordered=True)
    df_pheno = df_pheno.sort_values("IID").reset_index(drop=True)  # pyright: ignore

    if phenotypes is not None:
        df_pheno = df_pheno[["IID"] + phenotypes]

    Y = jnp.asarray(
        df_pheno.drop("IID", axis=1).drop("FID", axis=1, errors="ignore").to_numpy()
    )

    phenotypes = (
        df_pheno.drop("IID", axis=1)
        .drop("FID", axis=1, errors="ignore")
        .columns.to_list()
    )

    return (
        Y,
        X,
        phenotypes,
        samples,
    )


def _load_lanc_data(
    plink_prefix: Optional[list[str]],
    plink_list: Optional[str],
    lanc_file: Optional[list[str]],
    lanc_list: Optional[str],
    ancestries: Optional[list[str]],
) -> tuple[list[LancData], list[str], list[str]]:
    from lanctools import LancData

    if plink_prefix and plink_list:
        raise typer.BadParameter("Specify either --plink OR --plink-list, not both")
    if lanc_file and lanc_list:
        raise typer.BadParameter("Specify either --lanc OR --lanc-list, not both")

    plinks: list[str]
    if plink_prefix is None:
        if plink_list is None:
            raise typer.BadParameter("Specify one of --plink or --plink-list")
        with open(plink_list) as f:
            plinks = [line.strip() for line in f if line.strip()]
    else:
        plinks = plink_prefix

    lancs: list[str]
    if lanc_file is None:
        if lanc_list is None:
            raise typer.BadParameter("Specify one of --lanc or --lanc-list")
        with open(lanc_list) as f:
            lancs = [line.strip() for line in f if line.strip()]
    else:
        lancs = lanc_file

    logger.info("Loading local ancestry data")
    datasets = [
        LancData(plink_prefix=plinks[i], lanc_file=lancs[i], ancestries=ancestries)
        for i in range(len(plinks))
    ]
    logger.info("Local ancestry data loaded\n")

    return (
        datasets,
        plinks,
        lancs,
    )


def _get_version() -> str:
    try:
        return version("agricola")
    except PackageNotFoundError:
        return "unknown"


def _setup_logging(log_file: Optional[str], verbose: bool = False) -> None:
    logger = logging.getLogger()

    if verbose:
        logger.setLevel(logging.DEBUG)
        format = "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(message)s"
    else:
        logger.setLevel(logging.INFO)
        format = "%(levelname)s: %(message)s"

    formatter = logging.Formatter(format)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optionally log to file at the selected level
    if log_file:
        if os.path.exists(log_file):
            os.remove(log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def _get_options_msg(options: dict[str, str]) -> str:
    option_msg = ["Command options:"]
    for key, value in options.items():
        option_msg.append(f"  {key} = {value}")
    option_msg = "\n".join(option_msg)
    option_msg = option_msg + "\n"
    return option_msg


### ─────────────────────────────────────────────────────────────
### App
### ─────────────────────────────────────────────────────────────


@app.callback(invoke_without_command=True)
def main(
    version_flag: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the agricola version and exit",
        is_eager=True,
    ),
) -> None:
    if version_flag:
        typer.echo(f"agricola {_get_version()}")
        raise typer.Exit()


@app.command()
def step1(
    plink: Optional[list[str]] = typer.Option(
        None,
        help=(
            "Plink2 file prefix. "
            "This option can be repeated to specify multiple files. "
            "This option OR --plink-list must be provided (not both). "
            "Example: --plink-prefix chr1 --plink-prefix chr2"
        ),
    ),
    plink_list: Optional[str] = typer.Option(
        None,
        help=(
            "File containing plink2 prefixes, one per line. "
            "This option OR --plink-prefix must be provided (not both). "
        ),
    ),
    lanc: Optional[list[str]] = typer.Option(
        None,
        help=(
            "Local ancestry .lanc file. "
            "This option can be repeated to specify multiple files. "
            "This option OR --lanc-list must be provided (not both). "
            "Example: --lanc-file chr1.lanc --plink-prefix chr2.lanc"
        ),
    ),
    lanc_list: Optional[str] = typer.Option(
        None,
        help=(
            "File containing .lanc file paths, one per line. "
            "This option OR --lanc-file must be provided (not both). "
        ),
    ),
    level0_dir: Optional[str] = typer.Option(
        None,
        help=("Directory to save level 0 files to"),
    ),
    ancestries: Optional[str] = typer.Option(
        None, help="Ordered ancestry names, comma-separated"
    ),
    output: str = typer.Option(
        ...,
        help="Output prefix. Step 1 predictions will be serialized and written to {output}.pkl",
    ),
    pheno_file: str = typer.Option(..., help="Phenotype file"),
    pheno: Optional[list[str]] = typer.Option(
        None, help="Phenotype to include in analysis"
    ),
    pheno_list: Optional[str] = typer.Option(
        None, help="File containing phenotypes to include in analysis"
    ),
    covar_file: Optional[str] = typer.Option(None, help="Covariates file"),
    covar: Optional[list[str]] = typer.Option(
        None, help="Covariate to include in analysis"
    ),
    covar_list: Optional[str] = typer.Option(
        None, help="File containing covariates to include in analysis"
    ),
    catcovar: Optional[list[str]] = typer.Option(
        None, help="Categorical covariate to include in analysis"
    ),
    catcovar_list: Optional[str] = typer.Option(
        None, help="File containing categorical covariates to include in analysis"
    ),
    samples_file: Optional[str] = typer.Option(None, help="Samples file"),
    variant_file: Optional[str] = typer.Option(
        None, help="File with variants to include, one per line"
    ),
    h2_prior: str = typer.Option(
        DEFAULT_H2_PRIORS, help="SNP heritability priors, comma-separated"
    ),
    block_size: int = typer.Option(1000, help="Number of variants per block"),
    seed: int = typer.Option(100, help="Random seed"),
    trait_type: str = typer.Option(
        "qt", help="Trait type: quantitative (qt) or binary (bt)"
    ),
    loocv: bool = typer.Option(
        False, help="Use leave-one-out cross-validation (only for rare binary traits)"
    ),
    double_precision: bool = typer.Option(
        False,
        help=(
            "Force double precision (default single precision) in JAX. "
            "This option can be ignored if the environment variable JAX_ENABLE_X64=True is already set."
        ),
    ),
    prune_blocks: bool = typer.Option(
        True,
        help=(
            "Whether to sample variants in a dataset in level 0 so that n_variants (mod B) = 0. "
            "This will improve speed with JIT compilations."
        ),
    ),
    log: Optional[str] = typer.Option(
        None,
        help=("Log file"),
    ),
    verbose: bool = typer.Option(
        False,
        help=("Log file"),
    ),
) -> None:
    _setup_logging(log, verbose)
    logger.debug(_get_options_msg(locals()))

    import jax

    if double_precision:
        jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    from ._internal.utils import get_cv_mask
    from .step1 import step1
    import numpy as np

    ## Load data
    ancestries_list = _list_from_csv(ancestries)
    datasets, plinks, _ = _load_lanc_data(
        plink, plink_list, lanc, lanc_list, ancestries_list
    )
    variants = _load_variants(variant_file)
    h2_prior_arr = jnp.asarray([float(x) for x in h2_prior.split(",")])
    samples, samples_psam = _load_samples(plinks, samples_file)
    Y, X, phenotypes, samples = _load_pheno_and_covars(
        pheno_file,
        covar_file,
        pheno,
        pheno_list,
        covar,
        covar_list,
        catcovar,
        catcovar_list,
        samples,
    )
    idx_sample = np.where(np.isin(samples_psam, samples))[0].astype(np.uint32)
    logger.info("Datasets loaded")

    ## Get train/test split
    key = jax.random.PRNGKey(seed)
    train_mask, test_mask = get_cv_mask(len(Y), 5, key)

    ## Run step 1
    predictions = step1(
        datasets,
        Y,
        X,
        phenotypes,
        train_mask,
        test_mask,
        h2_prior_arr,
        trait_type,
        loocv,
        block_size,
        idx_sample,
        variants,
        level0_dir,
        prune_blocks,
    )

    ## Write predictions
    with open(output + ".pkl", "wb") as f:
        pickle.dump(predictions, f)


@app.command()
def step2(
    plink: Optional[list[str]] = typer.Option(
        None,
        help=(
            "Plink2 file prefix. "
            "This option can be repeated to specify multiple files. "
            "This option OR --plink-list must be provided (not both). "
            "Example: --plink-prefix chr1 --plink-prefix chr2"
        ),
    ),
    plink_list: Optional[str] = typer.Option(
        None,
        help=(
            "File containing plink2 prefixes, one per line. "
            "This option OR --plink-prefix must be provided (not both). "
        ),
    ),
    lanc: Optional[list[str]] = typer.Option(
        None,
        help=(
            "Local ancestry .lanc file. "
            "This option can be repeated to specify multiple files. "
            "This option OR --lanc-list must be provided (not both). "
            "Example: --lanc-file chr1.lanc --plink-prefix chr2.lanc"
        ),
    ),
    lanc_list: Optional[str] = typer.Option(
        None,
        help=(
            "File containing .lanc file paths, one per line. "
            "This option OR --lanc-file must be provided (not both). "
        ),
    ),
    ancestries: Optional[str] = typer.Option(
        None, help="Ordered ancestry names, comma-separated"
    ),
    step1_prefix: str = typer.Option(
        ..., help="Step 1 predictions are deserialized from prefix.pkl"
    ),
    outdir: str = typer.Option(
        None,
        help=(
            "Output directory. If --no-overwrite, this directory must not exist. Chunked parquet files are written to e.g. outdir/part-0001.parquet"
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        help="Whether to overwrite outdir. If true, any existing folders and files in outdir will be deleted.",
    ),
    pheno_file: str = typer.Option(..., help="Phenotype file"),
    pheno: Optional[list[str]] = typer.Option(
        None, help="Phenotype to include in analysis"
    ),
    pheno_list: Optional[str] = typer.Option(
        None, help="File containing phenotypes to include in analysis"
    ),
    covar_file: Optional[str] = typer.Option(None, help="Covariates file"),
    covar: Optional[list[str]] = typer.Option(
        None, help="Covariate to include in analysis"
    ),
    covar_list: Optional[str] = typer.Option(
        None, help="File containing covariates to include in analysis"
    ),
    catcovar: Optional[list[str]] = typer.Option(
        None, help="Categorical covariate to include in analysis"
    ),
    catcovar_list: Optional[str] = typer.Option(
        None, help="File containing categorical covariates to include in analysis"
    ),
    samples_file: Optional[str] = typer.Option(None, help="Samples file"),
    variant_file: Optional[str] = typer.Option(
        None, help="File with variants to include, one per line"
    ),
    chrom: Optional[str] = typer.Option(None, help="Chromosome"),
    block_size: int = typer.Option(1000, help="Number of variants per block"),
    min_ac: int = typer.Option(1, help="Minimum allele count"),
    trait_type: str = typer.Option(
        "qt", help="Trait type: quantitative (qt) or binary (bt)"
    ),
    test_type: str = typer.Option("score", help="Test type: score or wald"),
    adjust_lanc: bool = typer.Option(
        True, help="Adjust single variant tests for local ancestry"
    ),
    impute: bool = typer.Option(
        False,
        help="Impute quantitative traits in step 2 (must be --no-impute for binary traits)",
    ),
    double_precision: bool = typer.Option(
        False,
        help=(
            "Force double precision (default single precision) in JAX. "
            "This option can be ignored if the environment variable JAX_ENABLE_X64=True is already set."
        ),
    ),
    log: Optional[str] = typer.Option(
        None,
        help=("Log file"),
    ),
    verbose: bool = typer.Option(
        False,
        help=("Log file"),
    ),
) -> None:
    _setup_logging(log, verbose)
    logger.debug(_get_options_msg(locals()))

    import jax

    if double_precision:
        jax.config.update("jax_enable_x64", True)

    from .step2 import step2
    import numpy as np

    ## Load data
    ancestries_list = _list_from_csv(ancestries)
    datasets, plinks, _ = _load_lanc_data(
        plink, plink_list, lanc, lanc_list, ancestries_list
    )

    variants = _load_variants(variant_file)
    samples, samples_psam = _load_samples(plinks, samples_file)

    Y, X, phenotypes, samples = _load_pheno_and_covars(
        pheno_file,
        covar_file,
        pheno,
        pheno_list,
        covar,
        covar_list,
        catcovar,
        catcovar_list,
        samples,
    )
    idx_sample = np.where(np.isin(samples_psam, samples))[0].astype(np.uint32)

    ## Load step1 predictions
    with open(step1_prefix + ".pkl", "rb") as file:
        predictions = pickle.load(file)

    ## Run step 2
    step2(
        datasets,
        Y,
        X,
        predictions,
        outdir,
        phenotypes,
        trait_type,
        test_type,
        chrom,
        block_size,
        min_ac,
        idx_sample,
        variants,
        adjust_lanc,
        impute,
        overwrite,
    )


@app.command()
def all_steps(
    plink: Optional[list[str]] = typer.Option(
        None,
        help=(
            "Plink2 file prefix. "
            "This option can be repeated to specify multiple files. "
            "This option OR --plink-list must be provided (not both). "
            "Example: --plink-prefix chr1 --plink-prefix chr2"
        ),
    ),
    plink_list: Optional[str] = typer.Option(
        None,
        help=(
            "File containing plink2 prefixes, one per line. "
            "This option OR --plink-prefix must be provided (not both). "
        ),
    ),
    lanc: Optional[list[str]] = typer.Option(
        None,
        help=(
            "Local ancestry .lanc file. "
            "This option can be repeated to specify multiple files. "
            "This option OR --lanc-list must be provided (not both). "
            "Example: --lanc-file chr1.lanc --plink-prefix chr2.lanc"
        ),
    ),
    lanc_list: Optional[str] = typer.Option(
        None,
        help=(
            "File containing .lanc file paths, one per line. "
            "This option OR --lanc-file must be provided (not both). "
        ),
    ),
    level0_dir: Optional[str] = typer.Option(
        None,
        help=("Directory to save level 0 files to"),
    ),
    ancestries: Optional[str] = typer.Option(
        None, help="Ordered ancestry names, comma-separated"
    ),
    pheno_file: str = typer.Option(..., help="Phenotype file"),
    pheno: Optional[list[str]] = typer.Option(
        None, help="Phenotype to include in analysis"
    ),
    pheno_list: Optional[str] = typer.Option(
        None, help="File containing phenotypes to include in analysis"
    ),
    covar_file: Optional[str] = typer.Option(None, help="Covariates file"),
    covar: Optional[list[str]] = typer.Option(
        None, help="Covariate to include in analysis"
    ),
    covar_list: Optional[str] = typer.Option(
        None, help="File containing covariates to include in analysis"
    ),
    catcovar: Optional[list[str]] = typer.Option(
        None, help="Categorical covariate to include in analysis"
    ),
    catcovar_list: Optional[str] = typer.Option(
        None, help="File containing categorical covariates to include in analysis"
    ),
    outdir: str = typer.Option(
        None,
        help=(
            "Output directory. If --no-overwrite, this directory must not exist. Chunked parquet files are written to e.g. outdir/part-0001.parquet"
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        help="Whether to overwrite outdir. If true, any existing folders and files in outdir will be deleted.",
    ),
    samples_file: Optional[str] = typer.Option(None, help="Samples file"),
    variant_file1: Optional[str] = typer.Option(
        None, help="File with variants to include in step 0/1, one per line"
    ),
    variant_file2: Optional[str] = typer.Option(
        None, help="File with variants to include in step 2, one per line"
    ),
    chrom: Optional[str] = typer.Option(None, help="Chromosome"),
    h2_prior: str = typer.Option(
        DEFAULT_H2_PRIORS, help="SNP heritability priors, comma-separated"
    ),
    block_size1: int = typer.Option(
        1000, help="Number of variants per block in step 1"
    ),
    block_size2: int = typer.Option(500, help="Number of variants per block in step 2"),
    min_ac: int = typer.Option(1, help="Minimum allele count"),
    seed: int = typer.Option(100, help="Random seed"),
    trait_type: str = typer.Option(
        "qt", help="Trait type: quantitative (qt) or binary (bt)"
    ),
    test_type: str = typer.Option("score", help="Test type: score or wald"),
    loocv: bool = typer.Option(
        False, help="Use leave-one-out cross-validation (only for rare binary traits)"
    ),
    adjust_lanc: bool = typer.Option(
        True, help="Adjust single variant tests for local ancestry"
    ),
    impute: bool = typer.Option(
        False,
        help="Impute quantitative traits in step 2 (must be --no-impute for binary traits)",
    ),
    double_precision: bool = typer.Option(
        False,
        help=(
            "Force double precision (default single precision) in JAX. "
            "This option can be ignored if the environment variable JAX_ENABLE_X64=True is already set."
        ),
    ),
    prune_blocks: bool = typer.Option(
        True,
        help=(
            "Whether to sample variants in a dataset in level 0 so that n_variants (mod B) = 0. "
            "This will improve speed with JIT compilations."
        ),
    ),
    log: Optional[str] = typer.Option(
        None,
        help=("Log file"),
    ),
    verbose: bool = typer.Option(
        False,
        help=("Log file"),
    ),
) -> None:
    _setup_logging(log, verbose)
    logger.debug(_get_options_msg(locals()))

    import jax

    if double_precision:
        jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import numpy as np
    from ._internal.utils import get_cv_mask
    from .step1 import step1
    from .step2 import step2

    ## Catch bad impute early
    if trait_type == "bt" and impute:
        raise typer.BadParameter("Binary traits must use --no-impute")

    ## Load data
    ancestries_list = _list_from_csv(ancestries)
    datasets, plinks, _ = _load_lanc_data(
        plink, plink_list, lanc, lanc_list, ancestries_list
    )
    variants1 = _load_variants(variant_file1)
    variants2 = _load_variants(variant_file2)
    h2_prior_arr = jnp.asarray([float(x) for x in h2_prior.split(",")])

    samples, samples_psam = _load_samples(plinks, samples_file)
    Y, X, phenotypes, samples = _load_pheno_and_covars(
        pheno_file,
        covar_file,
        pheno,
        pheno_list,
        covar,
        covar_list,
        catcovar,
        catcovar_list,
        samples,
    )
    idx_sample = np.where(np.isin(samples_psam, samples))[0].astype(np.uint32)

    ## Get train/test split
    key = jax.random.PRNGKey(seed)
    train_mask, test_mask = get_cv_mask(len(Y), 5, key)

    ## Run step 1
    predictions = step1(
        datasets,
        Y,
        X,
        phenotypes,
        train_mask,
        test_mask,
        h2_prior_arr,
        trait_type,
        loocv,
        block_size1,
        idx_sample,
        variants1,
        level0_dir,
        prune_blocks,
    )

    step2(
        datasets,
        Y,
        X,
        predictions,
        outdir,
        phenotypes,
        trait_type,
        test_type,
        chrom,
        block_size2,
        min_ac,
        idx_sample,
        variants2,
        adjust_lanc,
        impute,
        overwrite,
    )


def main_entry() -> None:
    try:
        app()
    except Exception as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    main_entry()
