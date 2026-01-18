# MIT License
# Copyright (c) 2025 Franklin Ockerman
# See LICENSE.txt file for full license text

import logging
import pickle
import typer
from typing import Optional, List
import os
from importlib.metadata import version, PackageNotFoundError


DEFAULT_H2_PRIORS = "0.01,0.255,0.5,0.745,0.99"

app = typer.Typer(help="lagga CLI")
logger = logging.getLogger("lagga")


def _configure_jax_logging():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_LOGGING_MIN_LEVEL", "3")


def list_from_csv(arg: Optional[str]) -> Optional[List[str]]:
    return None if arg is None else [x.strip() for x in arg.split(",")]


def load_variants(path: Optional[str]) -> Optional[List[str]]:
    return None if path is None else open(path).read().splitlines()


def read_psam(path):
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


def read_pheno_covar(path):
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


def load_pheno_and_covars(
    pheno_file: str, covar_file: Optional[str], samples: List[str]
):
    import pandas as pd
    import jax.numpy as jnp

    df_pheno = read_pheno_covar(pheno_file)
    samples_pheno = df_pheno["IID"].astype(str).to_list()

    samples = [sample for sample in samples if sample in samples_pheno]

    if covar_file:
        df_covar = read_pheno_covar(covar_file)
        samples_covar = df_covar["IID"].astype(str).to_list()
        samples = [sample for sample in samples if sample in samples_covar]
        df_covar = df_covar[df_covar["IID"].astype(str).isin(samples)]
        df_covar["IID"] = df_covar["IID"].astype(str)
        df_covar["IID"] = pd.Categorical(
            df_covar["IID"], categories=samples, ordered=True
        )
        df_covar = df_covar.sort_values("IID").reset_index(drop=True)
        X = jnp.asarray(
            df_covar.drop("IID", axis=1).drop("FID", axis=1, errors="ignore").to_numpy()
        )
    else:
        X = None

    df_pheno = df_pheno[df_pheno["IID"].astype(str).isin(samples)]
    df_pheno["IID"] = df_pheno["IID"].astype(str)
    df_pheno["IID"] = pd.Categorical(df_pheno["IID"], categories=samples, ordered=True)
    df_pheno = df_pheno.sort_values("IID").reset_index(drop=True)
    Y = jnp.asarray(
        df_pheno.drop("IID", axis=1).drop("FID", axis=1, errors="ignore").to_numpy()
    )

    return (
        Y,
        X,
        df_pheno.drop("IID", axis=1)
        .drop("FID", axis=1, errors="ignore")
        .columns.to_list(),
        samples,
    )


def load_lanc_data(plinks, lancs, ancestries):
    from lanctools import LancData

    return [
        LancData(plink_prefix=plinks[i], lanc_file=lancs[i], ancestries=ancestries)
        for i in range(len(plinks))
    ]


def setup_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def get_version() -> str:
    try:
        return version("lagga")
    except PackageNotFoundError:
        return "unknown"


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose"),
    quiet: bool = typer.Option(False, "--quiet"),
):
    setup_logging(verbose, quiet)


@app.command()
def step1(
    plink_prefix: str = typer.Option(
        ..., help="Plink2 file prefix(es), comma-separated"
    ),
    lanc_file: str = typer.Option(
        ..., help="Local ancestry .lanc file(s), comma-separated"
    ),
    ancestries: Optional[str] = typer.Option(
        None, help="Ordered ancestry names, comma-separated"
    ),
    out_prefix: str = typer.Option(
        ..., help="Step 1 predictions will be serialized and written to prefix.pkl"
    ),
    pheno_file: str = typer.Option(..., help="Phenotype file"),
    covar_file: Optional[str] = typer.Option(None, help="Covariates file"),
    samples_file: Optional[str] = typer.Option(None, help="Samples file"),
    variant_file: Optional[str] = typer.Option(
        None, help="File with variants to include, one per line"
    ),
    h2_prior: str = typer.Option(
        DEFAULT_H2_PRIORS, help="SNP heritability priors, comma-separated"
    ),
    block_size: int = typer.Option(2000, help="Number of variants per block"),
    seed: int = typer.Option(100, help="Random seed"),
    trait_type: str = typer.Option(
        "qt", help="Trait type: quantitative (qt) or binary (bt)"
    ),
    loocv: bool = typer.Option(
        False, help="Use leave-one-out cross-validation (only for rare binary traits)"
    ),
):
    import jax.numpy as jnp
    import jax
    from ._utils import get_cv_mask
    from .step0 import step0
    from .step1 import step1_qt, step1_bt
    import pandas as pd
    import numpy as np

    plinks = list_from_csv(plink_prefix)
    lancs = list_from_csv(lanc_file)
    ancestries_list = list_from_csv(ancestries)
    variants = load_variants(variant_file)
    h2_prior_arr = jnp.asarray([float(x) for x in h2_prior.split(",")])

    ## Load data
    datasets = load_lanc_data(plinks, lancs, ancestries_list)
    df_psam = read_psam(plinks[0] + ".psam")
    samples_psam = df_psam["IID"].astype(str).to_list()

    if samples_file is not None:
        with open(samples_file, "r") as f:
            samples_keep = f.readlines()
        samples = [sample for sample in samples_psam if sample in samples_keep]
    else:
        samples = samples_psam

    Y, X, _, samples = load_pheno_and_covars(pheno_file, covar_file, samples)
    idx_sample = np.where(np.isin(samples_psam, samples))[0].astype(np.uint32)

    ## Get train/test split
    key = jax.random.PRNGKey(seed)
    train_mask, test_mask = get_cv_mask(len(Y), 5, key)

    ## Run steps 0 and 1
    Z = step0(
        datasets,
        Y,
        X,
        train_mask,
        test_mask,
        h2_prior_arr,
        block_size,
        idx_sample,
        variants,
    )

    if trait_type == "qt":
        predictions = step1_qt(Z, Y, X, train_mask, test_mask, h2_prior_arr)
    else:
        predictions = step1_bt(Z, Y, X, loocv, train_mask, test_mask, h2_prior_arr)

    ## Write predictions
    with open(out_prefix + ".pkl", "wb") as f:
        pickle.dump(predictions, f)


# ---------------------------------------------------------
# step2
# ---------------------------------------------------------


@app.command()
def step2(
    plink_prefix: str = typer.Option(
        ..., help="Plink2 file prefix(es), comma-separated"
    ),
    lanc_file: str = typer.Option(
        ..., help="Local ancestry .lanc file(s), comma-separated"
    ),
    ancestries: Optional[str] = typer.Option(
        None, help="Ordered ancestry names, comma-separated"
    ),
    step1_prefix: str = typer.Option(
        ..., help="Step 1 predictions are deserialized from prefix.pkl"
    ),
    out_prefix: str = typer.Option(
        ...,
        help="Output prefix(es), comma-separated, one per plink_prefix",
    ),
    pheno_file: str = typer.Option(..., help="Phenotype file"),
    covar_file: Optional[str] = typer.Option(None, help="Covariates file"),
    samples_file: Optional[str] = typer.Option(None, help="Samples file"),
    variant_file: Optional[str] = typer.Option(
        None, help="File with variants to include, one per line"
    ),
    block_size: int = typer.Option(1000, help="Number of variants per block"),
    trait_type: str = typer.Option(
        "qt", help="Trait type: quantitative (qt) or binary (bt)"
    ),
):
    from .step2 import step2
    import numpy as np

    plinks = list_from_csv(plink_prefix)
    lancs = list_from_csv(lanc_file)
    ancestries_list = list_from_csv(ancestries)
    out_prefixes = list_from_csv(out_prefix)
    variants = load_variants(variant_file)

    ## Load data
    datasets = load_lanc_data(plinks, lancs, ancestries_list)
    df_psam = read_psam(plinks[0] + ".psam")
    samples_psam = df_psam["IID"].astype(str).to_list()

    if samples_file is not None:
        with open(samples_file, "r") as f:
            samples_keep = f.readlines()
        samples = [sample for sample in samples_psam if sample in samples_keep]
    else:
        samples = samples_psam

    Y, X, pheno_names, samples = load_pheno_and_covars(pheno_file, covar_file, samples)
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
        out_prefixes,
        pheno_names,
        trait_type,
        block_size,
        idx_sample,
        variants,
    )


# ---------------------------------------------------------
# all_steps
# ---------------------------------------------------------


@app.command()
def all_steps(
    plink_prefix: str = typer.Option(
        ..., help="Plink2 file prefix(es), comma-separated"
    ),
    lanc_file: str = typer.Option(
        ..., help="Local ancestry .lanc file(s), comma-separated"
    ),
    ancestries: Optional[str] = typer.Option(
        None, help="Ordered ancestry names, comma-separated"
    ),
    pheno_file: str = typer.Option(..., help="Phenotype file"),
    covar_file: Optional[str] = typer.Option(None, help="Covariates file"),
    out_prefix: str = typer.Option(
        ..., help="Output prefix(es), comma-separated, one per plink_prefix"
    ),
    samples_file: Optional[str] = typer.Option(None, help="Samples file"),
    variant_file1: Optional[str] = typer.Option(
        None, help="File with variants to include in step 0/1, one per line"
    ),
    variant_file2: Optional[str] = typer.Option(
        None, help="File with variants to include in step 2, one per line"
    ),
    h2_prior: str = typer.Option(
        DEFAULT_H2_PRIORS, help="SNP heritability priors, comma-separated"
    ),
    block_size0: int = typer.Option(
        2000, help="Number of variants per block in step 0"
    ),
    block_size2: int = typer.Option(
        1000, help="Number of variants per block in step 2"
    ),
    seed: int = typer.Option(100, help="Random seed"),
    trait_type: str = typer.Option(
        "qt", help="Trait type: quantitative (qt) or binary (bt)"
    ),
    loocv: bool = typer.Option(
        False, help="Use leave-one-out cross-validation (only for rare binary traits)"
    ),
):
    import jax.numpy as jnp
    import jax
    import numpy as np
    from ._utils import get_cv_mask
    from .step0 import step0
    from .step1 import step1_qt, step1_bt
    from .step2 import step2

    plinks = list_from_csv(plink_prefix)
    lancs = list_from_csv(lanc_file)
    ancestries_list = list_from_csv(ancestries)
    variants1 = load_variants(variant_file1)
    variants2 = load_variants(variant_file2)
    h2_prior_arr = jnp.asarray([float(x) for x in h2_prior.split(",")])
    out_prefixes = list_from_csv(out_prefix)

    ## Load data
    datasets = load_lanc_data(plinks, lancs, ancestries_list)
    df_psam = read_psam(plinks[0] + ".psam")
    samples_psam = df_psam["IID"].astype(str).to_list()

    if samples_file is not None:
        with open(samples_file, "r") as f:
            samples_keep = f.readlines()
        samples = [sample for sample in samples_psam if sample in samples_keep]
    else:
        samples = samples_psam

    Y, X, pheno_names, samples = load_pheno_and_covars(pheno_file, covar_file, samples)
    idx_sample = np.where(np.isin(samples_psam, samples))[0].astype(np.uint32)

    ## Get train/test split
    key = jax.random.PRNGKey(seed)
    train_mask, test_mask = get_cv_mask(len(Y), 5, key)

    ## Run step 0
    Z = step0(
        datasets,
        Y,
        X,
        train_mask,
        test_mask,
        h2_prior_arr,
        block_size0,
        idx_sample,
        variants1,
    )

    ## Run steps 1 and 2
    if trait_type == "qt":
        predictions = step1_qt(Z, Y, X, train_mask, test_mask, h2_prior_arr)
    else:
        predictions = step1_bt(Z, Y, X, loocv, train_mask, test_mask, h2_prior_arr)

    step2(
        datasets,
        Y,
        X,
        predictions,
        out_prefixes,
        pheno_names,
        trait_type,
        block_size2,
        idx_sample,
        variants2,
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------


def main_entry():
    _configure_jax_logging()
    try:
        app()
    except Exception as exc:
        logger.debug("Unhandled exception", exc_info=True)
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    main_entry()
