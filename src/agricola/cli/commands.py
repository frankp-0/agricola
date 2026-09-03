# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""The command line interface for agricola."""

from __future__ import annotations

import pickle
from enum import Enum

import typer

from .data import (
    load_lanc_data,
    load_pgen_data,
    load_pheno_and_covars,
    load_samples,
    load_variants,
)
from .formatting import get_options_msg, list_from_csv
from .runtime import (
    get_version,
    logger,
    print_welcome,
    report_devices,
    setup_logging,
)

DEFAULT_H2_PRIORS = "0.01,0.255,0.5,0.745,0.99"

app = typer.Typer(help="agricola CLI")


class MemoryMode(str, Enum):
    STANDARD = "standard"
    LOW = "low"
    LOWEST = "lowest"


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
        typer.echo(f"agricola {get_version()}")
        raise typer.Exit()
    print_welcome()


@app.command()
def step1(
    plink: list[str] | None = typer.Option(
        None,
        help=(
            "Plink2 file prefix. "
            "This option can be repeated to specify multiple files. "
            "This option OR --plink-list must be provided (not both). "
            "Example: --plink-prefix chr1 --plink-prefix chr2"
        ),
    ),
    plink_list: str | None = typer.Option(
        None,
        help=(
            "File containing plink2 prefixes, one per line. "
            "This option OR --plink-prefix must be provided (not both). "
        ),
    ),
    level0_dir: str | None = typer.Option(
        None,
        help=("Directory to save level 0 files to"),
    ),
    output: str = typer.Option(
        ...,
        help="Output prefix. Step 1 predictions will be serialized and written to {output}.pkl",
    ),
    pheno_file: str = typer.Option(..., help="Phenotype file"),
    pheno: list[str] | None = typer.Option(None, help="Phenotype to include in analysis"),
    pheno_list: str | None = typer.Option(
        None, help="File containing phenotypes to include in analysis"
    ),
    covar_file: str | None = typer.Option(None, help="Covariates file"),
    covar: list[str] | None = typer.Option(None, help="Covariate to include in analysis"),
    covar_list: str | None = typer.Option(
        None, help="File containing covariates to include in analysis"
    ),
    catcovar: list[str] | None = typer.Option(
        None, help="Categorical covariate to include in analysis"
    ),
    catcovar_list: str | None = typer.Option(
        None, help="File containing categorical covariates to include in analysis"
    ),
    samples_file: str | None = typer.Option(None, help="Samples file"),
    variant_file: str | None = typer.Option(
        None, help="File with variants to include, one per line"
    ),
    h2_prior: str = typer.Option(
        DEFAULT_H2_PRIORS, help="SNP heritability priors, comma-separated"
    ),
    block_size: int = typer.Option(1000, help="Number of variants per block"),
    seed: int = typer.Option(100, help="Random seed"),
    trait_type: str = typer.Option("qt", help="Trait type: quantitative (qt) or binary (bt)"),
    loocv: bool = typer.Option(
        False, help="Use leave-one-out cross-validation (only for rare binary traits)"
    ),
    double_precision: bool = typer.Option(
        False,
        help=(
            "Force double precision (default single precision) in JAX. This option "
            "can be ignored if the environment variable JAX_ENABLE_X64=True is already set."
        ),
    ),
    prune_blocks: bool = typer.Option(
        True,
        help=(
            "Whether to sample variants in a dataset in level 0 so that n_variants (mod B) = 0. "
            "This will improve speed with JIT compilations."
        ),
    ),
    backend: str | None = typer.Option(
        None,
        help=(
            "Jax backend to use (e.g. --backend cpu or --backend cuda. Jax automatically "
            "detects the correct backend, but this can be specified to e.g. use "
            "cpu instead of cuda devices. "
        ),
    ),
    log: str | None = typer.Option(
        None,
        help=("Log file"),
    ),
    memory_mode: MemoryMode = typer.Option(
        MemoryMode.STANDARD,
        help="Ridge memory strategy: standard, low, or lowest.",
    ),
    verbose: bool = typer.Option(False),
) -> None:
    setup_logging(log, verbose)
    report_devices(backend)
    logger.debug(get_options_msg(locals()))

    import jax

    if double_precision:
        jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import numpy as np

    from ..pipeline.cross_validation import get_cv_mask
    from ..pipeline.step1 import step1

    ## Load data
    datasets, plinks = load_pgen_data(plink, plink_list)
    variants = load_variants(variant_file)
    h2_prior_arr = jnp.asarray([float(x) for x in h2_prior.split(",")])
    samples, samples_psam = load_samples(plinks, samples_file)
    Y, X, phenotypes, samples = load_pheno_and_covars(
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
    key_cv, key_step1 = jax.random.split(jax.random.key(seed))
    train_mask, test_mask = get_cv_mask(len(Y), 5, key_cv)

    ## Run step 1
    step1_predictions = step1(
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
        key_step1,
        memory_mode.value,
    )

    for _, i in enumerate(step1_predictions):
        step1_predictions[i].index = samples

    ## Write predictions
    with open(output + ".pkl", "wb") as f:
        pickle.dump(step1_predictions, f)


@app.command()
def step2(
    plink: list[str] | None = typer.Option(
        None,
        help=(
            "Plink2 file prefix. "
            "This option can be repeated to specify multiple files. "
            "This option OR --plink-list must be provided (not both). "
            "Example: --plink-prefix chr1 --plink-prefix chr2"
        ),
    ),
    plink_list: str | None = typer.Option(
        None,
        help=(
            "File containing plink2 prefixes, one per line. "
            "This option OR --plink-prefix must be provided (not both). "
        ),
    ),
    lanc: list[str] | None = typer.Option(
        None,
        help=(
            "Local ancestry .lanc file. "
            "This option can be repeated to specify multiple files. "
            "This option OR --lanc-list must be provided (not both). "
            "Example: --lanc-file chr1.lanc --plink-prefix chr2.lanc"
        ),
    ),
    lanc_list: str | None = typer.Option(
        None,
        help=(
            "File containing .lanc file paths, one per line. "
            "This option OR --lanc-file must be provided (not both). "
        ),
    ),
    ancestries: str | None = typer.Option(None, help="Ordered ancestry names, comma-separated"),
    step1_prefix: str | None = typer.Option(
        None,
        help=(
            "Step 1 predictions are deserialized from prefix.pkl. "
            "If not provided, agricola does not condition on whole-genome predictions. "
        ),
    ),
    outdir: str = typer.Option(
        None,
        help=("Output directory. If --no-overwrite, this directory must not exist. "),
    ),
    overwrite: bool = typer.Option(
        False,
        help=(
            "Whether to overwrite outdir. If true, any existing folders and "
            "files in outdir will be deleted."
        ),
    ),
    pheno_file: str = typer.Option(..., help="Phenotype file"),
    pheno: list[str] | None = typer.Option(None, help="Phenotype to include in analysis"),
    pheno_list: str | None = typer.Option(
        None, help="File containing phenotypes to include in analysis"
    ),
    covar_file: str | None = typer.Option(None, help="Covariates file"),
    covar: list[str] | None = typer.Option(None, help="Covariate to include in analysis"),
    covar_list: str | None = typer.Option(
        None, help="File containing covariates to include in analysis"
    ),
    catcovar: list[str] | None = typer.Option(
        None, help="Categorical covariate to include in analysis"
    ),
    catcovar_list: str | None = typer.Option(
        None, help="File containing categorical covariates to include in analysis"
    ),
    samples_file: str | None = typer.Option(None, help="Samples file"),
    variant_file: str | None = typer.Option(
        None, help="File with variants to include, one per line"
    ),
    chrom: str | None = typer.Option(None, help="Chromosome"),
    block_size: int = typer.Option(1000, help="Number of variants per block"),
    min_ac: int = typer.Option(1, help="Minimum allele count"),
    trait_type: str = typer.Option("qt", help="Trait type: quantitative (qt) or binary (bt)"),
    test_type: str = typer.Option("score", help="Test type: score or wald"),
    adjust_lanc: bool = typer.Option(True, help="Adjust single variant tests for local ancestry"),
    impute: bool = typer.Option(
        False,
        help="Impute quantitative traits in step 2 (must be --no-impute for binary traits)",
    ),
    double_precision: bool = typer.Option(
        False,
        help=(
            "Force double precision (default single precision) in JAX. This option "
            "can be ignored if the environment variable JAX_ENABLE_X64=True is already set."
        ),
    ),
    partition_phenotypes: bool = typer.Option(
        True,
        help=(
            "Whether to partition output parquet files in step 2 by phenotype . "
            "If True, output files are written to e.g. outdir/trait0/part-0_0.parquet "
            "With a large number of phenotypes, this can lead to very small .parquet "
            "files unless --max-rows is increased"
        ),
    ),
    max_rows: int = typer.Option(
        None,
        help=(
            "Max number of rows/variants per phenotype to keep in memory before "
            "writing an output file. If unspecified, agricola will use "
            "5000000 / len(phenotypes)"
        ),
    ),
    backend: str | None = typer.Option(
        None,
        help=(
            "Jax backend to use (e.g. --backend cpu or --backend cuda. "
            "Jax automatically detects the correct backend, but this can be specified "
            "to e.g. use cpu instead of cuda devices. "
        ),
    ),
    log: str | None = typer.Option(
        None,
        help=("Log file"),
    ),
    verbose: bool = typer.Option(False),
) -> None:
    setup_logging(log, verbose)
    report_devices(backend)
    logger.debug(get_options_msg(locals()))

    import jax

    if double_precision:
        jax.config.update("jax_enable_x64", True)

    import numpy as np

    from ..pipeline.step2 import step2

    ## Load data
    ancestries_list = list_from_csv(ancestries)
    lanc_datasets, plinks, _ = load_lanc_data(plink, plink_list, lanc, lanc_list, ancestries_list)

    variants = load_variants(variant_file)
    samples, samples_psam = load_samples(plinks, samples_file)

    Y, X, phenotypes, samples = load_pheno_and_covars(
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
    if step1_prefix is not None:
        with open(step1_prefix + ".pkl", "rb") as file:
            step1_predictions = pickle.load(file)
        for _, i in enumerate(step1_predictions):
            step1_predictions[i] = step1_predictions[i].loc[samples]
    else:
        step1_predictions = None
        logger.info(
            "--step1-prefix is not provided. Agricola will not condition on "
            "whole-genome regression.\n"
        )

    ## Run step 2
    step2(
        lanc_datasets,
        Y,
        X,
        step1_predictions,
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
        partition_phenotypes,
        max_rows,
    )


@app.command()
def all_steps(
    plink: list[str] | None = typer.Option(
        None,
        help=(
            "Plink2 file prefix. "
            "This option can be repeated to specify multiple files. "
            "This option OR --plink-list must be provided (not both). "
            "Example: --plink-prefix chr1 --plink-prefix chr2"
        ),
    ),
    plink_list: str | None = typer.Option(
        None,
        help=(
            "File containing plink2 prefixes, one per line. "
            "This option OR --plink-prefix must be provided (not both). "
        ),
    ),
    lanc: list[str] | None = typer.Option(
        None,
        help=(
            "Local ancestry .lanc file. "
            "This option can be repeated to specify multiple files. "
            "This option OR --lanc-list must be provided (not both). "
            "Example: --lanc-file chr1.lanc --plink-prefix chr2.lanc"
        ),
    ),
    lanc_list: str | None = typer.Option(
        None,
        help=(
            "File containing .lanc file paths, one per line. "
            "This option OR --lanc-file must be provided (not both). "
        ),
    ),
    level0_dir: str | None = typer.Option(
        None,
        help=("Directory to save level 0 files to"),
    ),
    ancestries: str | None = typer.Option(None, help="Ordered ancestry names, comma-separated"),
    pheno_file: str = typer.Option(..., help="Phenotype file"),
    pheno: list[str] | None = typer.Option(None, help="Phenotype to include in analysis"),
    pheno_list: str | None = typer.Option(
        None, help="File containing phenotypes to include in analysis"
    ),
    covar_file: str | None = typer.Option(None, help="Covariates file"),
    covar: list[str] | None = typer.Option(None, help="Covariate to include in analysis"),
    covar_list: str | None = typer.Option(
        None, help="File containing covariates to include in analysis"
    ),
    catcovar: list[str] | None = typer.Option(
        None, help="Categorical covariate to include in analysis"
    ),
    catcovar_list: str | None = typer.Option(
        None, help="File containing categorical covariates to include in analysis"
    ),
    outdir: str = typer.Option(
        None,
        help=("Output directory. If --no-overwrite, this directory must not exist. "),
    ),
    overwrite: bool = typer.Option(
        False,
        help=(
            "Whether to overwrite outdir. If true, any existing folders and files "
            "in outdir will be deleted."
        ),
    ),
    samples_file: str | None = typer.Option(None, help="Samples file"),
    variant_file1: str | None = typer.Option(
        None, help="File with variants to include in step 0/1, one per line"
    ),
    variant_file2: str | None = typer.Option(
        None, help="File with variants to include in step 2, one per line"
    ),
    chrom: str | None = typer.Option(None, help="Chromosome"),
    h2_prior: str = typer.Option(
        DEFAULT_H2_PRIORS, help="SNP heritability priors, comma-separated"
    ),
    block_size1: int = typer.Option(1000, help="Number of variants per block in step 1"),
    block_size2: int = typer.Option(500, help="Number of variants per block in step 2"),
    min_ac: int = typer.Option(1, help="Minimum allele count"),
    seed: int = typer.Option(100, help="Random seed"),
    trait_type: str = typer.Option("qt", help="Trait type: quantitative (qt) or binary (bt)"),
    test_type: str = typer.Option("score", help="Test type: score or wald"),
    loocv: bool = typer.Option(
        False, help="Use leave-one-out cross-validation (only for rare binary traits)"
    ),
    adjust_lanc: bool = typer.Option(True, help="Adjust single variant tests for local ancestry"),
    impute: bool = typer.Option(
        False,
        help="Impute quantitative traits in step 2 (must be --no-impute for binary traits)",
    ),
    double_precision: bool = typer.Option(
        False,
        help=(
            "Force double precision (default single precision) in JAX. This option "
            "can be ignored if the environment variable JAX_ENABLE_X64=True is already set."
        ),
    ),
    prune_blocks: bool = typer.Option(
        True,
        help=(
            "Whether to sample variants in a dataset in level 0 so that n_variants (mod B) = 0. "
            "This will improve speed with JIT compilations."
        ),
    ),
    partition_phenotypes: bool = typer.Option(
        True,
        help=(
            "Whether to partition output parquet files in step 2 by phenotype. If "
            "True, output files are written to e.g. outdir/trait0/part-0_0.parquet, "
            "With a large number of phenotypes, this can lead to very small parquet "
            "files unless --max-rows is increased."
        ),
    ),
    max_rows: int = typer.Option(
        None,
        help=(
            "Max number of rows/variants per phenotype to keep in memory before writing "
            "an output file in step 2. If unspecified, agricola will use 5000000 / len(phenotypes)"
        ),
    ),
    backend: str | None = typer.Option(
        None,
        help=(
            "Jax backend to use (e.g. --backend cpu or --backend cuda. Jax automatically "
            "detects the correct backend, but this can be specified to e.g. use "
            "cpu instead of cuda devices."
        ),
    ),
    log: str | None = typer.Option(
        None,
        help=("Log file"),
    ),
    memory_mode: MemoryMode = typer.Option(
        MemoryMode.STANDARD,
        help="Ridge memory strategy: standard, low, or lowest.",
    ),
    verbose: bool = typer.Option(False),
) -> None:
    setup_logging(log, verbose)
    report_devices(backend)
    logger.debug(get_options_msg(locals()))

    import jax

    if double_precision:
        jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import numpy as np

    from ..pipeline.cross_validation import get_cv_mask
    from ..pipeline.step1 import step1
    from ..pipeline.step2 import step2

    ## Catch bad impute early
    if trait_type == "bt" and impute:
        raise typer.BadParameter("Binary traits must use --no-impute")

    ## Load data
    ancestries_list = list_from_csv(ancestries)
    datasets, plinks, _ = load_lanc_data(plink, plink_list, lanc, lanc_list, ancestries_list)
    pgen_datasets, _ = load_pgen_data(plink, plink_list)
    variants1 = load_variants(variant_file1)
    variants2 = load_variants(variant_file2)
    h2_prior_arr = jnp.asarray([float(x) for x in h2_prior.split(",")])

    samples, samples_psam = load_samples(plinks, samples_file)
    Y, X, phenotypes, samples = load_pheno_and_covars(
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
    key_cv, key_step1 = jax.random.split(jax.random.key(seed))
    train_mask, test_mask = get_cv_mask(len(Y), 5, key_cv)

    ## Run step 1
    step1_predictions = step1(
        pgen_datasets,
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
        key_step1,
        memory_mode.value,
    )

    step2(
        datasets,
        Y,
        X,
        step1_predictions,
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
        partition_phenotypes,
        max_rows,
    )


def main_entry() -> None:
    try:
        app()
    except Exception as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    main_entry()
