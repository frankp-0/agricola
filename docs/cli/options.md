# CLI Options

## Global Options

These options are common to the `step1`, `step2`, and `all-steps` commands.

| Option | Argument | Type | Description |
| --- | --- | --- | --- |
| `--plink` | TEXT | optional | Plink2 file prefix. This option can be repeated to specify multiple files |
| `--plink-list` | TEXT | optional | File containing plink2 prefixes, one per line |
| `--lanc` | TEXT | optional | Local ancestry .lanc file. This option can be repeated to specify multiple files |
| `--lanc-list` | TEXT | optional | File containing .lanc file paths, one per line |
| `--ancestries` | TEXT | optional | Ancestry names, comma-separated and ordered as in .lanc files |
| `--pheno-file` | TEXT | required | Phenotype file |
| `--pheno` | TEXT | optional | Phenotype to include in the analysis. This option can be repeated to specify multiple phenotypes or omitted to use all phenotypes |
| `--pheno-list` | TEXT | optional | File containing phenotypes to include in the analysis, one per line. This option can be omitted to use all phenotypes |
| `--covar-file` | TEXT | optional | Covariates file |
| `--covar` | TEXT | optional | Covariate to include in the analysis. This option can be repeated to specify multiple covariates or omitted to use all covariates |
| `--covar-list` | TEXT | optional | File containing covariates to include in the analysis, one per line. This option can be omitted to use all phenotypes |
| `--catcovar` | TEXT | optional | Categorical covariate to include in the analysis. This option can be repeated to specify multiple categorical covariates |
| `--catcovar-list` | TEXT | optional | File containing categorical covariates to include in the analysis, one per line. |
| `--samples-file` | TEXT | optional | Samples file |
| `--trait-type` | TEXT | optional | Trait type: quantitative (qt) or binary (bt) [default: qt] |
| `--double-precision` | | optional | Whether to use double instead of single precision [default: `--no-double-precision`] |
| `--backend` | TEXT | optional | Jax backend to use (e.g. --backend cpu or --backend cuda. Jax automatically detects the correct backend, but this can be specified to e.g. use cpu instead of cuda devices. |
| `--log` | TEXT | optional | Optional log file |
| `--verbose` | | optional | Whether to log debugging info [default: `--no-double-precision`] |

!!! info

    `--plink` and `--lanc` can be repeated to specify multiple files.
    E.g., `--plink tests/data/chr20 --plink tests/data/chr21 --plink tests/data/chr22`

!!! warning

    Plink2 and .lanc files must match, meaning you must provide the same number
    of plink2/.lanc files in the same order.

!!! warning

    Either `--plink` or `--plink-list` must be provided, but not both.
    The same applies to `--lanc` and `--lanc-list`. For `--pheno` and
    `--pheno-list` and `--covar` and `--covar-list`, either one may be provided
    or neither (to use all phenotypes/covariates).

!!! warning

    Categorical covariates specified through `catcovar` or `catcovar-list` must
    be a subset of the full list provided through `covar` or `covar-list`

## Step 1 Options

These are the non-global options for `step1`:

| Option | Argument | Type | Description |
| --- | --- | --- | --- |
| `--output` | TEXT | required | Step 1 predictions will be serialized and written to prefix.pkl |
| `--level0-dir` | TEXT | optional | Directory where level 0 predictions are saved (use temp dir if not provided) |
| `--variant-file` | TEXT | optional | File with variants to include, one per line |
| `--h2-prior` | TEXT | optional | SNP heritability priors, comma-separated [default: 0.01,0.255,0.5,0.745,0.99] |
| `--block-size` | INTEGER | optional | Number of variants per block [default: 2000] |
| `--seed` | INTEGER | optional | Random seed [default: 100] |
| `--loocv` | | optional | Use leave-one-out cross-validation (only for rare binary traits) [default: no-loocv] |
| `--memory-mode` | TEXT | optional | Ridge memory strategy for step 1: `standard` (broadcasted), `low` (sequential over alphas), or `lowest` (sequential over alphas and folds) [default: `standard`] |

## Step 2 Options

These are the non-global options for `step2`:

| Option | Argument | Type | Description |
| --- | --- | --- | --- |
| `--outdir` | TEXT | optional | Output directory |
| `--overwrite` | | optional | If true, any existing folders and files in outdir will be deleted If False, `--outdir` must be empty [default: `--no-overwrite`] |
| `--step1-prefix` | TEXT | optional | Step 1 predictions are read from prefix.pkl. If not provided, agricola does not condition on whole-genome regression |
| `--variant-file` | TEXT | optional | File with variants to include, one per line |
| `--chrom` | TEXT | optional | Specify a single chromosome for step 2 |
| `--test-type` | TEXT | optional | Either "score" or "wald [default: score] |
| `--adjust-lanc` | | optional | Either `--adjust-lanc` or `--no-adjust-lanc` [default: `--adjust-lanc`] |
| `--impute` | | optional | Either `--impute` or `--no-impute`. This must be `--no-impute` for binary traits. [default: `--no-impute`] |
| `--block-size` | INTEGER | optional | Number of variants per block [default: 1000] |
| `--min-ac` | INTEGER | optional | Minimum allele count threshold [default: 1] |
| `--partition_phenotypes` | | optional | Whether to partition output parquet files by phenotyp. If True, output files are written to e.g. outdir/trait0/part-0_0.parquet [default: --partition-phenotypes] |
| `--max-rows` | INTEGER | optional | Max number of rows/variants per phenotype to keep in memory before writing an output file. If unspecified, agricola will use 5000000 / len(phenotypes) |

!!! info

    `--no-impute` must be used for binary traits. If any quantitative traits have
    missing values, computational performance can be (often greatly) improved
    by using `--impute`, which mean-imputes all missing phenotype values.

## All Steps Options

These are the non-global options for `all-steps`

| Option | Argument | Type | Description |
| --- | --- | --- | --- |
| `--outdir` | TEXT | optional | Output directory |
| `--overwrite` | | optional | If true, any existing folders and files in outdir will be deleted If False, `--outdir` must be empty [default: `--no-overwrite`] |
| `--variant-file1` | TEXT | optional | File with variants to include for step 0/1, one per line |
| `--variant-file2` | TEXT | optional | File with variants to include for step 2, one per line |
| `--test-type` | TEXT | optional | Either "score" or "wald [default: score] |
| `--adjust-lanc` | | optional | Either `--adjust-lanc` or `--no-adjust-lanc` [default: `--adjust-lanc`] |
| `--impute` | | optional | Either `--impute` or `--no-impute`. This must be `--no-impute` for binary traits. [default: `--no-impute`] |
| `--block-size0` | INTEGER | optional | Number of variants per block in step 0 [default: 2000] |
| `--block-size2` | INTEGER | optional | Number of variants per block in step 2 [default: 1000] |
| `--min-ac` | INTEGER | optional | Minimum allele count [default: 1] |
| `--seed` | INTEGER | optional | Random seed [default: 100] |
| `--loocv` | | optional | Use leave-one-out cross-validation (only for rare binary traits) [default: no-loocv] |
| `--memory-mode` | TEXT | optional | Ridge memory strategy for step 1: `standard` (broadcasted), `low` (sequential over alphas), or `lowest` (sequential over alphas and folds) [default: `standard`] |
| `--partition_phenotypes` | | optional | Whether to partition output parquet files by phenotyp. If True, output files are written to e.g. outdir/trait0/part-0_0.parquet [default: --partition-phenotypes] |
| `--max-rows` | INTEGER | optional | Max number of rows/variants per phenotype to keep in memory before writing an output file. If unspecified, agricola will use 5000000 / len(phenotypes) |

!!! info

    `--no-impute` must be used for binary traits. If any quantitative traits have
    missing values, computational performance can be (often greatly) improved
    by using `--impute`, which mean-imputes all missing phenotype values.
