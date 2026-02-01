# lagga CLI

`lagga` provides a command-line interface for running local-ancestry-aware GWAS.
The whole-genome regression and single variant tests can be performed in
**two steps** (`step1`, `step2`) or in a **single pipeline** (`all-steps`).

---

## Workflow Overview
```mermaid
flowchart LR
  A[plink2 files
.lanc files
phenotype file]
  B[Step 1: Whole-genome regression]
   C[Step 2: Variant inference]
  A --> B --> C
```


## Basic Usage

To get help, use the `--help` flag:

```bash
lagga --help
lagga step2 --help
```

### Examples
A typical `lagga` run may look like:

```bash
lagga all-steps \
  --plink-list tests/data/plinks.txt \
  --lanc-list tests/data/lancs.txt \
  --pheno-file tests/data/pheno.tsv \
  --covar-file tests/data/covar.tsv \
  --out-list tests/data/outs.txt \
  --variant-file1 tests/data/variants.txt \
  --trait-type qt
```

Alternatively, steps 0/1 and 2 can be performed separately:

```bash
lagga step1 \
  --plink-list tests/data/plinks.txt \
  --lanc-list tests/data/lancs.txt  \
  --pheno-file tests/data/pheno.tsv \
  --covar-file tests/data/covar.tsv \
  --out-prefix step1_preds \
  --variant-file tests/data/variants.txt \
  --trait-type qt

lagga step2 \
  --plink-list tests/data/plinks.txt \
  --lanc-list tests/data/lancs.txt  \
  --pheno-file tests/data/pheno.tsv \
  --covar-file tests/data/covar.tsv \
  --step1-prefix step1_preds \
  --out-list tests/data/outs.txt \
  --trait-type qt
```

---

## File Formats

### Inputs

#### Genotype

#### Local Ancestry

#### Phenotype

### Outputs

#### Step 1 Intermediate

#### Step 2 Results

---

## Options
### Global Options

These options are common to the `step1`, `step2`, and `all-steps` commands.


| Option      | Argument | Type | Description |
| --- | --- | --- | --- |
| `--plink-prefix` | TEXT | optional | Plink2 file prefix. This option can be repeated to specify multiple files |
| `--plink-list` | TEXT | optional | File containing plink2 prefixes, one per line |
| `--lanc-file` | TEXT | optional | Local ancestry .lanc file. This option can be repeated to specify multiple files |
| `--lanc-list` | TEXT | optional | File containing .lanc file paths, one per line |
| `--pheno-file` | TEXT | required | Phenotype file |
| `--ancestries` | TEXT | optional | Ancesry names, comma-separated and ordered as in .lanc files |
| `--covar-file` | TEXT | optional | Covariates file |
| `--samples-file` | TEXT | optional | Samples file |
| `--trait-type` | TEXT | optional | Trait type: quantitative (qt) or binary (bt) [default: qt] |


!!! info
    
    `--plink-prefix` and `--lanc-file` can be repeated to specify multiple files.
    E.g., `--plink-prefix tests/data/chr20 --plink-prefix tests/data/chr21 --plink-prefix tests/data/chr22`

!!! warning
    
    Plink2 and .lanc files must match, meaning you must provide the same number
    of plink2/.lanc files in the same order.

    Either `--plink-prefix` or `--plink-list` must be provided, but not both.
    The same applies to `--lanc-file` and `--lanc-list`

### Step 1 Options

These are the non-global options for `step1`:


| Option      | Argument | Type | Description |
| --- | --- | --- | --- |
| `--out-prefix` | TEXT | required | Step 1 predictions will be serialized and written to prefix.pkl |
| `--variant-file` | TEXT | optional| File with variants to include, one per line |
| `--h2-prior` | TEXT | optional | SNP heritability priors, comma-separated [default: 0.01,0.255,0.5,0.745,0.99] |
| `--block-size` | INTEGER | optional | Number of variants per block [default: 2000] |
| `--seed` | INTEGER | optional | Random seed [default: 100] |
| `--loocv` |   | optional | Use leave-one-out cross-validation (only for rare binary traits) [default: no-loocv] |


### Step 2 Options

These are the non-global options for `step2`:


| Option      | Argument | Type | Description |
| --- | --- | --- | --- |
| `--out-prefix` | TEXT | optional | Output prefix, one per plink_prefix |
| `--out-list` | TEXT | optional | File containg output file prefixes, one per line and plink2 prefix |
| `--variant-file` | TEXT | optional| File with variants to include, one per line |
| `--block-size` | INTEGER | optional | Number of variants per block [default: 1000] |
| `--min-ac` | INTEGER | optional | Minimum allele count [default: 1] |


!!! info
    
    `--out-prefix` can be repeated to specify multiple files, like `--plink-prefix` and `--lanc-file`.

!!! warning
    
    Either `--out-prefix` or `--out-list` must be provided, but not both.


### All Steps Options

These are the non-global options for `all-steps`


| Option      | Argument | Type | Description |
| --- | --- | --- | --- |
| `--out-prefix` | TEXT | optional | Output prefix, one per plink_prefix |
| `--out-list` | TEXT | optional | File containg output file prefixes, one per line and plink2 prefix |
| `--variant-file1` | TEXT | optional| File with variants to include for step 0/1, one per line |
| `--variant-file2` | TEXT | optional| File with variants to include for step 2, one per line |
| `--block-size0` | INTEGER | optional | Number of variants per block in step 0 [default: 2000] |
| `--block-size2` | INTEGER | optional | Number of variants per block in step 2 [default: 1000] |
| `--min-ac` | INTEGER | optional | Minimum allele count [default: 1] |
| `--seed` | INTEGER | optional | Random seed [default: 100] |
| `--loocv` |   | optional | Use leave-one-out cross-validation (only for rare binary traits) [default: no-loocv] |


!!! info
    
    `--out-prefix` can be repeated to specify multiple files, like `--plink-prefix` and `--lanc-file`.

!!! warning
    
    Either `--out-prefix` or `--out-list` must be provided, but not both.
