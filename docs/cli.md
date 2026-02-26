# agricola CLI

`agricola` provides a command-line interface for running local-ancestry-aware GWAS.
The whole-genome regression and single variant tests can be performed in
**two steps** (`step1`, `step2`) or in a **single pipeline** (`all-steps`).

---

## Basic Usage

To get help, use the `--help` flag:

```bash
agricola --help
agricola step2 --help
```

### Examples
A typical `agricola` run may look like:

```bash
agricola all-steps \
  --plink-list tests/data/plinks.txt \
  --lanc-list tests/data/lancs.txt \
  --pheno-file tests/data/pheno.tsv \
  --covar-file tests/data/covar.tsv \
  --output-list tests/data/outs.txt \
  --variant-file1 tests/data/variants.txt \
  --trait-type qt
```

Alternatively, steps 0/1 and 2 can be performed separately:

```bash
agricola step1 \
  --plink-list tests/data/plinks.txt \
  --lanc-list tests/data/lancs.txt  \
  --pheno-file tests/data/pheno.tsv \
  --covar-file tests/data/covar.tsv \
  --output step1_preds \
  --variant-file tests/data/variants.txt \
  --trait-type qt

agricola step2 \
  --plink-list tests/data/plinks.txt \
  --lanc-list tests/data/lancs.txt  \
  --pheno-file tests/data/pheno.tsv \
  --covar-file tests/data/covar.tsv \
  --step1-prefix step1_preds \
  --output-list tests/data/outs.txt \
  --trait-type qt
```

---

## File Formats

### Inputs

#### Genotype

**agricola** accepts genotype files in plink2 .pgen format (with corresponding
.pvar and .psam). Please see the [plink2 documentation](https://www.cog-genomics.org/plink/2.0/) for futher details.
**agricola** accepts either 1) a single pgen file with multiple chromosomes, or 2) a set of plink2
files, each corresponding to a separate chromosome.

!!! info
    
    Extension to other formats such as bgen or vcf is possible but
    is not currently a priority. **agricola** requires phasing information,
    so unphased formats such as .bed are not possible.

!!! warning

    It is assumed that all pgen/pvar files are sorted by chromosome and position.
    

#### Local Ancestry

**agricola** accepts only .lanc files, as defined by admix-kit, for local ancestry.
This format was chosen for its flexibility, low memory overhead, and simplicity.
Please see the admix-kit documentation for further details.

To make working with this format easier, we introduce the **lanctools**
Python package and CLI tool. **lanctools** can convert RFMix msp.tsv
files or FLARE vcf.gz files into .lanc format. We provide an example below.
Please see the [lanctools documentation](https://lanctools.readthedocs.io) for further details.

```bash
# convert FLARE to .lanc format
lanctools convert-rfmix --file chr1.msp.tsv --plink-prefix chr1 --output chr1.lanc
```

Although most local ancestry inference algorithms produce chromosome-specific files,
you may prefer to work with a single file containing multiple chromosomes.
To do so, first merge pgen files using plink2, then use the `lanctools merge`
command to combine the .lanc files.

```bash
# merge multiple .lanc files
lanctools merge --file chr1.lanc,chr2.lanc,chr3.lanc --outfile chr1_3.lanc
```

#### Phenotypes and Covariates

Phenotype and covariate files are expected to be whitespace-delimited text files.
This means that column names may not contain whitespace.
If a header is not provided, it is assumed that the first column is for family IDs
(FID) and the second column is for individual IDs (IID). If a header line
is provided, it must begin with a `#` character and must include "IID" as a
column name. Two valid examples are given below

```
#IID	height	crp_irnt
sample1	165	-1.23
sample2	175	-2.04
sample3	161	0.81
```

```
sample1	sample1	165	-1.23
sample2	sample2	175	-2.04
sample3	sample3	161	0.81
```

### Outputs

#### Step 1 Intermediate

Whole-genome leave-one-chromosome-out (LOCO) predictions from step 1 are saved
to the file `{prefix}.pkl`. This is file consists of a serialized dictionary,
where keys are chromosomes and values are $(N, P)$ NumPy arrays with
predictions for each sample and phenotype.

#### Step 2 Results

Summary statistics from step 2 of **agricola** are saved in [Apache Parquet](https://parquet.apache.org/)
format, one output file per input plink file and phenotype. These files have
the following schema:


| Field | Type | Description |
| --- | --- | --- |
| chrom | string | Chromosome |
| pos | int | Genomic position |
| ref | string | Reference allele |
| alt | string |  Alternate allele |
| id | string | Variant ID |
| log10p_het | double | P-value for test $\beta_{\text{anc}_0}=\cdots=\beta_{\text{anc}_k}=0$ |
| beta_{anc} | double | Effect estimate for $\beta_{\text{anc}}$ |
| log10p_hom | double | P-value for $\beta=0$ under homogeneous model (all ancestry effects equal) |
| N | int | Sample size |
| AF_{anc} | double | Ancestry-specific allele frequency |
| LAprop_{anc} | double | Proportion of haplotypes from ancestry anc |
| log10p_lrt | double | P-value for likelihood ratio test of heterogeneous vs. homogeneous model (only output for Wald test) |

---

## Options
### Global Options

These options are common to the `step1`, `step2`, and `all-steps` commands.


| Option      | Argument | Type | Description |
| --- | --- | --- | --- |
| `--plink` | TEXT | optional | Plink2 file prefix. This option can be repeated to specify multiple files |
| `--plink-list` | TEXT | optional | File containing plink2 prefixes, one per line |
| `--lanc` | TEXT | optional | Local ancestry .lanc file. This option can be repeated to specify multiple files |
| `--lanc-list` | TEXT | optional | File containing .lanc file paths, one per line |
| `--ancestries` | TEXT | optional | Ancestry names, comma-separated and ordered as in .lanc files |
| `--pheno-file` | TEXT | required | Phenotype file |
| `--pheno` | TEXT | optional | Phenotype to include in the analysis. This option
can be repeated to specify multiple files or omitted to use all phenotypes |
| `--pheno-list` | TEXT | optional | File containing phenotypes to include in
the analysis, one per line. This option can be omitted to use all phenotypes |
| `--covar-file` | TEXT | optional | Covariates file |
| `--covar` | TEXT | optional | Covariate to include in the analysis. This option
can be repeated to specify multiple files or omitted to use all covariates |
| `--covar-list` | TEXT | optional | File containing covariates to include in
the analysis, one per line. This option can be omitted to use all phenotypes |
| `--catcovar` | TEXT | optional | Categorical covariate to include in the analysis. This option
can be repeated to specify multiple files or omitted to use all covariates |
| `--catcovar-list` | TEXT | optional | File containing categorical covariates to include in
the analysis, one per line. This option can be omitted to use all phenotypes |
| `--samples-file` | TEXT | optional | Samples file |
| `--trait-type` | TEXT | optional | Trait type: quantitative (qt) or binary (bt) [default: qt] |


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

### Step 1 Options

These are the non-global options for `step1`:


| Option      | Argument | Type | Description |
| --- | --- | --- | --- |
| `--output` | TEXT | required | Step 1 predictions will be serialized and written to prefix.pkl |
| `--variant-file` | TEXT | optional| File with variants to include, one per line |
| `--h2-prior` | TEXT | optional | SNP heritability priors, comma-separated [default: 0.01,0.255,0.5,0.745,0.99] |
| `--block-size` | INTEGER | optional | Number of variants per block [default: 2000] |
| `--seed` | INTEGER | optional | Random seed [default: 100] |
| `--loocv` |   | optional | Use leave-one-out cross-validation (only for rare binary traits) [default: no-loocv] |


### Step 2 Options

These are the non-global options for `step2`:


| Option      | Argument | Type | Description |
| --- | --- | --- | --- |
| `--output` | TEXT | optional | Output prefix, one per plink_prefix |
| `--output-list` | TEXT | optional | File containg output file prefixes, one per line and plink2 prefix |
| `--variant-file` | TEXT | optional| File with variants to include, one per line |
| `--chrom` | TEXT | optional | Specify a single chromosome for step 2 |
| `--test-type` | TEXT | optional | Either "score" or "wald [default: score] |
| `--adjust-lanc` | | optional | Either `--adjust-lanc` or `--no-adjust-lanc` [default: `--adjust-lanc`] |
| `--impute` | | optional | Either `--impute` or `--no-impute`. This must be `--no-impute` for binary traits. [default: `--impute`] |
| `--block-size` | INTEGER | optional | Number of variants per block [default: 1000] |
| `--min-ac` | INTEGER | optional | Minimum allele count [default: 1] |


!!! info
    
    `--out-prefix` can be repeated to specify multiple files, like `--plink-prefix` and `--lanc-file`.

!!! warning
    
    Either `--out-prefix` or `--output-list` must be provided, but not both.

!!! info

    `--no-impute` must be used for binary traits. If any quantitative traits have
    missing values, computational performance can be (often greatly) improved
    by using `--impute`, which mean-imputes all missing phenotype values.


### All Steps Options

These are the non-global options for `all-steps`


| Option      | Argument | Type | Description |
| --- | --- | --- | --- |
| `--output` | TEXT | optional | Output prefix, one per plink_prefix |
| `--output-list` | TEXT | optional | File containg output file prefixes, one per line and plink2 prefix |
| `--variant-file1` | TEXT | optional| File with variants to include for step 0/1, one per line |
| `--variant-file2` | TEXT | optional| File with variants to include for step 2, one per line |
| `--test-type` | TEXT | optional | Either "score" or "wald [default: score] |
| `--adjust-lanc` | | optional | Either `--adjust-lanc` or `--no-adjust-lanc` [default: `--adjust-lanc`] |
| `--impute` | | optional | Either `--impute` or `--no-impute`. This must be `--no-impute` for binary traits. [default: `--impute`] |
| `--block-size0` | INTEGER | optional | Number of variants per block in step 0 [default: 2000] |
| `--block-size2` | INTEGER | optional | Number of variants per block in step 2 [default: 1000] |
| `--min-ac` | INTEGER | optional | Minimum allele count [default: 1] |
| `--seed` | INTEGER | optional | Random seed [default: 100] |
| `--loocv` |   | optional | Use leave-one-out cross-validation (only for rare binary traits) [default: no-loocv] |


!!! info
    
    `--output` can be repeated to specify multiple files, like `--plink` and `--lanc`.

!!! warning
    
    Either `--output` or `--out-list` must be provided, but not both.

!!! info

    `--no-impute` must be used for binary traits. If any quantitative traits have
    missing values, computational performance can be (often greatly) improved
    by using `--impute`, which mean-imputes all missing phenotype values.
