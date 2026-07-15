# File Formats

## Inputs

### Genotype

**agricola** accepts genotype files in plink2 .pgen format (with corresponding
.pvar and .psam). Please see the [plink2 documentation][plink2_docs] for futher
details. **agricola** accepts either 1) a single pgen file with multiple
chromosomes, or 2) a set of plink2 files, each corresponding to a separate
chromosome.

!!! info

    Extension to other formats such as bgen or vcf is possible but
    is not currently a priority. **agricola** requires phasing information,
    so unphased formats such as .bed are not possible.

!!! warning

    It is assumed that all pgen/pvar files are sorted by chromosome and position.

!!! warning

    Agricola relies on a leave-one-chromosome-out scheme for whole genome
    predictionn. In step 1, multiple chromosomes (ideally all autosomes) must be
    provided.

### Local Ancestry

**agricola** accepts only .lanc files, as defined by admix-kit, for local ancestry.
This format was chosen for its flexibility, low memory overhead, and simplicity.
Please see the admix-kit documentation for further details.

To make working with this format easier, we introduce the **lanctools**
Python package and CLI tool. **lanctools** can convert RFMix msp.tsv
files or FLARE vcf.gz files into .lanc format. We provide an example below.
Please see the [lanctools documentation][flare_docs] for further details.

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
lanctools merge --input chr1.lanc --input chr2.lanc --input chr3.lanc --output chr1_3.lanc
```

### Phenotypes and Covariates

Phenotype and covariate files are expected to be whitespace-delimited text files.
This means that column names may not contain whitespace.
If a header is not provided, it is assumed that the first column is for family IDs
(FID) and the second column is for individual IDs (IID). If a header line
is provided, it must begin with a `#` character and must include "IID" as a
column name. Two valid examples are given below

```
#IID height crp_irnt
sample1 165 -1.23
sample2 175 -2.04
sample3 161 0.81
```

```
sample1 sample1 165 -1.23
sample2 sample2 175 -2.04
sample3 sample3 161 0.81
```
