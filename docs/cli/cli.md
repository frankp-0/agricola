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
  --outdir tmp/results \
  --variant-file1 tests/data/variants.txt \
  --trait-type qt
```

Alternatively, steps 0/1 and 2 can be performed separately:

```bash
agricola step1 \
  --plink-list tests/data/plinks.txt \
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
  --outdir tmp/results \
  --trait-type qt
```
