---
icon: lucide/wheat
---

# agricola

**agricola** is a command-line tool and Python package for conducting **genome-wide
association studies (GWAS) in admixed populations**. Inspired by [regenie](https://rgcgithub.github.io/regenie)
and [Tractor](https://atkinson-lab.github.io/Tractor-tutorial/), **agricola**
provides a scalable, local-ancestry–aware framework that handles relatedness, population
structure, and ancestry effect heterogeneity.

---

## Why agricola?

Admixed individuals have unique LD patterns that can improve signal localization and 
improve power for population-specific causal variants. However, standard GWAS
tools fail to adjust for local ancestry or model effect heterogeneity in admixed
individuals.

Tools like Tractor, Tractor-Mix, and SAIGE-Tractor address this gap. **agricola** follows the same
conceptual approach—performing single-variant association tests with explicit
local ancestry adjustment—and combines it with:

- **Accelerated linear algebra** via [JAX](https://docs.jax.dev)
- **CUDA GPU, TPU, or CPU support** for flexible compute environments
- **Efficient local ancestry queries** using [lanctools](https://frankp-0.github.io/lanctools/)
- **Multi-phenotype** modeling
- **Adjustment for sample relatedness**

---

## Installation

**Requirements:** Python 3.10+

Install via pip:

```bash
pip install agricola
```

For GPU or TPU support:

```bash
pip install agricola[cuda]
```

```bash
pip install agricola[tpu]
```
