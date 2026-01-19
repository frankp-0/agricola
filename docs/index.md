---
icon: lucide/rocket
---

# lagga

**lagga** is a python package and command line tool for conducting genome-wide
association studies in admixed populations. Taking inspiration from the
[regenie](https://rgcgithub.github.io/regenie) and [Tractor](
https://atkinson-lab.github.io/Tractor-tutorial/) GWAS tools, **lagga** uses
whole-genome regression to efficiently account for related samples and population
structure, then performs local ancestry-adjusted single variant tests.

**lagga** has several key features:

- It uses [JAX](https://docs.jax.dev) for GPU-accelerated linear algebra and just-in-time compilation
- It uses [lanctools](https://github.com/frankp-0/lanctools) for highly-efficient local ancestry queries
- It performs local ancestry-adjusted single variant score tests
