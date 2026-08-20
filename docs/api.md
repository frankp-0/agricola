# agricola Python API

`agricola` exports two functions which can be used in place of the recommended command line workflow:

The implementation is organized into focused namespaces:

- `agricola.pipeline` contains step orchestration and output writing.
- `agricola.io` contains genotype, local-ancestry, and variant access helpers.
- `agricola.models` contains ridge and logistic-ridge models.
- `agricola.statistics` contains quantitative and binary association kernels.
- `agricola.numerical` contains shared preprocessing and linear-algebra helpers.
- `agricola.validation` contains input validation and preparation.

---

## ::: agricola.pipeline.step1.step1

    handler: python
    options:
      show_root_heading: true
      show_source: true

## ::: agricola.pipeline.step2.step2

    handler: python
    options:
      show_root_heading: true
      show_source: true
