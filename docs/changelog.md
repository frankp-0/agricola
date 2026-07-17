# CHANGELOG

Notable changes to agricola (starting with v0.9.0) will be documented here.

## [v0.11.0] - 2026-07-17

[v0.11.0]: https://github.com/frankp-0/agricola/compare/v0.10.0...v0.11.0>

### Added

- Options to control output partitioning and write frequency
- Improved performance in step 1 using Cholesky decomposition
- Option to prune blocks in a dataset so that n_variants (mod block_size) = 0

### Fixed

- Force tab or space delim in pheno/covar files to fix silent errors when reading missing data

## [v0.10.0] - 2026-07-15

[v0.10.0]: https://github.com/frankp-0/agricola/compare/v0.9.0...v0.10.0>

### Added

- Tractor-style ancestry-specific p-values
- Option for double precision
- Simplified step 2 output (single parquet dir with rolling output files)
- Logger

### Fixed

- Improved numerical accuracy for edge cases in step 2 tests
- Failure to correctly mask likelihoods for missing samples in LRT
