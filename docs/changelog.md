# CHANGELOG

Notable changes to agricola (starting with v0.9.0) will be documented here.

## [v0.16.0] - 2026-07-28

[v0.16.0]: https://github.com/frankp-0/agricola/compare/v0.15.0...v0.16.0>

### Fixed

- Fit y ~ covar + offset(pgs) for bt traits.
This null model should follow regenie. Previously, fit y ~ covar + pgs for null
model. Before that (<= 0.15.0), fit y ~ covar then added offset.

## [v0.15.0] - 2026-07-27

[v0.15.0]: https://github.com/frankp-0/agricola/compare/v0.14.1...v0.15.0>

### Fixed

- Fit y ~ pgs + covar for bt traits. Previously summed covar and pgs offsets, leading to conservative tests

### Added

- Allow chromosomes in step2 not included in step 1

## [v0.14.1] - 2026-07-22

[v0.14.1]: https://github.com/frankp-0/agricola/compare/v0.14.0...v0.14.1>

### Fixed

- Bug in step2 CLI only using the last chromosome
- Do not log warning about backend when none is specified

## [v0.14.0] - 2026-07-22

[v0.14.0]: https://github.com/frankp-0/agricola/compare/v0.13.0...v0.14.0>

### Added

- Option to specify JAX backend
- Improved speed for level 1 logistic regression by vmapping across folds and
blocks in level1 logistic regression

## [v0.13.0] - 2026-07-21

[v0.13.0]: https://github.com/frankp-0/agricola/compare/v0.12.0...v0.13.0>

### Added

- Support for using subset of step1 samples in step2

## [v0.12.0] - 2026-07-21

[v0.12.0]: https://github.com/frankp-0/agricola/compare/v0.11.0...v0.12.0>

### Added

- Welcome message
- Lower memory, slightly improved speed for imputed traits in step 2

### Fixed

- Bug in calculating residuals for _qt_lanc_wald
- Use convergence instead of fixed iterations for logistic regression

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
