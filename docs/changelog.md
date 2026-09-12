# CHANGELOG

Notable changes to agricola (starting with v1.0.0) will be documented here.

## [v1.4.0] - 2026-09-12

[v1.4.0]: https://github.com/frankp-0/agricola/compare/v1.3.5...v1.4.0>

### Added

- Milestone progress logging for batch jobs

## [v1.3.5] - 2026-09-12

[v1.3.5]: https://github.com/frankp-0/agricola/compare/v1.3.4...v1.3.5>

### Fixed

- Bug where phenotype column has NULL type in step 2

## [v1.3.4] - 2026-09-11

[v1.3.4]: https://github.com/frankp-0/agricola/compare/v1.3.3...v1.3.4>

### Fixed

- Omit reference columns for categorical covariates

## [v1.3.3] - 2026-09-11

[v1.3.3]: https://github.com/frankp-0/agricola/compare/v1.3.2...v1.3.3>

### Fixed

- Align NumPy datatypes with JAX

## [v1.3.2] - 2026-09-10

[v1.3.2]: https://github.com/frankp-0/agricola/compare/v1.3.1...v1.3.2>

### Fixed

- Bug in _bt_score_lanc computing chisq test before mask is applied

## [v1.3.1] - 2026-09-10

[v1.3.1]: https://github.com/frankp-0/agricola/compare/v1.3.0...v1.3.1>

### Fixed

- Use minor instead of coded allele for allele count thresholds

## [v1.3.0] - 2026-09-10

[v1.3.0]: https://github.com/frankp-0/agricola/compare/v1.2.0...v1.3.0>

### Added

- Option to specify allele count threshold for genotype (min-ac-h) and ancestry-deconvoluted genotype (min-ac-g)

## [v1.2.0] - 2026-09-09

[v1.2.0]: https://github.com/frankp-0/agricola/compare/v1.1.0...v1.2.0>

### Added

- Score test for heterogeneous vs. homogeneous model
- p-het-threshold option to test het vs. hom model only for variants/phenotypes passing P_HET < threshold
- Warm start for _bt_lanc_score

### Fixed

- Remove 1-step approximation from v1.1.0

## [v1.1.0] - 2026-09-08

[v1.1.0]: https://github.com/frankp-0/agricola/compare/v1.0.0...v1.1.0>

### Added

- Increased tolerance for binary trait tests
- Use 1-step approximation for the null Y ~ offset + L model in _bt_lanc_score

## [v1.0.0] - 2026-09-06

- First release version
