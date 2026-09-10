# CHANGELOG

Notable changes to agricola (starting with v1.0.0) will be documented here.

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
