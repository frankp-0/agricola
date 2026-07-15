# CHANGELOG

Notable changes to agricola (starting with v0.9.0) will be documented here.

## v0.10.0 - 2026-07-15

### Added

- Tractor-style ancestry-specific p-values
- Option for double precision
- Simplified step 2 output (single parquet dir with rolling output files)
- Logger

### Fixed

- Improved numerical accuracy for edge cases in step 2 tests
- Failure to correctly mask likelihoods for missing samples in LRT
