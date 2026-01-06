# lagga

## To Do

- [ ] Tests for models
  - ridge
    - OLS for lambda = 0
    - 0 for large lambda
    - All 0 for no test set
- [ ] Tests for step 0
  - Check dtypes
  - Check shapes
    - train, test masks have same col num
    - X, Y, masks have same n
  - negative h2_prior
  - B > 0
  - h2_prior > 0
- [ ] Tests for step 1
- [ ] Tests for step 2
- [ ] Tests for CLI
- [ ] Fix type hints

## In progress

## Done

- [x] Add license
- [x] Tests for _utils
- [x] Make optional dependencies for jax gpu/tpu
- [x] Move local ancestry code to lanctools package
