import jax
import jax.numpy as jnp


def _ridge_masked(X, Y, w_train, w_test, alphas):
    _, b = X.shape
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    w_train = w_train.reshape(-1, 1)
    w_test = w_test.reshape(-1, 1)
    XtX = X.T @ (X * w_train)
    XtY = X.T @ (Y * w_train)
    I_ = jnp.eye(b, dtype=jnp.float32)

    def ridge_pred(alpha):
        A = XtX + alpha * I_
        beta = jnp.linalg.solve(A, XtY)
        return (X @ beta) * w_test

    preds = jax.vmap(ridge_pred)(alphas)
    preds = jnp.moveaxis(preds, 0, -1)

    # returns (N, P, len(alphas))
    # preds is 0 for samples with w_test = False
    return preds
