import jax
import jax.numpy as jnp


def _ridge_masked(X, Y, w_train, w_test, alphas):
    """Perform ridge regression using test/train masks

    The w_train and w_test are train/test "weights" for samples.
    Both should only take values 0 and 1. w_train is used as a "mask" for
    training samples. Samples with w_test = 0 have predicted value 0
    in the output. Passing "masks" like this allows data to have the
    same input size regardless of train/test split and avoid recompilation
    should we choose to use JIT compilation.


    Args:
        X: (N, V) jax array of predictors
        Y: (N, P) or (N,) jax array of outcome(s)
        w_train: (N, 1) jax array indicating training set status (0/1)
        w_test: (N, 1) jax array indicating test set status (0/1)
        alphas: A 1d jax array of ridge penalty weights

    Returns:
        preds: A (N, len(alphas), P) jax array of predictions (with non-test samples masked out)
    """
    _, b = X.shape

    ## Reshape weights to 1D and Y to 2D
    w_train = w_train.reshape(-1, 1)
    w_test = w_test.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    ## Perform ridge regression
    XtX = X.T @ (X * w_train)
    XtY = X.T @ (Y * w_train)
    I_ = jnp.eye(b, dtype=jnp.float32)

    def ridge_pred(alpha):
        A = XtX + alpha * I_
        beta = jnp.linalg.solve(A, XtY)
        return (X @ beta) * w_test

    preds = jax.vmap(ridge_pred)(alphas)
    preds = jnp.moveaxis(preds, 0, -1)

    return preds
