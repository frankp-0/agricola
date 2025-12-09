import jax
import jax.numpy as jnp
import jax.lax as lax

### ─────────────────────────────────────────────────────────────
### Quantitative Traits
### ─────────────────────────────────────────────────────────────


def _ridge(X, Y, w_train, w_test, alphas):
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


### ─────────────────────────────────────────────────────────────
### Binary Traits
### ─────────────────────────────────────────────────────────────


def _sigmoid(x):
    return jnp.clip(0.5 * (1.0 + jax.nn.tanh(0.5 * x)), 1e-12, 1 - 1e-12)


def _logistic_ridge_step(beta, X, y, offset, w_train, alpha):
    """Updates coefficients in logistic ridge regression

    Args:
        beta: (V,) jax array of current coefficients
        X: (N, V) jax array of predictors
        y: (N,) jax array of the outcome
        offset: (N,) jax array with offset
        w_train: (N,) jax array indicating training set status (0/1)
        alpha: ridge penalty weight

    Returns:
        beta: (V,) jax array of coefficients
    """

    eta = X @ beta + offset
    mu = _sigmoid(eta)
    r = (y - mu) * w_train
    w = mu * (1 - mu) * w_train
    XW = X * w[:, None]
    XT_r = X.T @ r
    H = (X.T @ XW) + (alpha * jnp.eye(X.shape[1]))
    delta = jnp.linalg.solve(H, XT_r)
    beta_new = beta + delta
    return beta_new


def _logistic_ridge(X, y, offset, w_train, alpha, max_iter=50):
    """Perform logistic ridge regression

    Returns estimated coefficients

    Args:
        X: (N, V) jax array of predictors
        y: (N,) jax array of the outcome
        offset: (N,) jax array with offset
        w_train: (N,) jax array indicating training set status (0/1)
        alpha: ridge penalty weight
        max_iter: max number of iterations

    Returns:
        eta: (N,) jax array of linear predictors
    """
    beta0 = jnp.zeros(X.shape[1])

    def body_fun(i, beta):
        return _logistic_ridge_step(beta, X, y, offset, w_train, alpha)

    beta = lax.fori_loop(0, max_iter, body_fun, beta0)
    eta = X @ beta + offset

    return eta


def _logistic_ridge_loo(X, y, offset, alpha, max_iter=50):
    """Perform logistic ridge regression with leave-one-out scheme

    Returns leave-one-out linear predictor

    Args:
        X: (N, V) jax array of predictors
        y: (N,) jax array of the outcome
        offset: (N,) jax array with offset
        alpha: ridge penalty weight
        max_iter: max number of iterations

    Returns:
        eta_loo: (N,) jax array of linear predictions
    """
    beta0 = jnp.zeros(X.shape[1])

    w_train = jnp.ones((X.shape[0]))

    def body_fun(i, beta):
        return _logistic_ridge_step(beta, X, y, offset, w_train, alpha)

    beta = lax.fori_loop(0, max_iter, body_fun, beta0)
    eta = X @ beta + offset
    mu = _sigmoid(eta)
    w = mu * (1 - mu)
    XW = X * w[:, None]
    H_inv = jax.scipy.linalg.inv((X.T @ XW) + (alpha * jnp.eye(X.shape[1])))

    def foo(x):
        return (x @ H_inv) @ x.T

    Gamma = w * jax.vmap(foo, in_axes=0)(X)

    beta_loo = beta[:, None] - ((H_inv @ X.T) * (y - mu) / (1 - Gamma))
    eta_loo = jnp.sum(X * (beta_loo.T), axis=1) + offset

    return eta_loo
