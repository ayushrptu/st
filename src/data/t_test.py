import numpy as np
from scipy.stats import f, chi2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import sympy
import matplotlib.pyplot as plt


def hotelling_t2_test(X, mu, Sigma):
    """
    compute the p-value for X of shape [n_samples, n_dims] to be sampled
    from a multivariate gaussian with mean <mu> and Covariance <Sigma>
    """
    n, p = X.shape
    mu_hat = np.mean(X, axis=0)
    t_squared = (mu_hat - mu).T @ np.linalg.solve(Sigma, mu_hat - mu)
    f_stat = (n - p) / (p * (n-1)) * t_squared
    p_value = 1 - f(p, n-p).cdf(f_stat)
    print("p-value: %g" % p_value)
    return p_value


def bartletts_test_cov(X, Sigma, logdet_hat=None, logdet=None):
    n, p = X.shape
    Sigma_hat = np.cov(X)
    # Exact calculations for log of det
    if logdet is None:
        logdet = float(sympy.log(sympy.Matrix(Sigma).det()))
    if logdet_hat is None:
        logdet_hat = float(sympy.log(sympy.Matrix(Sigma_hat).det()))

    U = (n-1) * (logdet - logdet_hat + np.sum(np.diag(Sigma_hat @ np.linalg.inv(Sigma).T)) - p)
    df = 0.5 * p * (p+1)

    if n >= 50:
        p_value = 1 - chi2(df).cdf(U)
    else:
        U_adj = U * (1-((1/(6*(n-1)-1))*(2*p+1-(2/(p+1)))))
        p_value = 1 - chi2(df).cdf(U_adj)
    print("p-value: %g" % p_value)
    return p_value


if __name__ == "__main__":
    n = 128
    samples = 100
    x = np.arange(n).reshape(-1, 1)
    gp1 = GaussianProcessRegressor(RBF(20) + WhiteKernel(0))
    gp2 = GaussianProcessRegressor(RBF(20) + WhiteKernel(0.5))
    X1 = gp1.sample_y(x, samples, random_state=0)
    X2 = gp2.sample_y(x, samples, random_state=0)

    # p11 = hotelling_t2_test(X1, np.zeros(n), gp1.kernel(x))
    # p12 = hotelling_t2_test(X1, np.zeros(n), gp2.kernel(x))
    # p21 = hotelling_t2_test(X2, np.zeros(n), gp1.kernel(x))
    # p22 = hotelling_t2_test(X2, np.zeros(n), gp2.kernel(x))
    # logdet_1 = float(sympy.log(sympy.Matrix(gp1.kernel(x)).det()))
    # logdet_2 = float(sympy.log(sympy.Matrix(gp2.kernel(x)).det()))
    # logdet_hat1 = float(sympy.log(sympy.Matrix(np.cov(X1)).det()))
    # logdet_hat2 = float(sympy.log(sympy.Matrix(np.cov(X2)).det()))
    # p11 = bartletts_test_cov(X1, gp1.kernel(x), logdet_hat=logdet_hat1, logdet=logdet_1)
    # p12 = bartletts_test_cov(X1, gp2.kernel(x), logdet_hat=logdet_hat1, logdet=logdet_2)
    # p21 = bartletts_test_cov(X2, gp1.kernel(x), logdet_hat=logdet_hat2, logdet=logdet_1)
    # p22 = bartletts_test_cov(X2, gp2.kernel(x), logdet_hat=logdet_hat2, logdet=logdet_2)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(np.cov(X1), vmin=0, vmax=1.5)
    ax[0, 1].imshow(gp1.kernel(x), vmin=0, vmax=1.5)
    ax[1, 0].imshow(np.cov(X2), vmin=0, vmax=1.5)
    ax[1, 1].imshow(gp2.kernel(x), vmin=0, vmax=1.5)
    plt.show()

    print("MSE X1 - smooth: %.4f" % np.linalg.norm(np.cov(X1) - gp1.kernel(x)))
    print("MSE X1 - spiky: %.4f" % np.linalg.norm(np.cov(X1) - gp2.kernel(x)))
    print("MSE X2 - smooth: %.4f" % np.linalg.norm(np.cov(X2) - gp1.kernel(x)))
    print("MSE X2 - spiky: %.4f" % np.linalg.norm(np.cov(X2) - gp2.kernel(x)))
    print("MAE X1 - smooth: %.4f" % np.linalg.norm(np.cov(X1) - gp1.kernel(x), ord=1))
    print("MAE X1 - spiky: %.4f" % np.linalg.norm(np.cov(X1) - gp2.kernel(x), ord=1))
    print("MAE X2 - smooth: %.4f" % np.linalg.norm(np.cov(X2) - gp1.kernel(x), ord=1))
    print("MAE X2 - spiky: %.4f" % np.linalg.norm(np.cov(X2) - gp2.kernel(x), ord=1))
