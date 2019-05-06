import cvxstrat

import networkx as nx
import numpy as np


def test_simple():
    def l_prox(lambd, eta, warm_start, pool):
        return (lambd * z + eta) / (lambd + 1)

    def r_prox(lambd, eta, warm_start, pool):
        return eta

    K = 10_000
    n = 10
    shape = (n,)
    z = np.random.randn(K, n)

    L = 1e-1 * nx.laplacian_matrix(nx.gnm_random_graph(K, 5 * K))

    result, info = cvxstrat.fit_stratified_model(
        L, shape, l_prox, r_prox, abs_tol=1e-8, rel_tol=1e-8, maxiter=200, n_jobs=1, verbose=True)

    theta = result['theta']
    residual = theta - z + L @ theta
    np.testing.assert_allclose(np.linalg.norm(
        residual) / (K * n), 0, atol=1e-5)


def test_poisson():

    K = 10_000
    N = np.ones((K, 1))
    S = np.random.randint(0, 10, size=(K, 1))
    shape = (1,)
    min_lambda = 1e-5

    def l_prox(lambd, eta, warm_start, pool):
        b = lambd * N - eta
        c = - lambd * S
        theta = (-b + np.sqrt(b**2 - 4 * c)) / 2.
        return np.maximum(theta, min_lambda)

    def r_prox(lambd, eta, warm_start, pool):
        return np.maximum(eta, min_lambda)

    L = 1e-1 * nx.laplacian_matrix(nx.gnm_random_graph(K, 5 * K))

    result, info = cvxstrat.fit_stratified_model(
        L, shape, l_prox, r_prox, abs_tol=1e-5, rel_tol=1e-5, maxiter=200, n_jobs=6, verbose=True)
    theta = result['theta']
    residual = N - S / theta + L @ theta
    np.testing.assert_allclose(np.abs(residual) / K, 0, atol=1e-4)


def test_regression():
    K = 100
    G = nx.cycle_graph(K)
    n = 10
    m = 2
    X = np.random.randn(500, n)
    Z = np.random.randint(K, size=500)
    Y = np.random.randn(500, m)

    p = cvxstrat.RidgeRegression()
    p.fit(X, Y, Z, G, inplace=True, verbose=True)

    score = p.score(X, Y, Z)
    predict = p.predict(X, Z)
    print(score)
    print(predict[:5])


def test_log_reg():
    K = 300
    G = nx.cycle_graph(K)
    n = 10
    X = np.random.randn(1000, n)
    Z = np.random.randint(K, size=1000)
    Y = np.random.randint(1, 10, size=1000)

    p = cvxstrat.LogisticRegression()
    p.fit(X, Y, Z, G, inplace=True, verbose=True, n_jobs=12)

    anll = p.anll(X, Y, Z)
    predict = p.predict(X, Z)
    probs = p.predict(X, Z, probs=True)
    print(anll)
    print(predict[:5])
    print(probs)


def test_bernoulli():
    K = 10_000
    G = nx.cycle_graph(K)
    Z = np.random.randint(K, size=1_00_000)
    Y = np.random.randint(0, 2, size=1_00_000)

    p = cvxstrat.Bernoulli()
    p.fit(Y, Z, G, inplace=True, verbose=True, n_jobs=12)

    anll = p.anll(Y, Z)
    sample = p.sample(Z)
    print(sample)
    print(anll)


def test_poisson_model():
    K = 10_000
    G = nx.cycle_graph(K)
    Z = np.random.randint(K, size=1_000_000)
    Y = np.random.randint(1, 10, size=1_000_000)

    p = cvxstrat.Poisson()
    p.fit(Y, Z, G, inplace=True, verbose=True, n_jobs=12)

    anll = p.anll(Y, Z)
    sample = p.sample(Z)
    print(anll)
    print(sample)


if __name__ == '__main__':
    np.random.seed(0)
    test_simple()
    test_poisson()
    test_regression()
    test_log_reg()
    test_bernoulli()
    test_poisson_model
    print("All tests passed!")
