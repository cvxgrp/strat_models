from cvxstrat.fit import fit_stratified_model

import numpy as np
from scipy.stats import poisson, bernoulli
from scipy.special import softmax
import torch
from sklearn import preprocessing
import networkx as nx

import copy


def G_to_data(G, shape):
    """Vectorizes the variables in G and returns a dictionary."""
    theta_init = np.zeros(shape)
    theta_tilde_init = np.zeros(shape)
    theta_hat_init = np.zeros(shape)
    u_init = np.zeros(shape)
    u_tilde_init = np.zeros(shape)
    for i, node in enumerate(G.nodes()):
        vertex = G.node[node]
        if 'theta' in vertex:
            theta_init[i] = vertex['theta']
        if 'theta_tilde' in vertex:
            theta_tilde_init[i] = vertex['theta_tilde']
        if 'theta_hat' in vertex:
            theta_hat_init[i] = vertex['theta_hat']
        if 'u' in vertex:
            u_init[i] = vertex['u']
        if 'u_tilde' in vertex:
            u_tilde_init[i] = vertex['u_tilde']

    data = {
        'theta_init': theta_init,
        'theta_tilde_init': theta_tilde_init,
        'theta_hat_init': theta_hat_init,
        'u_init': u_init,
        'u_tilde_init': u_tilde_init,
    }

    return data


def transfer_result_to_G(result, G):
    """Puts solution vectors into a graph G"""
    theta = result['theta']
    theta_tilde = result['theta_tilde']
    theta_hat = result['theta_hat']
    u = result['u']
    u_tilde = result['u_tilde']
    for i, node in enumerate(G.nodes()):
        vertex = G.node[node]
        vertex['theta'] = theta[i]
        vertex['theta_tilde'] = theta_tilde[i]
        vertex['theta_hat'] = theta_hat[i]
        vertex['u'] = u[i]
        vertex['u_tilde'] = u_tilde[i]
    return None


def turn_into_iterable(x):
    try:
        iter(x)
    except TypeError:
        return [x]
    else:
        return x


def find_solution(x):
    """Finds the real solution to ax^3 + bx^2 + cx + d = 0."""
    roots = np.roots(x)
    for root in roots:
        if np.isreal(root) and root >= 1e-4 and root <= 1 - 1e-4:
            return np.real(root)
    return 0.5


def log_reg_prox(XY, eta, theta, lambd):
    """Proximal operator for multinomial logistic regression."""
    if XY is None:
        return eta
    X, Y = XY
    eta_tch = torch.from_numpy(eta)
    theta_i = torch.from_numpy(theta).requires_grad_(True)
    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    optim = torch.optim.LBFGS([theta_i], lr=1, max_iter=50)

    def closure():
        optim.zero_grad()
        l = lambd * loss(X@theta_i, Y) + 0.5 * torch.sum((theta_i - eta_tch)**2)
        l.backward()
        return l

    optim.step(closure)
    return theta_i.data.numpy()


class Bernoulli():
    """Stratified Bernoulli model."""

    def __init__(self, min_theta=1e-5, max_theta=1 - 1e-5):
        self.min_theta = min_theta
        self.max_theta = max_theta

    def fit(self, Y, Z, G, inplace=False, **kwargs):
        if inplace:
            self.G = G
        else:
            self.G = copy.deepcopy(G)

        L = nx.laplacian_matrix(self.G)
        nodelist = self.G.nodes()
        K = L.shape[0]
        shape = (1,)
        theta_shape = (K,) + shape

        # preprocess data
        for y, z in zip(Y, Z):
            vertex = self.G.node[z]
            if 'Y' in vertex:
                vertex['Y'] += [y]
            else:
                vertex['Y'] = [y]

        S = np.zeros((K, 1))
        N = np.zeros((K, 1))

        for i, node in enumerate(nodelist):
            vertex = self.G.node[node]
            if 'Y' in vertex:
                S[i] = np.sum(vertex['Y'])
                N[i] = len(vertex['Y'])
                del vertex['Y']

        def l_prox(lambd, eta, warm_start, pool):
            a = np.ones(eta.shape) * -1
            b = eta + 1
            c = lambd * N - eta
            d = -S * lambd
            coefs = np.hstack([a, b, c, d])
            theta = np.array(pool.map(find_solution, coefs))[:, np.newaxis]
            return np.clip(theta, self.min_theta, self.max_theta)

        def r_prox(lambd, eta, warm_start, pool):
            return np.clip(eta, self.min_theta, self.max_theta)

        data = G_to_data(self.G, theta_shape)

        result, info = fit_stratified_model(
            L, shape, l_prox, r_prox, data=data, **kwargs)

        transfer_result_to_G(result, self.G)

        return info

    def anll(self, Y, Z):
        return -np.mean(self.logpmf(Y, Z))

    def logpmf(self, Y, Z):
        Y = turn_into_iterable(Y)
        Z = turn_into_iterable(Z)
        parameter = np.array([self.G.node[z]['theta'][0] for z in Z])
        return bernoulli.logpmf(Y, p=parameter)

    def sample(self, Z):
        Z = turn_into_iterable(Z)
        parameter = np.array([self.G.node[z]['theta'][0] for z in Z])
        return np.random.binomial(1, p=parameter)


class Poisson():
    """Stratified Poisson distribution."""

    def __init__(self, min_lambda=1e-5):
        self.min_lambda = min_lambda

    def fit(self, Y, Z, G, inplace=False, **kwargs):
        if inplace:
            self.G = G
        else:
            self.G = copy.deepcopy(G)

        L = nx.laplacian_matrix(self.G)
        nodelist = self.G.nodes()
        K = L.shape[0]
        shape = (1,)
        theta_shape = (K,) + shape

        # preprocess data
        for y, z in zip(Y, Z):
            vertex = self.G.node[z]
            if 'Y' in vertex:
                vertex['Y'] += [y]
            else:
                vertex['Y'] = [y]

        S = np.zeros((K, 1))
        N = np.zeros((K, 1))

        for i, node in enumerate(nodelist):
            vertex = self.G.node[node]
            if 'Y' in vertex:
                S[i] = np.sum(vertex['Y'])
                N[i] = len(vertex['Y'])
                del vertex['Y']

        def l_prox(lambd, eta, warm_start, pool):
            b = lambd * N - eta
            c = - lambd * S
            theta = (-b + np.sqrt(b**2 - 4 * c)) / 2.
            return np.maximum(theta, self.min_lambda)

        def r_prox(lambd, eta, warm_start, pool):
            return np.maximum(eta, self.min_lambda)

        data = G_to_data(self.G, theta_shape)

        result, info = fit_stratified_model(
            L, shape, l_prox, r_prox, data=data, **kwargs)

        transfer_result_to_G(result, self.G)

        return info

    def anll(self, Y, Z):
        return -np.mean(self.logpmf(Y, Z))

    def logpmf(self, Y, Z):
        Y = turn_into_iterable(Y)
        Z = turn_into_iterable(Z)
        parameter = np.array([self.G.node[z]['theta'][0] for z in Z])
        return poisson.logpmf(Y, mu=parameter)

    def sample(self, Z):
        Z = turn_into_iterable(Z)
        parameter = np.array([self.G.node[z]['theta'][0] for z in Z])
        return np.random.poisson(lam=parameter)


class LogisticRegression():
    """Stratified logistic regression."""

    def __init__(self, num_threads=2, lambd=0.01):
        self.lambd = lambd
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(num_threads)

    def fit(self, X, Y, Z, G, inplace=False, **kwargs):
        if inplace:
            self.G = G
        else:
            self.G = copy.deepcopy(G)

        # fit label encoder
        self.le = preprocessing.LabelEncoder()
        Y = self.le.fit_transform(Y).copy()
        num_classes = len(self.le.classes_)

        # calculate Laplacian matrix
        L = nx.laplacian_matrix(self.G)
        nodelist = self.G.nodes()
        K = L.shape[0]
        n = X.shape[1]
        shape = (n + 1, num_classes)  # intercept
        theta_shape = (K,) + shape

        # preprocess data
        for x, y, z in zip(X, Y, Z):
            vertex = self.G.node[z]
            if 'X' in vertex:
                vertex['X'] += [x]
                vertex['Y'] += [y]
            else:
                vertex['X'] = [x]
                vertex['Y'] = [y]

        XY_data = []
        for i, node in enumerate(nodelist):
            vertex = self.G.node[node]
            if 'Y' in vertex:
                X, Y = torch.tensor(vertex['X']), torch.tensor(vertex['Y'])
                X = torch.cat(
                    [X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)
                XY_data += [(X, Y)]
                del vertex['X']
                del vertex['Y']
            else:
                XY_data += [None]

        def l_prox(lambd, eta, warm_start, pool):
            res = pool.starmap(log_reg_prox, zip(
                XY_data, eta, warm_start, lambd * np.ones(K)))
            return np.array(res)

        def r_prox(lambd, eta, warm_start, pool):
            theta = eta.copy()
            theta[:, :-1, :] /= (1 + lambd * self.lambd)
            return eta

        data = G_to_data(self.G, theta_shape)

        result, info = fit_stratified_model(
            L, shape, l_prox, r_prox, data=data, **kwargs)

        transfer_result_to_G(result, self.G)

        return info

    def scores(self, X, Z):
        X = torch.from_numpy(X)
        X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)
        theta = torch.tensor(([self.G.node[z]['theta_tilde'] for z in Z]))
        scores = (X.unsqueeze(-1) * theta).sum(1)
        return scores

    def predict(self, X, Z, probs=False):
        scores = self.scores(X, Z)
        if probs:
            return torch.nn.Softmax(1)(scores).numpy()
        else:
            return self.le.inverse_transform(torch.argmax(scores, 1).numpy())

    def logprob(self, X, Y, Z):
        Y = torch.from_numpy(self.le.transform(Y))
        loss = torch.nn.CrossEntropyLoss(reduction='none')
        scores = self.scores(X, Z)
        l = loss(scores, Y)
        return -l.numpy()

    def anll(self, X, Y, Z):
        return -np.mean(self.logprob(X, Y, Z))


class RidgeRegression():
    """Stratified ridge regression"""

    def __init__(self, num_threads=2, lambd=0.01):
        self.lambd = lambd
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(num_threads)

    def fit(self, X, Y, Z, G, inplace=False, **kwargs):
        if inplace:
            self.G = G
        else:
            self.G = copy.deepcopy(G)

        # calculate Laplacian matrix
        L = nx.laplacian_matrix(self.G)
        nodelist = self.G.nodes()
        K = L.shape[0]
        N, n = X.shape
        _, m = Y.shape
        n = n + 1
        shape = (n, m)
        theta_shape = (K,) + shape

        # preprocess data
        for x, y, z in zip(X, Y, Z):
            vertex = self.G.node[z]
            if 'X' in vertex:
                vertex['X'] += [x]
                vertex['Y'] += [y]
            else:
                vertex['X'] = [x]
                vertex['Y'] = [y]

        XtX = torch.zeros(K, n, n).double()
        XtY = torch.zeros(K, n, m).double()
        for i, node in enumerate(nodelist):
            vertex = self.G.node[node]
            if 'Y' in vertex:
                X = torch.tensor(vertex['X']).double()
                Y = torch.tensor(vertex['Y']).double()
                X = torch.cat(
                    [X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)
                XtX[i] = X.t() @ X
                XtY[i] = X.t() @ Y
                del vertex['X']
                del vertex['Y']

        def l_prox(lambd, eta, warm_start, pool):
            A_LU = torch.btrifact(
                XtX + 1. / (2 * lambd) * torch.eye(n).unsqueeze(0).double())
            b = XtY + 1. / (2 * lambd) * torch.from_numpy(eta)
            x = torch.btrisolve(b, *A_LU)
            return x.numpy()

        def r_prox(lambd, eta, warm_start, pool):
            theta = eta.copy()
            theta[:, :, :-1] /= (1 + lambd * self.lambd)
            return eta

        data = G_to_data(self.G, theta_shape)

        result, info = fit_stratified_model(
            L, shape, l_prox, r_prox, data=data, **kwargs)

        transfer_result_to_G(result, self.G)

        return info

    def predict(self, X, Z):
        X = torch.from_numpy(X)
        X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)
        theta = torch.tensor(([self.G.node[z]['theta_tilde'] for z in Z]))
        return (X.unsqueeze(-1) * theta).sum(1).numpy()

    def score(self, X, Y, Z):
        predictions = self.predict(X, Z)
        return np.sqrt(np.mean((Y - predictions)**2))
