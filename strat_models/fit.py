import networkx as nx
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

import warnings
import multiprocessing as mp
import time

def fit_stratified_model(L, shape, l_prox, r_prox, G_data=dict(), abs_tol=1e-3, rel_tol=1e-3,
                         rho=1, mu=10, tau_incr=2, tau_decr=2, max_rho=1e1, min_rho=1e-1,
                         maxiter=100, verbose=False, n_jobs=1, max_cg_iterations=10):
    """Fits a stratified model using ADMM, as described in the paper
    `A Distributed Method for Fitting Laplacian Regularized Stratified Models`.
    Args:
        - L: Laplacian matrix, represented as scipy sparse matrix.
        - shape: shape of an individual theta.
        - l_prox: Applies l proximal operator in batch.
        - r_prox: Applies r proximal operator in batch.
        - G_data (optional): dictionary of warm starting values. (default=dict())
        - abs_tol (optional): Absolute tolerance. (default=1e-3)
        - rel_tol (optional): Relative tolerance. (default=1e-3)
        - rho (optional): Initial penalty parameter. (default=1.0)
        - mu/tau_incr/tau_decr/max_rho/min_rho (optional): Adaptive penalty parameters.
        - maxiter (optional): Maximum number of ADMM iterations. (default=100)
        - verbose (optional): True to print status messages, False to be silent. (default=False)
        - n_jobs (optional): number of jobs to spawn. (default=1)
        - max_cg_iterations (optional): Max number of CG iterations per ADMM iteration. (defulat=10)
    Returns:
        - result: Dictionary with the solution vectors.
        - info: Information about the algorithm's performance.
    """
    K = L.shape[0]
    n = np.prod(shape)

    # Initialization
    if 'theta_init' in G_data:
        theta = G_data['theta_init'].copy()
    else:
        theta = np.zeros((K,) + shape)
    if 'theta_tilde' in G_data:
        theta_tilde = G_data['theta_tilde'].copy()
    else:
        theta_tilde = theta.copy()
    if 'theta_hat' in G_data:
        theta_hat = G_data['theta_hat'].copy()
    else:
        theta_hat = theta.copy()
    if 'u' in G_data:
        u = G_data['u'].copy()
    else:
        u = np.zeros(theta.shape)
    if 'u_tilde' in G_data:
        u_tilde = G_data['u_tilde'].copy()
    else:
        u_tilde = np.zeros(theta.shape)

    res_pri = np.zeros(theta.shape)
    res_pri_tilde = np.zeros(theta.shape)
    res_dual = np.zeros(theta.shape)
    res_dual_tilde = np.zeros(theta.shape)

    optimal = False
    n_jobs = n_jobs if K > n_jobs else K
    prox_pool = mp.Pool(n_jobs)

    if verbose:
        print("%3s | %10s %10s %10s %10s %6s %6s %6s %6s" %
              ("it", "s_norm", "r_norm", "eps_pri", "eps_dual", "rho", "time1", "time2", "time3"))

    # Main ADMM loop
    start_time = time.perf_counter()
    for t in range(1, maxiter + 1):

        # theta update
        start_time_1 = time.perf_counter()
        theta = l_prox(1. / rho, theta_hat - u, theta, prox_pool)
        time_1 = time.perf_counter() - start_time_1

        # theta_tilde update
        start_time_2 = time.perf_counter()
        theta_tilde = r_prox(1. / rho, theta_hat -
                             u_tilde, theta_tilde, prox_pool)
        time_2 = time.perf_counter() - start_time_2

        # theta_hat update
        start_time_3 = time.perf_counter()
        sys = L + 2 * rho * sparse.eye(K)
        M = sparse.diags(1. / sys.diagonal())
        indices = np.ndindex(shape)
        rhs = rho * (theta.T + u.T + theta_tilde.T + u_tilde.T)
        for i, ind in enumerate(indices):
            index = ind[::-1]
            sol = splinalg.cg(sys, rhs[index], M=M,
                              x0=theta_hat.T[index], maxiter=max_cg_iterations)[0]
            res_dual.T[index] = -rho * (sol - theta_hat.T[index])
            res_dual_tilde.T[index] = res_dual.T[index]
            theta_hat.T[index] = sol
        time_3 = time.perf_counter() - start_time_3

        # u and u_tilde update
        res_pri = theta - theta_hat
        res_pri_tilde = theta_tilde - theta_hat
        u += theta - theta_hat
        u_tilde += theta_tilde - theta_hat

        # calculate residual norms
        res_pri_norm = np.linalg.norm(np.append(res_pri, res_pri_tilde))
        res_dual_norm = np.linalg.norm(np.append(res_dual, res_dual_tilde))

        eps_pri = np.sqrt(2 * K * np.prod(shape)) * abs_tol + \
            rel_tol * max(res_pri_norm, res_dual_norm)
        eps_dual = np.sqrt(2 * K * np.prod(shape)) * abs_tol + \
            rel_tol * np.linalg.norm(rho * np.append(u, u_tilde))

        if verbose:
            print("%3d | %8.4e %8.4e %8.4e %8.4e %4.3f %4.3f %4.3f %4.3f" %
                  (t, res_pri_norm, res_dual_norm, eps_pri, eps_dual, rho,
                   time_1 * 1000, time_2 * 1000, time_3 * 1000))

        # check stopping condition
        if res_pri_norm <= eps_pri and res_dual_norm <= eps_dual:
            optimal = True
            break

        # penalty parameter update
        new_rho = rho
        if res_pri_norm > mu * res_dual_norm:
            new_rho = tau_incr * rho
        elif res_dual_norm > mu * res_pri_norm:
            new_rho = rho / tau_decr
        new_rho = np.clip(new_rho, min_rho, max_rho)
        u *= rho / new_rho
        u_tilde *= rho / new_rho
        rho = new_rho
    main_loop_time = time.perf_counter() - start_time

    # clean up the multiprocessing pool
    prox_pool.close()
    prox_pool.join()

    if verbose:
        if optimal:
            print(f"Terminated (optimal) in {t} iterations.")
        else:
            print("Terminated (reached max iterations).")
        print("run time: %8.4e seconds" % main_loop_time)

    # construct result
    result = {
        'theta': theta,
        'theta_tilde': theta_tilde,
        'theta_hat': theta_hat,
        'u': u,
        'u_tilde': u_tilde
    }

    info = {
        'time': main_loop_time,
        'iterations': t,
        'optimal': optimal
    }

    return result, info

def fit_eigen_stratified_model(Q_tilde, eigvals, shape, l_prox, r_prox, G_data=dict(), abs_tol=1e-3, rel_tol=1e-3,
                         rho=1, mu=10, tau_incr=2, tau_decr=2, max_rho=1e1, min_rho=1e-1,
                         maxiter=100, verbose=False, n_jobs=1, max_cg_iterations=10,
                         num_eigenvectors=None):
    """Fits a stratified model using ADMM, as described in the paper
    `A Distributed Method for Fitting Laplacian Regularized Stratified Models`.
    Args:
        - L: Laplacian matrix, represented as scipy sparse matrix.
        - shape: shape of an individual theta.
        - l_prox: Applies l proximal operator in batch.
        - r_prox: Applies r proximal operator in batch.
        - G_data (optional): dictionary of warm starting values. (default=dict())
        - abs_tol (optional): Absolute tolerance. (default=1e-3)
        - rel_tol (optional): Relative tolerance. (default=1e-3)
        - rho (optional): Initial penalty parameter. (default=1.0)
        - mu/tau_incr/tau_decr/max_rho/min_rho (optional): Adaptive penalty parameters.
        - maxiter (optional): Maximum number of ADMM iterations. (default=100)
        - verbose (optional): True to print status messages, False to be silent. (default=False)
        - n_jobs (optional): number of jobs to spawn. (default=1)
        - max_cg_iterations (optional): Max number of CG iterations per ADMM iteration. (default=10)
        - num_eigenvectors (optional): Number of eigenvectors to estimate model params (default=None=use all of them)
    Returns:
        - result: Dictionary with the solution vectors.
        - info: Information about the algorithm's performance.
    """
    K = Q_tilde.shape[0]
    n = np.prod(shape)

    # Initialization
    if 'theta_init' in G_data:
        theta = G_data['theta_init'].copy()
    else:
        np.zeros((K,) + shape)
    if 'theta_tilde' in G_data:
        theta_tilde = G_data['theta_tilde'].copy()
    else:
        theta_tilde = theta.copy()
    if 'Z_init' in G_data:
        Z = G_data['Z_init'].copy()
    else:
        Z = np.zeros(shape + (num_eigenvectors,))
    if 'u' in G_data:
        u = G_data['u'].copy()
    else:
        u = np.zeros(theta.shape)
    if 'u_tilde' in G_data:
        u_tilde = G_data['u_tilde'].copy()
    else:
        u_tilde = np.zeros(theta.shape)

    res_pri = np.zeros(theta.shape)
    res_pri_tilde = np.zeros(theta.shape)
    res_dual = np.zeros(theta.shape)
    res_dual_tilde = np.zeros(theta.shape)

    optimal = False
    n_jobs = n_jobs if K > n_jobs else K
    prox_pool = mp.Pool(n_jobs)

    if verbose:
        print("%3s | %10s %10s %10s %10s %6s %6s %6s %6s" %
              ("it", "s_norm", "r_norm", "eps_pri", "eps_dual", "rho", "time1", "time2", "time3"))

    # Main ADMM loop
    start_time = time.perf_counter()
    for t in range(1, maxiter + 1):

        # theta update
        start_time_1 = time.perf_counter()
        theta = l_prox(1. / rho, theta_tilde - u, theta, prox_pool)
        time_1 = time.perf_counter() - start_time_1

        #Z update
        start_time_2 = time.perf_counter()

        Z = rho * (u.T + theta_tilde.T) @ Q_tilde * (1/(eigvals+rho))
        time_2 = time.perf_counter() - start_time_2

        # theta_tilde update
        start_time_3 = time.perf_counter()

        ZQ_tilde_T = (Z@Q_tilde.T).T

        theta_tilde = r_prox(1. / rho, ZQ_tilde_T -
                             u_tilde, theta_tilde, prox_pool)
        time_3 = time.perf_counter() - start_time_3

        # u and u_tilde update
        res_pri = theta - theta_tilde
        res_pri_tilde = theta_tilde - ZQ_tilde_T
        u += theta - theta_tilde
        u_tilde += theta_tilde - ZQ_tilde_T

        # calculate residual norms
        res_pri_norm = np.linalg.norm(np.append(res_pri, res_pri_tilde))
        res_dual_norm = np.linalg.norm(np.append(res_dual, res_dual_tilde))

        eps_pri = np.sqrt(2 * K * np.prod(shape)) * abs_tol + \
            rel_tol * max(res_pri_norm, res_dual_norm)
        eps_dual = np.sqrt(2 * K * np.prod(shape)) * abs_tol + \
            rel_tol * np.linalg.norm(rho * np.append(u, u_tilde))

        if verbose:
            print("%3d | %8.4e %8.4e %8.4e %8.4e %4.3f %4.3f %4.3f %4.3f" %
                  (t, res_pri_norm, res_dual_norm, eps_pri, eps_dual, rho,
                   time_1 * 1000, time_2 * 1000, time_3 * 1000))

        # check stopping condition
        if res_pri_norm <= eps_pri and res_dual_norm <= eps_dual:
            optimal = True
            break

        # penalty parameter update
        new_rho = rho
        if res_pri_norm > mu * res_dual_norm:
            new_rho = tau_incr * rho
        elif res_dual_norm > mu * res_pri_norm:
            new_rho = rho / tau_decr
        new_rho = np.clip(new_rho, min_rho, max_rho)
        u *= rho / new_rho
        u_tilde *= rho / new_rho
        rho = new_rho
    main_loop_time = time.perf_counter() - start_time

    # clean up the multiprocessing pool
    prox_pool.close()
    prox_pool.join()

    if verbose:
        if optimal:
            print(f"Terminated (optimal) in {t} iterations.")
        else:
            print("Terminated (reached max iterations).")
        print("run time: %8.4e seconds" % main_loop_time)

    # construct result
    result = {
        'theta': theta,
        'theta_tilde': theta_tilde,
        'Z': Z,
        'u': u,
        'u_tilde': u_tilde,
        'Q_tilde': Q_tilde
    }

    info = {
        'time': main_loop_time,
        'iterations': t,
        'optimal': optimal
    }

    return result, info