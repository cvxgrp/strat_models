import numpy as np
import torch
from sklearn import preprocessing
from scipy.stats import poisson, bernoulli
import cvxpy as cp
from scipy.optimize import minimize

class Loss:
	"""
	Inputs:
		N/A

	All losses have an attribute of isDistribution, which is a Boolean
	that denotes whether or not a Loss is a distribution estimate
	(i.e., isDistribution==True -> accepts Y,Z, and
		   isDistribution==False -> accepts X,Y,Z.)

	All losses implement the following functions:

	1. evaluate(theta, data). Evaluates the regularizer at theta with data.
	2. prox(t, nu, data, warm_start, pool): Evaluates the proximal operator of the regularizer at theta
	"""

	def __init__(self):
		pass

	def evaluate(self, theta):
		raise NotImplementedError("This method is not implemented for the parent class.")

	def setup(self, data, K):
		"""This function has any important setup required for the problem."""
		raise NotImplementedError("This method is not implemented for the parent class.")

	def prox(self, t, nu, data, warm_start, pool):
		raise NotImplementedError("This method is not implemented for the parent class.")

	def anll(self, data, G):
		return np.mean(self.logprob(data, G))

def turn_into_iterable(x):
    try:
        iter(x)
    except TypeError:
        return [x]
    else:
        return x

def mean_cov_prox_lbfgs(Y, eta, theta, t):
	if Y is None:
		return eta
	Y = Y[0]
	n,N = Y.shape

	ybar = np.mean(Y,1).reshape(-1,1)
	# Yemp = np.cov(Y)
	Yemp = (Y)@(Y).T/N

	T = eta[:,:-1]
	tt = eta[:, -1].reshape(-1,1)

	def loss_fcn(Snu):
		Snu = Snu.reshape(n, n+1)

		S = Snu[:,:-1]
		nu = Snu[:, -1].reshape(-1,1)

		if not np.all(np.linalg.eigvals(S) >= 0):
			return float(np.inf)

		eps = 1./(2*t)

		main_part = -N*np.linalg.slogdet(S)[1] + N*np.trace(S @ Yemp) - 2*N*ybar.T @ nu
		main_part += N*nu.T @ np.linalg.inv(S) @ nu

		prox_part = eps*np.linalg.norm(Snu - eta, "fro")**2

		# print(np.linalg.eigvals(Snu[:,:-1]))

		return float(main_part+prox_part)

	def jac_loss(Snu):
		Snu = Snu.reshape(n, n+1)
		S = Snu[:,:-1]
		nu = Snu[:, -1].reshape(-1,1)

		S_inv = np.linalg.inv(S)

		grad_S = N * (-S_inv + Yemp - S_inv @ nu @ nu.T @ S_inv) + (1/t)*(S-T)
		grad_nu = N * (-2*ybar + 2*S_inv@nu).reshape(-1,1) + (1/t)*(nu - tt)

		grad = np.hstack((grad_S, grad_nu.reshape(-1,1)))

		return grad.reshape(-1)

	if N < n: #if the number of samples is less than the number of dimensions:
		Semp = np.linalg.inv(Yemp+100*np.eye(n))
	else:
		Semp = np.linalg.inv(Yemp)

	Snu_0 = np.hstack((Semp, Semp @ ybar)).reshape(-1)
	options = dict(maxls=1000)
	res = minimize(fun=loss_fcn, x0=Snu_0, method="L-BFGS-B", jac=jac_loss, options=options)

	return res.x.reshape(n, n+1)

def solve_cvxpy(Y, eta, theta, t):
	if Y is None:
		return eta
	Y = Y[0]
	n,N = Y.shape
	ybar = np.mean(Y,1).reshape(-1,1)
	Yemp = Y @ Y.T / N

	S = cp.Variable((n,n))
	nu = cp.Variable((n,1))
	eps = (1./(2*t))

	main_part = -cp.log_det(S) 
	main_part += cp.trace(S@Yemp) 
	main_part += - 2*ybar.T@nu 
	main_part += cp.matrix_frac(nu, S) 

	prox_part = eps * cp.sum_squares(nu-eta[:, -1].reshape(-1,1))
	prox_part += eps * cp.norm(S-eta[:, :-1], "fro")**2
	prob = cp.Problem(cp.Minimize(N*main_part+prox_part))
	    
	prob.solve(verbose=False, warm_start=True)

	return np.hstack((S.value, nu.value))

def find_solution(x):
    """Finds the real solution to ax^3 + bx^2 + cx + d = 0."""
    roots = np.roots(x)
    for root in roots:
        if np.isreal(root) and root >= 1e-4 and root <= 1 - 1e-4:
            return np.real(root)
    return 0.5

def joint_cov_prox(Y, nu, theta, t):
	"""
	Proximal operator for joint covariance estimation
	"""
	if Y is None:
		return nu

	Y = Y[0]
	N = Y.shape[1]
	Yemp = Y
	# Yemp = Y@Y.T/N

	s, Q = np.linalg.eigh(Yemp - (1./t)*nu)
	w = (1./2)*(-t*s + np.sqrt((t*s)**2 + 4*t))

	return Q @ np.diag(w) @ Q.T

def log_reg_prox(XY, nu, theta, t):
	if XY is None:
		return nu

	X, Y = XY

	nu_tch = torch.from_numpy(nu)
	theta_i = torch.from_numpy(theta).requires_grad_(True)
	loss = torch.nn.CrossEntropyLoss(reduction="sum")
	optim = torch.optim.LBFGS([theta_i], lr=1, max_iter=50)

	def closure():
		optim.zero_grad()
		l = t * loss(X@theta_i, Y) + 0.5 * torch.sum((theta_i - nu_tch)**2)
		l.backward()
		return l

	optim.step(closure)
	return theta_i.data.numpy()

#### Losses
class sum_squares_loss(Loss):
	"""
	f(theta) = ||X @ theta - Y||_2^2
	"""
	def __init__(self, intercept=False):
		super().__init__()
		self.isDistribution = False
		self.intercept=intercept

	def evaluate(self, theta, data):
		assert 'X' in data and 'Y' in data
		return sum( (theta @ data['X'] - data['Y'])**2 )

	def setup(self, data, G):
		X = data['X']
		Y = data['Y']
		Z = data['Z']

		N, n = X.shape
		_, m = Y.shape

		if self.intercept:
			n = n+1

		K = len(G.nodes())

		shape = (n,m)
		theta_shape = (K,) + shape

		for x, y, z in zip(X, Y, Z):
			vertex = G.node[z]
			if 'X' in vertex:
				vertex['X'] += [x]
				vertex['Y'] += [y]
			else:
				vertex['X'] = [x]
				vertex['Y'] = [y]

		XtX = torch.zeros(K, n, n).double()
		XtY = torch.zeros(K, n, m).double()
		for i, node in enumerate(G.nodes()):
			vertex = G.node[node]
			if 'Y' in vertex:
				X = torch.tensor(vertex['X']).double()
				Y = torch.tensor(vertex['Y']).double()

				if self.intercept:
					X = torch.cat(
						[X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)

				XtX[i] = X.t() @ X
				XtY[i] = X.t() @ Y
				del vertex['X']
				del vertex['Y']

		cache = {'XtX':XtX, 'XtY':XtY, 'n':n, 'theta_shape':theta_shape, 'shape':shape}
		return cache


	def prox(self, t, nu, warm_start, pool, cache):
		# raise NotImplementedError("This method is not yet done!!!")

		XtX = cache['XtX']
		XtY = cache['XtY']
		n = cache['n']

		A_LU = torch.btrifact(
			XtX + 1. / (2 * t) * torch.eye(n).unsqueeze(0).double())
		b = XtY + 1. / (2 * t) * torch.from_numpy(nu)
		x = torch.btrisolve(b, *A_LU)

		return x.numpy()

	def predict(self, data, G):
		X = torch.from_numpy(data["X"])
		X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)
		theta = torch.tensor(([G.node[z]['theta_tilde'] for z in data["Z"]]))
		return (X.unsqueeze(-1) * theta).sum(1).numpy()

	def scores(self, data, G):
		predictions = self.predict(data, G)
		return np.sqrt(np.mean((data["Y"] - predictions)**2))

	def logprob(self, data, G):
		return self.scores(data, G)

class logistic_loss(Loss):
	"""
	f(theta) = sum[ log(1 + exp{-Y * theta @ X} )  ]
	"""
	def __init__(self, intercept=False):
		super().__init__()
		self.isDistribution = False
		self.intercept=intercept

	def evaluate(self, theta, data):
		assert "X" in data and "Y" in data
		return sum( np.log(1 + np.exp(-data["Y"] * theta @ data["X"])) )

	def setup(self, data, G):
		X = data["X"]
		Y = data["Y"]
		Z = data["Z"]

		self.le = preprocessing.LabelEncoder()
		Y = self.le.fit_transform(Y).copy()
		num_classes = len(self.le.classes_)

		K = len(G.nodes())
		n = X.shape[1]

		if self.intercept:
			n = n+1

		shape = (n, num_classes)
		theta_shape = (K,) + shape

		for x, y, z in zip(X, Y, Z):
			vertex = G.node[z]
			if 'X' in vertex:
				vertex['X'] += [x]
				vertex['Y'] += [y]
			else:
				vertex['X'] = [x]
				vertex['Y'] = [y]

		XY_data = []
		for i, node in enumerate(G.nodes()):
			vertex = G.node[node]
			if 'Y' in vertex:
				X, Y = torch.tensor(vertex['X']), torch.tensor(vertex['Y'])
				X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)
				XY_data += [(X, Y)]
				del vertex['X']
				del vertex['Y']
			else:
				XY_data += [None]

		cache = {"XY": XY_data, 'n':n, 'theta_shape':theta_shape, 'shape':shape, 'K':K}
		return cache

	def prox(self, t, nu, warm_start, pool, cache):
		res = pool.starmap(log_reg_prox, zip(cache["XY"], nu, warm_start, t * np.ones(cache["K"])))
		return np.array(res)

	def logprob(self, data, G):
		Y = torch.from_numpy(self.le.transform(data["Y"]))
		scores = self.scores(data, G)
		loss = torch.nn.CrossEntropyLoss(reduction="none")
		l = loss(scores, Y)
		return -l.numpy()

	def scores(self, data, G):
		X = torch.from_numpy(data["X"])
		X = torch.cat([X, torch.ones_like(X[:,0]).unsqueeze(1)],1)
		theta = torch.tensor(([G.node[z]['theta_tilde'] for z in data["Z"]]))
		scores = (X.unsqueeze(-1) * theta).sum(1)
		return scores

	def predict(self, data, G):
		probs=False
		scores = self.scores(data, G)
		if probs:
			return torch.nn.Softmax(1)(scores).numpy()
		else:
			return self.le.inverse_transform(torch.argmax(scores, 1).numpy())

class mean_covariance_max_likelihood_loss(Loss):
	"""
	f(theta) = Trace(S yy^T) - logdet(S) - 2y^T \nu + \nu^T S^{-1} \nu
	where theta = (S, \nu)
	"""
	def __init__(self):
		super().__init__()
		self.isDistribution = True

	def evaluate(self, theta, data):
		assert y in data
		y = data["y"]
		S, nu = theta[:, :-1], theta[:, -1].reshape(-1,1)
		return np.trace(S @ y @ y.T) - np.linalg.slogdet(S)[1] - 2*y.T@nu + nu.T @ np.linalg.inv(S) @ nu

	def setup(self, data, G):
		Y = data["Y"]
		Z = data["Z"]

		K = len(G.nodes())
		shape = (data["n"], data["n"]+1)
		theta_shape = (K,) + shape

		#preprocess data
		for y, z in zip(Y, Z):
			vertex = G.node[z]
			if "Y" in vertex:
				vertex["Y"] += [y]
			else:
				vertex['Y'] = [y]

		Y_data = []
		for i, node in enumerate(G.nodes()):
			vertex = G.node[node]
			if 'Y' in vertex:
				Y = vertex['Y']
				Y_data += [Y]
				del vertex['Y']
			else:
				Y_data += [None]

		cache = {"Y": Y_data, "n":data["n"], "theta_shape":theta_shape, "shape":shape, "K":K}
		return cache

	def prox(self, t, nu, warm_start, pool, cache):
		"""
		Proximal operator for joint mean-covariance estimation
		"""
		res = pool.starmap(mean_cov_prox_lbfgs, zip(cache["Y"], nu, warm_start, t*np.ones(cache["K"])))
		return np.array(res)

	def logprob(self, data, G):
		Y = data["Y"]

		N = data["Y"][1].shape[0]

		thetas = [G.node[z]["theta"] for z in data["Z"]]
		S = [theta[:,:-1] for theta in thetas]
		nu = [theta[:,-1].reshape(-1,1) for theta in thetas]

		logprobs = [np.trace(S[i] @ Y[i] @ Y[i].T) - N*np.linalg.slogdet(S[i])[1] 
						- 2*N*(np.mean(Y[i],1).reshape(-1,1)).T@nu[i] 
						+ N*nu[i].T @ np.linalg.inv(S[i]) @ nu[i] 
							for i in range(len(thetas))]
		return logprobs

	def sample(self, data, G):
		Z = turn_into_iterable(data["Z"])
		thetas = [G.node[z]["theta"] for z in data["Z"]]

		sigmas = [np.linalg.inv(theta[:,:-1]) for theta in thetas]
		mus = [sigmas[i] @ thetas[i][:,-1] for i in range(len(sigmas))]

		return [np.random.multivariate_normal(mus[i], sigmas[i]) for i in range(len(sigmas))]
		

class covariance_max_likelihood_loss(Loss):
	"""
	f(theta) = Trace(theta @ Y) - logdet(theta)
	"""
	def __init__(self):
		super().__init__()
		self.isDistribution = True

	def evaluate(self, theta, data):
		assert "Y" in data
		return np.trace(theta @ data["Y"]) - np.linalg.slogdet(theta)[1]

	def setup(self, data, G):
		Y = data["Y"]
		Z = data["Z"]

		K = len(G.nodes())

		shape = (data["n"], data["n"])
		theta_shape = (K,) + shape

		#preprocess data
		for y, z in zip(Y, Z):
			vertex = G.node[z]
			if "Y" in vertex:
				vertex["Y"] += [y]
			else:
				vertex["Y"] = [y]

		Y_data = []
		for i, node in enumerate(G.nodes()):
			vertex = G.node[node]
			if 'Y' in vertex:
				Y = vertex['Y']
				Y_data += [Y]
				del vertex['Y']
			else:
				Y_data += [None]

		cache = {"Y": Y_data, "n":data["n"], "theta_shape":theta_shape, "shape":shape, "K":K}
		return cache

	def prox(self, t, nu, warm_start, pool, cache):
		"""
		Proximal operator for joint covariance estimation
		"""
		res = pool.starmap(joint_cov_prox, zip(cache["Y"], nu, warm_start, t*np.ones(cache["K"])))
		return np.array(res)

	def logprob(self, data, G):
		Y = data["Y"]
		thetas = [G.node[z]["theta"] for z in data["Z"]]
		logprobs = [np.trace(Y[i]@thetas[i]) - np.linalg.slogdet(thetas[i])[1] for i in range(len(thetas))]
		return logprobs

	def sample(self, data, G):
		Z = turn_into_iterable(data["Z"])
		sigmas = [np.linalg.inv(G.node[z]["theta"]) for z in Z]

		n = sigmas[0].shape[0]
		return [np.random.multivariate_normal(np.zeros(n), sigma) for sigma in sigmas]

class poisson_loss(Loss):
	"""
	f(theta) = N*theta - log(theta)*sum(Y), 
	for y in integers_+^N and theta > 0
	"""
	def __init__(self, min_theta=1e-5):
		super().__init__()
		self.isDistribution = True
		
		#cannot allow theta to be exactly equal to 0 for rounding errors
		self.min_theta = min_theta

	def evaluate(self, theta, data):
		y = data["Y"]
		N = len(data["Y"])
		return float(N*theta - np.log(theta)*sum(y))

	def setup(self, data, G):
		Y = data["Y"]
		Z = data["Z"]

		K = len(G.nodes())

		shape = (1,)
		theta_shape = (K,) + shape

		#preprocess data
		for y,z in zip(Y,Z):
			vertex = G.node[z]
			if "Y" in vertex:
				vertex["Y"] += [y]
			else:
				vertex["Y"] = [y]

		S = np.zeros((K,1))
		N = np.zeros((K,1))

		for i, node in enumerate(G.nodes()):
			vertex = G.node[node]
			if "Y" in vertex:
				S[i] = np.sum(vertex["Y"])
				N[i] = len(vertex["Y"])

		cache = {"S": S, "N":N, "theta_shape":theta_shape, "shape":shape, "K":K}
		return cache

	def prox(self, t, nu, warm_start, pool, cache):
		S = cache["S"]
		N = cache["N"]
		b = t*N - nu
		c = -t * S

		theta = (-b + np.sqrt(b**2 - 4*c)) / 2.

		return np.maximum(theta, self.min_theta)

	def logprob(self, data, G):
		Y = turn_into_iterable(data["Y"])
		Z = turn_into_iterable(data["Z"])
		parameter = [G.node[z]["theta"][0] for z in Z]
		return poisson.logpmf(Y, mu=parameter)

	def sample(self, data, G):
		Z = turn_into_iterable(data["Z"])
		parameter = [G.node[z]["theta"][0] for z in Z]
		return poisson.rvs(mu=parameter)

class bernoulli_loss(Loss):
	"""
	f(theta) = -sum(y)log(theta) - (n - sum(y))log(1-theta),
	where y in reals^n and theta in [0,1].
	"""

	def __init__(self, min_theta=1e-5, max_theta = 1-1e-5):
		super().__init__()
		self.isDistribution = True
		self.min_theta = min_theta
		self.max_theta = max_theta

	def evaluate(self, theta, data):
		return 0

	def setup(self, data, G):
		Y = data["Y"]
		Z = data["Z"]

		K = len(G.nodes())

		shape = (1,)
		theta_shape = (K,) + shape

		#preprocess data
		for y, z in zip(Y,Z):
			vertex = G.node[z]
			if "Y" in vertex:
				vertex["Y"] += [y]
			else:
				vertex["Y"] = [y]

		S = np.zeros((K,1))
		N = np.zeros((K,1))

		for i, node in enumerate(G.nodes()):
			vertex = G.node[node]
			if 'Y' in vertex:
				S[i] = np.sum(vertex['Y'])
				N[i] = len(vertex['Y'])
				del vertex['Y']

		cache = {"S": S, "N":N, "theta_shape":theta_shape, "shape":shape, "K":K}
		return cache

	def prox(self, t, nu, warm_start, pool, cache):
		S = cache["S"]
		N = cache["N"]

		a = -1*np.ones(nu.shape)
		b = (1+nu)
		c = t*N - nu
		d = -t * S

		coefs = np.hstack([a,b,c,d])
		theta = np.array(pool.map(find_solution, coefs))[:, np.newaxis]

		return np.clip(theta, self.min_theta, self.max_theta)

	def logprob(self, data, G):
		Y = turn_into_iterable(data["Y"])
		Z = turn_into_iterable(data["Z"])
		parameter = [G.node[z]["theta"][0] for z in Z]
		return bernoulli.logpmf(Y, p=parameter)

	def sample(self, data, G):
		Z = turn_into_iterable(data["Z"])
		parameter = [G.node[z]["theta"][0] for z in Z]
		return bernoulli.rvs(p=parameter)
