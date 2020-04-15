import numpy as np
import torch
from sklearn import preprocessing
from scipy.stats import poisson, bernoulli, multinomial

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
		return -np.mean(self.logprob(data, G))

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

def joint_cov_prox(Y, nu, theta, t):
    """
    Proximal operator for joint covariance estimation
    """
    if Y is None:
        return nu

    n, nk = Y[0].shape
    Yemp = Y[0]@Y[0].T/nk
    
    s, Q = np.linalg.eigh(nu/(t*nk)-Yemp)
    w = ((t*nk)*s + np.sqrt(((t*nk)*s)**2 + 4*(t*nk)))/2
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

def nonparametric_discrete_prox(Y, nu, theta, t):
	if Y is None:
		return nu
	nu_tch = torch.from_numpy(nu)
	theta_i = torch.from_numpy(theta).requires_grad_(True)
	Y_tch = torch.from_numpy(Y).type(torch.float64)
	optim = torch.optim.LBFGS([theta_i], lr=1, max_iter=60)

	def closure():
		optim.zero_grad()
		loss = torch.log(torch.sum(torch.exp(theta_i))) * torch.sum(Y_tch) - Y_tch@theta_i
		loss += torch.sum((theta_i - nu_tch)**2) / (2*t)
		loss.backward()
		return loss

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

		if X.ndim == 1:
			X = X.reshape(-1,1)

		N, n = X.shape
		_, m = Y.shape

		if self.intercept:
			n = n+1

		K = len(G.nodes())

		shape = (n,m)
		theta_shape = (K,) + shape

		for x, y, z in zip(X, Y, Z):
			vertex = G._node[z]
			if 'X' in vertex:
				vertex['X'] += [x]
				vertex['Y'] += [y]
			else:
				vertex['X'] = [x]
				vertex['Y'] = [y]

		XtX = torch.zeros(K, n, n).double()
		XtY = torch.zeros(K, n, m).double()
		for i, node in enumerate(G.nodes()):
			vertex = G._node[node]
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

		A_LU = torch.lu(
			XtX + 1. / (2 * t) * torch.eye(n).unsqueeze(0).double())
		b = XtY + 1. / (2 * t) * torch.from_numpy(nu)
		x = torch.lu_solve(b, *A_LU)

		return x.numpy()

	def predict(self, data, G):
		X = torch.from_numpy(data["X"])

		if self.intercept:
			X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)

		theta = torch.tensor(([G._node[z]['theta_tilde'] for z in data["Z"]]))
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
			vertex = G._node[z]
			if 'X' in vertex:
				vertex['X'] += [x]
				vertex['Y'] += [y]
			else:
				vertex['X'] = [x]
				vertex['Y'] = [y]

		XY_data = []
		for i, node in enumerate(G.nodes()):
			vertex = G._node[node]
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
		theta = torch.tensor(([G._node[z]['theta_tilde'] for z in data["Z"]]))
		scores = (X.unsqueeze(-1) * theta).sum(1)
		return scores

	def predict(self, data, G):
		probs=False
		scores = self.scores(data, G)
		if probs:
			return torch.nn.Softmax(1)(scores).numpy()
		else:
			return self.le.inverse_transform(torch.argmax(scores, 1).numpy())

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
            vertex = G._node[z]
            if "Y" in vertex:
                vertex["Y"] += [y]
            else:
                vertex["Y"] = [y]

        Y_data = []
        for i, node in enumerate(G.nodes()):
            vertex = G._node[node]
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
        
        logprobs = []
        
        for y,z in zip(data["Y"], data["Z"]):
            n, nk = y.shape
            Y = (y@y.T)/nk
            
            if (np.zeros((n,n)) == Y).all():
                continue            
            
            theta = G._node[z]["theta_tilde"]
            logprobs += [np.linalg.slogdet(theta)[1] - np.trace(Y@theta)]

        return logprobs

    def sample(self, data, G):
        Z = turn_into_iterable(data["Z"])
        sigmas = [np.linalg.inv(G._node[z]["theta"]) for z in Z]

        n = sigmas[0].shape[0]
        return [np.random.multivariate_normal(np.zeros(n), sigma) for sigma in sigmas]

class nonparametric_discrete_loss(Loss):
	def __init__(self):
		super().__init__()
		self.isDistribution = True

	def evaluate(self, theta, data):
		return 0

	def setup(self, data, G):
		Y = data["Y"]
		Z = data["Z"]

		self.le = preprocessing.LabelEncoder()
		Y = self.le.fit_transform(Y).copy()
		num_classes = len(self.le.classes_)

		K = len(G.nodes())

		shape = (num_classes,)
		theta_shape = (K,) + shape

		for y, z in zip(Y, Z):
			vertex = G._node[z]
			if "Y" not in vertex:
				vertex['Y'] = np.zeros(num_classes)
			vertex['Y'][y] += 1		

		Y_data = []
		counts = np.zeros((K, num_classes))
		for i, node in enumerate(G.nodes()):
			vertex = G._node[node]
			if 'Y' in vertex:
				Y_data += [vertex['Y']]
				del vertex['Y']
			else:
				Y_data += [np.zeros(num_classes)]

		cache = {"Y": Y_data, 'num_classes':num_classes, 'theta_shape':theta_shape, 'shape':shape, 'K':K}
		return cache

	def prox(self, t, nu, warm_start, pool, cache):
		res = pool.starmap(nonparametric_discrete_prox, zip(cache["Y"], nu, warm_start, t * np.ones(cache["K"])))
		return np.array(res)

	def logprob(self, data, G):
		Y = turn_into_iterable(self.le.transform(data["Y"]))
		Z = turn_into_iterable(data["Z"])

		nodes = {}
		for y,z in zip(Y,Z):
			if z not in nodes.keys():
				nodes[z] = np.zeros((G._node[z]["theta"]).shape)
			nodes[z][y] += 1

		dim = int((G._node[z]["theta"]).shape[0])

		logprobs = []
		for z in nodes.keys():
			theta = torch.from_numpy(G._node[z]["theta"])
			Y_tch = torch.from_numpy(nodes[z]).type(torch.float64)
			loss = (Y_tch@theta - torch.log(torch.sum(torch.exp(theta))) * torch.sum(Y_tch)).numpy()
			logprobs += [loss/dim]
		return logprobs

	def sample(self, data, G):
		Z = turn_into_iterable(data["Z"])
		parameter = [G._node[z]["theta"] for z in Z]
		return multinomial.rvs(n=1, p=parameter)

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
			vertex = G._node[z]
			if "Y" in vertex:
				vertex["Y"] += [y]
			else:
				vertex["Y"] = [y]

		S = np.zeros((K,1))
		N = np.zeros((K,1))

		for i, node in enumerate(G.nodes()):
			vertex = G._node[node]
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
		parameter = [G._node[z]["theta"][0] for z in Z]
		return poisson.logpmf(Y, mu=parameter)

	def sample(self, data, G):
		Z = turn_into_iterable(data["Z"])
		parameter = [G._node[z]["theta"][0] for z in Z]
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
			vertex = G._node[z]
			if "Y" in vertex:
				vertex["Y"] += [y]
			else:
				vertex["Y"] = [y]

		S = np.zeros((K,1))
		N = np.zeros((K,1))

		for i, node in enumerate(G.nodes()):
			vertex = G._node[node]
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
		parameter = [G._node[z]["theta"][0] for z in Z]
		return bernoulli.logpmf(Y, p=parameter)

	def sample(self, data, G):
		Z = turn_into_iterable(data["Z"])
		parameter = [G._node[z]["theta"][0] for z in Z]
		return bernoulli.rvs(p=parameter)
