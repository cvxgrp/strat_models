import numpy as np

class Regularizer:
	"""
	Inputs:
		lambd (scalar > 0): regularization coefficient. Default value is 1.

	All regularizers implement the following functions:

	1. evaluate(theta). Evaluates the regularizer at theta.
	2. prox(t, nu, warm_start, pool): Evaluates the proximal operator of the regularizer at theta
	"""

	def __init__(self, lambd=1):
		if lambd < 0:
			raise ValueError("Regularization coefficient must be a nonnegative scalar.")

		self.lambd = lambd

	def evaluate(self, theta):
		raise NotImplementedError("This method is not implemented for the parent class.")

	def prox(self, t, nu, warm_start, pool):
		raise NotImplementedError("This method is not implemented for the parent class.")

#### Regularizers
class zero_reg(Regularizer):
	def __init__(self, lambd=0):
		super().__init__(lambd)
		self.lambd=lambd

	def evaluate(self, theta):
		return 0

	def prox(self, t, nu, warm_start, pool):
		return nu

class sum_squares_reg(Regularizer):
	def __init__(self, lambd=1):
		super().__init__(lambd)
		self.lambd=lambd

	def evaluate(self, theta):
		return (self.lambd/2)*sum(theta**2)

	def prox(self, t, nu, warm_start, pool):
		if self.lambd == 0:
			return nu
		return nu / (1+t*self.lambd)

class trace_reg(Regularizer):
	def __init__(self, lambd=1):
		super().__init__(lambd)
		self.lambd = lambd

	def evaluate(self, theta):
		return np.trace(theta)

	def prox(self, t, nu, warm_start, pool):
		#nu of shape K, n, n
		res = np.zeros(nu.shape)
		for k in range(nu.shape[0]):
			res[k, :, :] = nu[k, :, :] - self.lambd*t*np.eye(nu.shape[1])
		return res

class mtx_scaled_sum_squares_reg(Regularizer):
	"""
	r(theta) = (lambd/2) * || A @ theta ||^2
	"""
	def __init__(self, A, lambd=1):
		super().__init__(lambd)
		self.lambd=lambd
		self.A = A
		self.AtA = A.T @ A

	def evaluate(self, theta):
		if self.A.shape[1] != theta.shape[0]:
			raise AssertionError("Dimension of scaling matrix is incompatible with dimension of vector")
		return (self.lambd/2) * sum( (self.A @ theta)**2 )

	def prox(self, t, nu, warm_start, pool):
		if self.lambd == 0:
			return nu

		K, n = nu.shape

		inv_mtx = np.linalg.inv(np.eye(n) + t*self.lambd*self.AtA)

		return nu@inv_mtx.T

class mtx_scaled_plus_sum_squares_reg(Regularizer):
	"""
	r(theta) = (lambd/2) * (|| A @ theta ||^2 + ||theta||^2)
	"""
	def __init__(self, A, lambd=1):
		super().__init__(lambd)
		self.lambd=lambd
		self.A = A
		self.AtA = A.T @ A

	def evaluate(self, theta):
		if self.A.shape[1] != theta.shape[0]:
			raise AssertionError("Dimension of scaling matrix is incompatible with dimension of vector")
		return (self.lambd/2) * ( sum( (self.A @ theta)**2 ) + sum(theta**2) )

	def prox(self, t, nu, warm_start, pool):
		if self.lambd == 0:
			return nu

		K, n = nu.shape

		inv_mtx = np.linalg.inv((1+t*self.lambd)*np.eye(n) + t*self.lambd*self.AtA)

		return nu@inv_mtx.T

class scaled_plus_sum_squares_reg(Regularizer):
	"""
	r(theta) = (1/2) * (lambd_1 * || A @ theta ||^2 + lambd_2 * ||theta||^2)
	"""
	def __init__(self, A, lambd=(1,1)):
		super().__init__(lambd)
		self.lambd=lambd
		self.A = A
		self.AtA = A.T @ A

	def evaluate(self, theta):
		if self.A.shape[1] != theta.shape[0]:
			raise AssertionError("Dimension of scaling matrix is incompatible with dimension of vector")
		return (self.lambd[0]/2) * sum( (self.A @ theta)**2 ) + (self.lambd[1]/2) * sum( theta**2 ) 

	def prox(self, t, nu, warm_start, pool):
		if self.lambd == 0:
			return nu

		K, n = nu.shape

		inv_mtx = np.linalg.inv((1+t*self.lambd[1])*np.eye(n) + t*self.lambd[0]*self.AtA)

		return nu@inv_mtx.T

class L1_reg(Regularizer):
	def __init__(self, lambd=1):
		super().__init__(lambd)

	def evaluate(self, theta):
		return self.lambd*sum(abs(theta))

	def prox(self, t, nu, warm_start, pool):
		return np.maximum(nu - t*self.lambd, 0) - np.maximum(-nu - t*self.lambd, 0)

class L2_reg(Regularizer):
	def __init__(self, lambd=1):
		super().__init__(lambd)

	def evaluate(self, theta):
		return self.lambd*np.linalg.norm(theta,2)

	def prox(self, t, nu, warm_start, pool):
		nus = []
		for i in range(nu.shape[0]):
			nus += [nu[i] * np.maximum(1 - t*self.lambd / np.linalg.norm(nu[i], 2), 0)]
		return np.rollaxis(np.dstack(nus),-1)

class elastic_net_reg(Regularizer):
	def __init__(self, lambd=1):
		super().__init__(lambd)

	def evaluate(self, theta):
		return sum(abs(theta)) + (self.lambd/2)*sum(theta**2)

	def prox(self, t, nu, warm_start, pool):
		return (1/(1+t*self.lambd)) * np.maximum(nu - t, 0) - np.maximum(-nu - t, 0)

class neg_log_reg(Regularizer):
	def __init__(self, lambd=1):
		super().__init__(lambd)

	def evaluate(self, theta):

		if theta < 0:
			return np.inf

		return -self.lambd*np.log(theta)

	def prox(self, t, nu, warm_start, pool):
		return (nu + np.sqrt(nu**2 + 4*t*self.lambd))/2

class nonnegative_reg(Regularizer):
	def __init__(self, lambd=1):
		super().__init__(lambd)

	def evaluate(self, theta):
		for theta_i in theta:
			if theta_i < 0:
				return np.inf
		return 0

	def prox(self, t, nu, warm_start, pool):
		nu[nu < 0] = 0
		return nu

class simplex_reg(Regularizer):
	def __init__(self, lambd=None):
		super().__init__(lambd)

	def evaluate(self, theta):
		if theta >= 0:
			if abs(sum(theta) - 1) <= 1e-4:
				return 0
		return np.inf

	def prox(self, t, nu, warm_start, pool):
		new_nu = np.zeros(nu.shape)
		for i in range(nu.shape[0]):
			new_nu[i,:] = project_onto_simplex(nu[i,:])
		return new_nu

class min_threshold_reg_one_elem(Regularizer):
	def __init__(self, lambd=1e-5):
		super().__init__(lambd)

	def evaluate(self, theta):
		if theta < self.lambd:
			return np.inf
		return 0

	def prox(self, t, nu, warm_start, pool):
		return np.maximum(nu, self.lambd)

class clip_reg(Regularizer):
	def __init__(self, lambd=(1e-5,1-1e-5)):
		super().__init__(lambd)

	def evaluate(self, theta):
		if theta > self.lambd[1] or theta < self.lambd[0]:
			return np.inf
		return 0

	def prox(self, t, nu, warm_start, pool):
		return np.clip(nu, self.lambd[0], self.lambd[1])


########## Utility Functions ##########

def project_onto_simplex(y):
 
	a = np.ones(len(y))
	l = y
	idx = np.argsort(l)
	d = len(l)
 
	evalpL = lambda k: np.sum(a[idx[k:]]*(y[idx[k:]] - l[idx[k]]*a[idx[k:]]) ) -1
 
 
	def bisectsearch():
		idxL, idxH = 0, d-1
		L = evalpL(idxL)
		H = evalpL(idxH)
 
		if L<0:
			return idxL
 
		while (idxH-idxL)>1:
			iMid = int((idxL+idxH)/2)
			M = evalpL(iMid)
 
			if M>0:
				idxL, L = iMid, M
			else:
				idxH, H = iMid, M
 
		return idxH
 
	k = bisectsearch()
	lam = (np.sum(a[idx[k:]]*y[idx[k:]])-1)/np.sum(a[idx[k:]])
 
	small_eps = 1e-6 #small amount so that prob can never be 0
	x = np.maximum(0+small_eps, y-lam)
 
	return x