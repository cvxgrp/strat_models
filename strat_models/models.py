import torch

import numpy as np
import networkx as nx
import cvxpy as cp
import scipy

import strat_models.losses as losses
import strat_models.regularizers as regularizers
from strat_models.fit import fit_stratified_model, fit_eigen_stratified_model

def G_to_data(G, shape):
	"""Vectorizes the variables in G and returns a dictionary."""
	theta_init = np.zeros(shape)
	theta_tilde_init = np.zeros(shape)
	theta_hat_init = np.zeros(shape)
	u_init = np.zeros(shape)
	u_tilde_init = np.zeros(shape)
	for i, node in enumerate(G.nodes()):
		vertex = G._node[node]
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
		vertex = G._node[node]
		vertex['theta'] = theta[i]
		vertex['theta_tilde'] = theta_tilde[i]
		vertex['theta_hat'] = theta_hat[i]
		vertex['u'] = u[i]
		vertex['u_tilde'] = u_tilde[i]
	return None

def G_to_data_eigen(G, shape, theta_shape, num_eigen):
	"""Vectorizes the variables in G and returns a dictionary."""
	theta_init = np.zeros(theta_shape)
	theta_tilde_init = np.zeros(theta_shape)
	Z_init = np.zeros(shape+(num_eigen,))
	u_init = np.zeros(theta_shape)
	u_tilde_init = np.zeros(theta_shape)
	for i, node in enumerate(G.nodes()):
		vertex = G._node[node]
		if 'theta' in vertex:
			theta_init[i] = vertex['theta']
		if 'theta_tilde' in vertex:
			theta_tilde_init[i] = vertex['theta_tilde']
		if 'Z' in vertex:
			Z_init[i] = vertex['Z']
		if 'u' in vertex:
			u_init[i] = vertex['u']
		if 'u_tilde' in vertex:
			u_tilde_init[i] = vertex['u_tilde']

	data = {
		'theta_init': theta_init,
		'theta_tilde_init': theta_tilde_init,
		'Z_init': Z_init,
		'u_init': u_init,
		'u_tilde_init': u_tilde_init,
	}

	return data

def transfer_result_to_G_eigen(result, G):
	"""Puts solution vectors into a graph G"""
	theta = result['theta']
	theta_tilde = result['theta_tilde']
	Z = result['Z']
	u = result['u']
	u_tilde = result['u_tilde']
	Q_tilde = result['Q_tilde']
	G.Q_tilde = Q_tilde
	G.Z = Z
	for i, node in enumerate(G.nodes()):
		vertex = G._node[node]
		vertex['theta'] = theta[i]
		vertex['theta_tilde'] = theta_tilde[i]
		vertex['u'] = u[i]
		vertex['u_tilde'] = u_tilde[i]
		vertex['Z'] = Z
		vertex["Q_tilde"] = Q_tilde
	return None

def turn_into_iterable(x):
	try:
		iter(x)
	except TypeError:
		return [x]
	else:
		return x


class BaseModel:
	def __init__(self, loss, reg=regularizers.zero_reg()):
		self.loss = loss
		self.local_reg = reg

class StratifiedModel:
	def __init__(self, BaseModel:BaseModel, graph):
		self.change_base_model(BaseModel)
		self.G = graph

	def change_base_model(self, base_model):
		"""
		Alters/edits the Basemodel inside the StratifiedModel
		and updates all relevant attributes.
		"""
		self.base_model = base_model

		self.loss = base_model.loss
		self.isDistribution = base_model.loss.isDistribution

		self.local_reg = base_model.local_reg
		self.lambd = base_model.local_reg.lambd

	def compute_graph_data(self, num_eigen):
		"""
		Computes all necessary graph data:
			L: Laplacian mtx
			nodelist: node set of G
			K: number of classes
		"""
		L = nx.laplacian_matrix(self.G)
		if num_eigen is None or num_eigen <= 0 or num_eigen >= len(self.G.nodes()):
			self.nodelist = self.G.nodes()
			self.K = L.shape[0]
			return L
		else:
			eigvals, Q_tilde = np.linalg.eigh(nx.laplacian_matrix(self.G).toarray())
			eigvals = eigvals[:num_eigen]
			Q_tilde = Q_tilde[:,:num_eigen]
			self.nodelist = self.G.nodes()
			self.K = L.shape[0]
			return eigvals, Q_tilde	

	def fit(self, data, num_eigen=None, **kwargs):

		#calculate Laplacian matrix
		if num_eigen is None or num_eigen <= 0 or num_eigen >= len(self.G.nodes()):
			L = self.compute_graph_data(num_eigen)
		else:
			eigvals, Q_tilde = self.compute_graph_data(num_eigen)
			
		cache = self.loss.setup(data, self.G)

		#proximals
		def l_prox(t, nu, warm_start, pool):
			return self.loss.prox(t, nu, warm_start, pool, cache)

		r_prox = self.local_reg.prox

		#G_data
		if num_eigen is None or num_eigen <= 0 or num_eigen >= len(self.G.nodes()):
			G_data = G_to_data(self.G, cache['theta_shape'])
			result, info = fit_stratified_model(
				L, cache['shape'], l_prox, r_prox, G_data=G_data, **kwargs)
			transfer_result_to_G(result, self.G)
		else:
			G_data = G_to_data_eigen(self.G, cache['shape'], cache['theta_shape'], num_eigen)
			result, info = fit_eigen_stratified_model(
				Q_tilde, eigvals, cache['shape'], l_prox, r_prox, G_data=G_data, **kwargs)
			transfer_result_to_G_eigen(result, self.G)

		return info

	def scores(self, data):
		return self.base_model.loss.scores(data, self.G)
	def anll(self, data):
		return self.base_model.loss.anll(data, self.G)
	def predict(self, data):
		return self.base_model.loss.predict(data, self.G)

	def sample(self, data):
		if not self.isDistribution:
			raise NotImplementedError("This model is not a distribution.")
			return None
		else:
			return self.base_model.loss.sample(data, self.G)