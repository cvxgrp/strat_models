import strat_models

import networkx as nx
import numpy as np

def test_eigen():
	"""Example: solve ||X\theta - Y||^2 + ||\theta||^2"""

	print("ridge regression test...")
	K = 100
	G = nx.cycle_graph(K)
	n = 10
	m = 2
	X = np.random.randn(500, n)
	Z = np.random.randint(K, size=500)
	Y = np.random.randn(500, m)

	bm = strat_models.BaseModel(
		loss=strat_models.losses.sum_squares_loss(intercept=False), 
		reg=strat_models.regularizers.sum_squares_reg(lambd=1))

	sm = strat_models.StratifiedModel(bm, graph=G)

	data = dict(X=X, Y=Y, Z=Z)
	kwargs = dict(verbose=True, abs_tol=1e-6, maxiter=500)

	info = sm.fit(data, num_eigen=30, **kwargs)
	assert info["optimal"]

	predictions = sm.predict(data=data)

	print("ANLL is {}".format(sm.anll(data)))

	print("eigen-stratified ridge regression done.")

def test_nonparametric_discrete():
	print("Non-parametric discrete distibution test...")

	K = 100
	num_classes = 10
	SIZE=100

	G = nx.cycle_graph(K)
	strat_models.utils.set_edge_weight(G, 10)

	K_eye = np.eye(K)

	Z = np.random.randint(K, size=SIZE)
	Y = np.random.randint(0,num_classes, size=SIZE)

	print(Z)

	bm = strat_models.BaseModel(loss=strat_models.losses.nonparametric_discrete_loss(), 
						reg=strat_models.regularizers.sum_squares_reg(lambd=0.4))

	sm = strat_models.StratifiedModel(bm, graph=G)

	data = dict(Y=Y, Z=Z)
	kwargs = dict(verbose=True, abs_tol=1e-6, maxiter=500)

	info = sm.fit(data, **kwargs)
	assert info["optimal"]	

	data_test = dict(Z=np.random.randint(K, size=SIZE),Y=np.random.randint(0,num_classes, size=SIZE))

	print("ANLL is {}".format(sm.anll(data_test)))

	print("Non-parametric discrete loss done.")

def test_poisson():
	print("Poisson test...")

	K = 1000
	G = nx.cycle_graph(K)
	Z = np.random.randint(K, size=10000)
	Y = np.random.randint(1, 10, size=10000)

	bm = strat_models.BaseModel(loss=strat_models.losses.poisson_loss(min_theta=1e-3), 
			reg=strat_models.regularizers.min_threshold_reg_one_elem(lambd=1e-3))

	sm = strat_models.StratifiedModel(bm, graph=G)

	data = dict(Y=Y,Z=Z)
	kwargs = dict(verbose=True, abs_tol=1e-6, maxiter=500)

	info = sm.fit(data, **kwargs)
	assert info["optimal"]

	data_sample = dict(Z=np.random.randint(2, size=100))
	samples = sm.sample(data=data_sample)

	print("ANLL is {}".format(sm.anll(data)))

	print("Poisson done.")

def test_ridge_regression():
	"""Example: solve ||X\theta - Y||^2 + ||\theta||^2"""

	print("ridge regression test...")
	K = 100
	G = nx.cycle_graph(K)
	n = 10
	m = 2
	X = np.random.randn(500, n)
	Z = np.random.randint(K, size=500)
	Y = np.random.randn(500, m)

	bm = strat_models.BaseModel(
		loss=strat_models.losses.sum_squares_loss(intercept=False), 
		reg=strat_models.regularizers.sum_squares_reg(lambd=1))

	sm = strat_models.StratifiedModel(bm, graph=G)

	data = dict(X=X, Y=Y, Z=Z)
	kwargs = dict(verbose=True, abs_tol=1e-6, maxiter=500)

	info = sm.fit(data, **kwargs)
	assert info["optimal"]

	predictions = sm.predict(data=data)

	print("ANLL is {}".format(sm.anll(data)))

	print("ridge regression done.")

def test_lasso():
	"""Example: solve ||X\theta - Y||^2 + ||\theta||^2"""

	print("lasso test...")
	K = 100
	G = nx.cycle_graph(K)
	n = 10
	m = 2
	X = np.random.randn(500, n)
	Z = np.random.randint(K, size=500)
	Y = np.random.randn(500, m)

	bm = strat_models.BaseModel(
		loss=strat_models.losses.sum_squares_loss(intercept=True), 
		reg=strat_models.regularizers.L1_reg(lambd=1))

	sm = strat_models.StratifiedModel(bm, graph=G)

	data = dict(X=X, Y=Y, Z=Z)
	kwargs = dict(verbose=True, abs_tol=1e-6, maxiter=500)

	info = sm.fit(data, **kwargs)
	assert info["optimal"]

	predictions = sm.predict(data=data)

	print("ANLL is {}".format(sm.anll(data)))

	print("lasso done.")

def test_log_reg():
	print("Logistic regression test...")
	K = 30
	G = nx.cycle_graph(K)
	n = 10
	X = np.random.randn(1000, n)
	Z = np.random.randint(K, size=1000)
	Y = np.random.randint(1, 10, size=1000)

	bm = strat_models.BaseModel(loss=strat_models.losses.logistic_loss(intercept=True))
	sm = strat_models.StratifiedModel(bm, graph=G)
	data = dict(X=X, Y=Y, Z=Z)
	kwargs = dict(verbose=True, abs_tol=1e-6, maxiter=500)

	info = sm.fit(data, **kwargs)
	assert info["optimal"]

	data_predict = dict(X=X[:20, :], Z=Z[:20])
	predictions = sm.predict(data=data_predict)

	print("ANLL is {}".format(sm.anll(data)))

	print("logreg done.")

def test_bernoulli():
	print("Bernoulli test...")

	K = 2
	G = nx.cycle_graph(K)
	Z = np.random.randint(K, size=1000)
	Y = np.random.randint(0, 2, size=1000)
	# p = strat_strat_models.Bernoulli()
	# p.fit(Y, Z, G, inplace=True, verbose=True, n_jobs=12)

	# anll = p.anll(Y, Z)
	# sample = p.sample(Z)
	# print(sample)
	# print(anll)

	bm = strat_models.BaseModel(loss=strat_models.losses.bernoulli_loss(1e-5,1-1e-5), 
		reg=strat_models.regularizers.clip_reg((1e-5,1-1e-5)))
	sm = strat_models.StratifiedModel(bm, graph=G)
	data = dict(Y=Y,Z=Z)
	kwargs = dict(verbose=True, abs_tol=1e-4, maxiter=500, n_jobs=2)

	info = sm.fit(data, **kwargs)
	assert info["optimal"]

	data_sample = dict(Z=np.random.randint(2, size=100))
	samples = sm.sample(data=data_sample)

	print("ANLL is {}".format(sm.anll(data)))

	print("Bernoulli done.")

if __name__ == '__main__':
	np.random.seed(0)
	
	test_eigen()
	test_nonparametric_discrete()
	test_ridge_regression()
	test_lasso()
	test_poisson()
	test_bernoulli()
	test_log_reg()

	print("All tests passed!")