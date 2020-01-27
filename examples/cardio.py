import strat_models

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import latexify
import matplotlib.pyplot as plt

np.random.seed(123)

"""
Cardiovascular disease dataset
data is from https://www.kaggle.com/sulianova/cardiovascular-disease-dataset
"""
#gender: 1-women, 2-men

data = pd.read_csv("data/cardio_train.csv", sep=";")

#Basic feature engineering
data = data.drop(["id"], axis=1)
data.age = [int(days) for days in np.round(data.age/365.25)]
data.gender = ["Male" if gender==2 else "Female" for gender in data.gender]
data = data[data.age != 30]
dummies_chol = pd.get_dummies(data.cholesterol, prefix="cholesterol")
dummies_gluc = pd.get_dummies(data.gluc, prefix="glucose")
data = pd.concat([data, dummies_chol, dummies_gluc], axis=1)
data = data.drop(["cholesterol", "gluc"], axis=1)

Y = 1-np.array(data.cardio)
ages = data.age
sexes = data.gender

data.drop(["age", "gender", "cardio"], inplace=True, axis=1)
X = np.array(data, dtype=np.double)
Z = [(gender, age) for gender, age in zip(sexes, ages)]

X_train, X_, Y_train, Y_, Z_train, Z_ = train_test_split(X, Y, Z, test_size=0.95)
X_val, X_test, Y_val, Y_test, Z_val, Z_test = train_test_split(X_, Y_, Z_, test_size=0.94736)

print("Training set has {} examples.".format(len(Z_train)))
print("Validation set has {} examples.".format(len(Z_val)))
print("Test set has {} examples.".format(len(Z_test)))

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_val = ss.transform(X_val)
X_test = ss.transform(X_test)

# Create graphs
def create_sex_graph(weight=0):
    G_sex = nx.Graph()
    G_sex.add_nodes_from(["Male", "Female"])
    G_sex.add_edge("Male", "Female")
    for _, _, e in G_sex.edges(data=True):
        e["weight"] = weight
    return G_sex

list_of_ages = list(set(ages))
num_ages = len(list_of_ages)
index_to_age = dict(zip(np.arange(num_ages), list_of_ages))

def create_age_graph(weight=0):
    G_age = nx.path_graph(num_ages)
    G_age = nx.relabel_nodes(G_age, index_to_age)
    for _, _, e in G_age.edges(data=True):
        e["weight"] = weight
    return G_age

data_train = dict(X=X_train, Y=Y_train, Z=Z_train)
data_val = dict(X=X_val, Y=Y_val, Z=Z_val)
data_test = dict(X=X_test, Y=Y_test, Z=Z_test)

K = num_ages * 2 #num_ages * num_sexes
print("K = ", K)

# Fit models
kwargs = dict(rel_tol=1e-4, abs_tol=1e-4, maxiter=400,
              n_jobs=4, verbose=False, rho=3., max_cg_iterations=30)

## Separate model
G_sex = create_sex_graph(weight=0)
G_age = create_age_graph(weight=0)
G = strat_models.utils.cartesian_product([G_sex, G_age])

loss=strat_models.logistic_loss(intercept=True)
reg=strat_models.sum_squares_reg(lambd=35)

bm_sep = strat_models.BaseModel(loss=loss, reg=reg)
sm_sep = strat_models.StratifiedModel(bm_sep, graph=G)

info = sm_sep.fit(data_train, **kwargs)
anll_train_sep = sm_sep.anll(data_train)
anll_val_sep = sm_sep.anll(data_val)
anll_test_sep = sm_sep.anll(data_test)

print('Separate model')
print('\tlambda =', 35)
print('\t', info)
print('\t', anll_train_sep, anll_val_sep, anll_test_sep)

## Common model
G = nx.empty_graph(1)

loss=strat_models.logistic_loss(intercept=True)
reg=strat_models.sum_squares_reg(lambd=5)

data_common_train = dict(X=data_train["X"], Y=data_train["Y"], Z=[0]*len(data_train["Y"]))
data_common_val = dict(X=data_val["X"], Y=data_val["Y"], Z=[0]*len(data_val["Y"]))
data_common_test = dict(X=data_test["X"], Y=data_test["Y"], Z=[0]*len(data_test["Y"]))

bm_common = strat_models.BaseModel(loss=loss, reg=reg)
sm_common = strat_models.StratifiedModel(bm_common, graph=G)

info = sm_common.fit(data_common_train, **kwargs)
print("Common model")
print('\tlambda =', 5)
print("\t", info)
print('\t', sm_common.anll(data_common_train), sm_common.anll(data_common_val), sm_common.anll(data_common_test))


## Standard stratified model
wt_sex = 125
wt_age = 150
lambd = 0.01

G_sex = create_sex_graph(weight=wt_sex)
G_age = create_age_graph(weight=wt_age)
G = strat_models.utils.cartesian_product([G_sex, G_age])

loss=strat_models.logistic_loss(intercept=True)
reg=strat_models.sum_squares_reg(lambd=lambd)  

bm_strat = strat_models.BaseModel(loss=loss, reg=reg)
sm_strat = strat_models.StratifiedModel(bm_strat, graph=G)

info = sm_strat.fit(data_train, **kwargs)
anll_train_SM = sm_strat.anll(data_train)
anll_val_SM = sm_strat.anll(data_val)
anll_test_SM = sm_strat.anll(data_test)

print('Stratified model')
print("\t weights/lambd: {}".format((wt_sex, wt_age, lambd)))
print('\t', info)
print('\t', anll_train_SM, anll_val_SM, anll_test_SM)

## Eigen-stratified model
print("fitting...")
train_anlls = []
val_anlls = []
test_anlls = []
HPs = []

kwargs = dict(rel_tol=1e-5, abs_tol=1e-5, maxiter=1000,
              n_jobs=4, verbose=False, rho=3., max_cg_iterations=30)

weight_sex = 15
weight_age = 175
lambd = 2.5
k = 5

G_sex = create_sex_graph(weight=weight_sex)
G_age = create_age_graph(weight=weight_age)
G_eigen = strat_models.utils.cartesian_product([G_sex, G_age])

loss=strat_models.logistic_loss(intercept=True)
reg=strat_models.sum_squares_reg(lambd=lambd)

bm_eigen = strat_models.BaseModel(loss=loss,reg=reg)

sm_eigen = strat_models.StratifiedModel(bm_eigen, graph=G_eigen)

info = sm_eigen.fit(data_train, num_eigen=k, **kwargs)
anll_train = sm_eigen.anll(data_train)
anll_val = sm_eigen.anll(data_val)
anll_test = sm_eigen.anll(data_test)

print('Eigen-stratified model, {} eigenvectors used'.format(k))
print("\tsex={}, age={}, lambd={}, m={}".format(weight_sex, weight_age, lambd, k))
print('\t', info)
print('\t', anll_train, anll_val, anll_test)

HPs += [(weight_sex, weight_age, lambd, k)]
train_anlls += [anll_train]
val_anlls += [anll_val]
test_anlls += [anll_test]