import strat_models

import networkx as nx
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split
from utils import latexify

np.random.seed(0)

data = pd.read_csv("data/temperature_data.csv")

n=len(np.round(data["temp"]).unique())

#data
BORDER_YEAR = 2013
CAP_YEAR = 2014

train, test = train_test_split(data, test_size=0.8, random_state=0)
val, test = train_test_split(test, test_size=0.5, random_state=0)

#train
weeks_train = list(train["week"])
hours_train = list(train["hour"])
temps_train = list(train["temp"])
Z_train = []
for w,h in zip(weeks_train, hours_train):
    Z_train += [(w,h)]

#val
weeks_val = list(val["week"])
hours_val = list(val["hour"])
temps_val = list(val["temp"])

Z_val = []
for w,h in zip(weeks_val, hours_val):
    Z_val += [(w,h)]
    
#test
weeks_test = list(test["week"])
hours_test = list(test["hour"])
temps_test = list(test["temp"])

Z_test = []
for w,h in zip(weeks_test, hours_test):
    Z_test += [(w,h)]

data_train = dict(Y=temps_train, Z=Z_train)
data_val = dict(Y=temps_val, Z=Z_val)
data_test = dict(Y=temps_test, Z=Z_test)

train_common_data = dict(Y=data_train["Y"], Z=[0]*len(data_train["Y"]))
val_common_data = dict(Y=data_val["Y"], Z = [0]*len(data_val["Y"]))
test_common_data = dict(Y=data_test["Y"], Z=[0]*len(data_test["Y"]))

kwargs = dict(verbose=False, abs_tol=1e-4, maxiter=100, n_jobs=3)

D = np.eye(n)
D[0,0] = 0
for i in range(1, n):
    D[i, i-1] = -1

def train_strat_model(weights, data_train, data_val, data_test, lambd):

    loss=strat_models.nonparametric_discrete_loss()
    reg = strat_models.scaled_plus_sum_squares_reg(A=D, lambd=lambd)

    bm = strat_models.BaseModel(loss=loss, reg=reg)

    G_week = nx.cycle_graph(53)
    G_hr = nx.cycle_graph(24)
    strat_models.set_edge_weight(G_week, weights[0])
    strat_models.set_edge_weight(G_hr, weights[1])
    G = strat_models.cartesian_product([G_week, G_hr])
    
    sm = strat_models.StratifiedModel(bm, graph=G)

    info = sm.fit(data_train, **kwargs)
    anll_train = sm.anll(data_train)
    anll_val = sm.anll(data_val)
    anll_test = sm.anll(data_test)
    
    print("Stratified model with (weights, lambd) =", (weights, lambd))
    print("\t", info)
    print("\t", anll_train, anll_val, anll_test)
    
    return anll_train, anll_val, anll_test

## Separate model
train_strat_model(weights=(0,0), data_train=data_train, data_val=data_val, data_test=data_test, lambd=(0.75, 0.3))

## Standard Stratified model
weight_week = 0.6
weight_hr = 0.5
anll_train_SM, anll_val_SM, anll_test_SM = train_strat_model(weights=(weight_week, weight_hr), 
                                            data_train=data_train, data_val=data_val, 
                                            data_test=data_test, lambd=(0.05, 0.05))

## Common model
train_strat_model(weights=(1e20, 1e20), data_train=data_train, data_val=data_val, data_test=data_test, lambd=(0.65, 0.55))  

## Eigen-stratified model
print("fitting eigen-stratified models...")
kwargs["maxiter"] = 600
kwargs["verbose"] = False

K = 53*24

weight_week = .45
weight_hr = .55
lambd = (0.01, 0.001)
m = 90

G_week = nx.cycle_graph(53)
G_hr = nx.cycle_graph(24)
strat_models.set_edge_weight(G_week, weight_week)
strat_models.set_edge_weight(G_hr, weight_hr)
G_eigen = strat_models.cartesian_product([G_week, G_hr])

loss=strat_models.nonparametric_discrete_loss()
reg = strat_models.scaled_plus_sum_squares_reg(A=D, lambd=lambd)
bm_eigen = strat_models.BaseModel(loss=loss, reg=reg)

sm_eigen = strat_models.StratifiedModel(bm_eigen, graph=G_eigen)

info = sm_eigen.fit(data_train, num_eigen=m, **kwargs)
anll_train = sm_eigen.anll(data_train)
anll_val = sm_eigen.anll(data_val)
anll_test = sm_eigen.anll(data_test)

print('Eigen-stratified model, {} eigenvectors used'.format(m))
print('\t(weight_week, weight_hour, lambd, m)=', (weight_week, weight_hr, lambd, m))
print('\t', info)
print('\t', anll_train, anll_val, anll_test)