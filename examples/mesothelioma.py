"""
Mesothelioma prediction

The data is from: https://archive.ics.uci.edu/ml/datasets/Mesothelioma%C3%A2%E2%82%AC%E2%84%A2s+disease+data+set+
"""

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("..")

import strat_models

from utils import latexify


def prediction_error(data, model):
    return np.mean(model.predict(data) != data["Y"])

np.random.seed(0)

# Load data
raw_data = pd.read_excel('./data/Mesothelioma data set.xlsx')

# Extract data
data = raw_data.copy()
data['age'] = pd.to_numeric(data['age']).astype(np.int)
data = pd.get_dummies(
    data, columns=['city', 'keep side', 'type of MM', 'habit of cigarette'])

ages = data['age']
sexes = ['Male' if s == 1 else 'Female' for s in pd.to_numeric(data['gender'])]
list_of_ages = np.arange(19, 86)
list_of_sexes = ['Male', 'Female']

Y = np.array(data['class of diagnosis'] - 1)
data.drop(['age', 'gender', 'class of diagnosis',
           'diagnosis method'], inplace=True, axis=1)
X = np.array(data, dtype=np.double)
Z = [(gender, age) for gender, age in zip(sexes, ages)]

X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
    X, Y, Z, test_size=0.1)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Create graphs
G_sex = nx.Graph()
G_sex.add_nodes_from(['Male', 'Female'])
G_sex.add_edge('Male', 'Female')

num_ages = len(list_of_ages)
G_age = nx.path_graph(num_ages)
index_to_age = dict(zip(np.arange(num_ages), list_of_ages))
G_age = nx.relabel_nodes(G_age, index_to_age)

data_train = dict(X=X_train, Y=Y_train, Z=Z_train)
data_test = dict(X=X_test, Y=Y_test, Z=Z_test)

loss=strat_models.logistic_loss(intercept=True)

# Fit models
print("fitting...")
kwargs = dict(rel_tol=1e-4, abs_tol=1e-4, maxiter=500,
              n_jobs=12, verbose=True, rho=2., max_cg_iterations=30)

strat_models.set_edge_weight(G_sex, 0)
strat_models.set_edge_weight(G_age, 0)
G = strat_models.utils.cartesian_product([G_sex, G_age])

bm_fully = strat_models.BaseModel(loss=loss)
sm_fully = strat_models.StratifiedModel(bm_fully, graph=G)

info = sm_fully.fit(data_train, **kwargs)
anll_test = sm_fully.anll(data_test)
pred_error = prediction_error(data_test, sm_fully)

print('Separate model')
print('\t', info)
print('\t', anll_test, pred_error)

strat_models.set_edge_weight(G_sex, 10)
strat_models.set_edge_weight(G_age, 500)
G = strat_models.utils.cartesian_product([G_sex, G_age])

bm_strat = strat_models.BaseModel(loss=loss)
sm_strat = strat_models.StratifiedModel(bm_strat, graph=G)

info = sm_strat.fit(data_train, **kwargs)
anll_test = sm_strat.anll(data_test)
pred_error = prediction_error(data_test, sm_strat)

print('Stratified model')
print('\t', info)
print('\t', anll_test, pred_error)

data_common_train = dict(X=X_train, Y=Y_train, Z=[0]*len(Y_train))
data_common_test = dict(X=X_test, Y=Y_test, Z=[0]*len(Y_test))

G = nx.empty_graph(1)
bm_common = strat_models.BaseModel(loss=loss)
sm_common = strat_models.StratifiedModel(bm_common, graph=G)

info = sm_common.fit(data_common_train, **kwargs)
anll_test = sm_common.anll(data_common_test)
pred_error = prediction_error(data_common_test, sm_common)

print('Common model')
print('\t', info)
print('\t', anll_test, pred_error)

# Visualize
latexify(6)

for i in [17, -1]:
    male_params, female_params = [], []
    for node in sm_strat.G.nodes():
        if node[0] == 'Male':
            male_params.append(sm_strat.G.node[node]['theta'][i][0])
        else:
            female_params.append(sm_strat.G.node[node]['theta'][i][0])
    title = data.columns[i] if i > 0 else 'intercept'
    plt.title(title)
    print(title)
    plt.plot(list_of_ages, male_params, label='Male')
    plt.plot(list_of_ages, female_params, label='Female')
    plt.xlabel('Age')
    plt.legend()
    plt.savefig('figs/mesothelioma_%s.pdf' %
                (title))
    plt.close()