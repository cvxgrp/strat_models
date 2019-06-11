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

import strat_models
from utils import latexify


def prediction_error(x, y, z, model):
    return np.mean(model.predict(x, z) != y)

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

# Fit models
kwargs = dict(rel_tol=1e-4, abs_tol=1e-4, maxiter=500,
              n_jobs=12, verbose=0, rho=2., max_cg_iterations=30)

fully = strat_models.LogisticRegression(lambd=.1)
strat_models.set_edge_weight(G_sex, 0)
strat_models.set_edge_weight(G_age, 0)
G = strat_models.cartesian_product([G_sex, G_age])
info = fully.fit(X_train, Y_train, Z_train, G, **kwargs)
anll_test = fully.anll(X_test, Y_test, Z_test)
pred_error = prediction_error(X_test, Y_test, Z_test, fully)

print('Separate model')
print('\t', info)
print('\t', anll_test, pred_error)

strat = strat_models.LogisticRegression(lambd=.1)
strat_models.set_edge_weight(G_sex, 10)
strat_models.set_edge_weight(G_age, 500)
G = strat_models.cartesian_product([G_sex, G_age])
info = strat.fit(X_train, Y_train, Z_train, G, **kwargs)
anll_test = strat.anll(X_test, Y_test, Z_test)
pred_error = prediction_error(X_test, Y_test, Z_test, strat)

print('Stratified model')
print('\t', info)
print('\t', anll_test, pred_error)

common = strat_models.LogisticRegression(lambd=.1)
info = common.fit(X_train, Y_train, [0] *
                  len(Y_train), nx.empty_graph(1), **kwargs)
anll_test = common.anll(X_test, Y_test, [0] * len(Y_test))
pred_error = prediction_error(X_test, Y_test, [0] * len(Y_test), common)

print('Common model')
print('\t', info)
print('\t', anll_test, pred_error)

# Visualize
latexify(6)

for i in [17, -1]:
    male_params, female_params = [], []
    for node in strat.G.nodes():
        if node[0] == 'Male':
            male_params.append(strat.G.node[node]['theta'][i][0])
        else:
            female_params.append(strat.G.node[node]['theta'][i][0])
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
