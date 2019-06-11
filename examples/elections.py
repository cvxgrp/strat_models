"""Elections example.

The data is from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PEJ5QU&version=2.1
Neighboring states are from: https://github.com/ubikuity/List-of-neighboring-states-for-each-US-state/blob/master/neighbors-states.csv
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

import strat_models
from utils import latexify

# Load data
raw_data = pd.read_csv('data/1976-2016-senate.csv')
data = raw_data.copy()
data['democrat'] = (data['party'] == 'democrat').astype(np.double)
data.drop(['candidate', 'writein', 'state', 'state_ic', 'state_fips',
           'state_cen', 'office', 'version', 'stage', 'special', 'party',
           'totalvotes'], inplace=True, axis=1)
states = data.state_po.unique()
years = np.sort(data.year.unique())
districts = data.district.unique()

neighbors = pd.read_csv('data/neighbors-states.csv')

# Create dataset
data_train = data.query('year != 2014 & year != 2016')
data_test = data.query('year == 2014 | year == 2016')


def extract_data(df):
    Y = []
    Z = []
    for state in states:
        for year in years:
            data = df[(raw_data.year == year) & (df.state_po == state)]
            for district in data.district.unique():
                data_dist = data[df.district == district]
                Y.append(data_dist.democrat[data_dist.candidatevotes.idxmax()])
                Z.append((state, year))
    return Y, Z

Y_train, Z_train = extract_data(data_train)
Y_test, Z_test = extract_data(data_test)


# Create graph
G_state = nx.Graph()
for state in states:
    G_state.add_node(state)

for state1 in states:
    for state2 in states:
        if state2 in list(neighbors[neighbors.StateCode == state1]['NeighborStateCode']):
            G_state.add_edge(state1, state2)

n_years = len(years)
G_time = nx.path_graph(n_years)
G_time = nx.relabel_nodes(G_time, dict(zip(np.arange(n_years), years)))


kwargs = dict(abs_tol=1e-5, rel_tol=1e-5, maxiter=200, n_jobs=4)

fully = strat_models.Bernoulli()
strat_models.set_edge_weight(G_state, 0)
strat_models.set_edge_weight(G_time, 0)
G = strat_models.cartesian_product([G_state, G_time])
info = fully.fit(Y_train, Z_train, G, **kwargs)
anll_train = fully.anll(Y_train, Z_train)
anll_test = fully.anll(Y_test, Z_test)
print("Separate model")
print("\t", info)
print("\t", anll_train, anll_test)

strat = strat_models.Bernoulli()
strat_models.set_edge_weight(G_state, 1)
strat_models.set_edge_weight(G_time, 4)
G = strat_models.cartesian_product([G_state, G_time])
info = strat.fit(Y_train, Z_train, G, **kwargs)
anll_train = strat.anll(Y_train, Z_train)
anll_test = strat.anll(Y_test, Z_test)
print("Stratified model")
print("\t", info)
print("\t", anll_train, anll_test)

common = strat_models.Bernoulli()
info = common.fit(Y_train, [0] * len(Y_train), nx.empty_graph(1), **kwargs)
anll_train = common.anll(Y_train, [0] * len(Y_train))
anll_test = common.anll(Y_test, [0] * len(Y_test))
print("Common model")
print("\t", info)
print("\t", anll_train, anll_test)


# Visualize model
states = list(G_state.nodes())
L = nx.laplacian_matrix(G_state).todense()
eigvals, eigvecs = np.linalg.eig(L)
eigvals_sorted = np.argsort(eigvals)
first_nonzero_eigvec = eigvecs[:, eigvals_sorted[1]]
states_sorted = [x for _, x in sorted(zip(first_nonzero_eigvec, states))]

statemap = {}
for state in states_sorted:
    state_vals = []
    for year in years:
        state_vals.append(float(strat.G.node[(state, year)]['theta']))
    statemap[state] = state_vals
output = pd.DataFrame(statemap).T
output.columns = years

latexify(fig_width=13, fontsize=10)
plt.pcolormesh(output, cmap='coolwarm_r')
plt.yticks(np.arange(0.5, len(output.index), 1), output.index)
plt.xticks(np.arange(0.5, len(output.columns), 1), output.columns)
plt.colorbar()
plt.savefig('./figs/election_heatmap.pdf')
plt.close()
