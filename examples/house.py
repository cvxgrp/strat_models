"""House price prediction.

The data is from: https://www.kaggle.com/harlfoxem/housesalesprediction
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
import scipy.spatial.distance as distance
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.append("..")

import strat_models

from utils import latexify

# Load data
df = pd.read_csv('data/kc_house_data.csv',
                 usecols=["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
                          "floors", "waterfront", "condition", "grade", "yr_built", "lat", "long"],
                 low_memory=False)

df['log_price'] = np.log(df['price'])
df = df.query('long <= -121.6')
bins = 50
df['lat_bin'] = pd.cut(df['lat'], bins=bins)
df['long_bin'] = pd.cut(df['long'], bins=bins)
code_to_latbin = dict(enumerate(df['lat_bin'].cat.categories))
code_to_longbin = dict(enumerate(df['long_bin'].cat.categories))
df['lat_bin'] = df['lat_bin'].cat.codes
df['long_bin'] = df['long_bin'].cat.codes
df.drop(["price"], axis=1, inplace=True)

# Create dataset
np.random.seed(0)
df_train, df_test = model_selection.train_test_split(df)
G = nx.grid_2d_graph(bins, bins)

def get_data(df):
    Xs = []
    Ys = []
    Zs = []
    for node in G.nodes():
        latbin, longbin = node
        df_node = df.query('lat_bin == %d & long_bin == %d' %
                           (latbin, longbin))
        X_node = np.array(df_node.drop(
            ['log_price', 'lat', 'long', 'lat_bin', 'long_bin'], axis=1))
        Y_node = np.array(df_node['log_price'])
        N = X_node.shape[0]
        Xs += [X_node]
        Ys += [Y_node]
        Zs.extend([node] * N)

    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)[:, np.newaxis], Zs

X_train, Y_train, Z_train = get_data(df_train)
X_test, Y_test, Z_test = get_data(df_test)

K, n = bins * bins, X_train.shape[1]
print(K * n, "variables")

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Fit models
data_train = dict(X=X_train, Y=Y_train, Z=Z_train)
data_test = dict(X=X_test, Y=Y_test, Z=Z_test)

kwargs = dict(rel_tol=1e-5, abs_tol=1e-5, maxiter=1000, n_jobs=2, verbose=1)

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

loss=strat_models.sum_squares_loss(intercept=True)
reg= strat_models.sum_squares_reg(lambd=1e-4)
bm = strat_models.BaseModel(loss=loss, reg=reg)

strat_models.set_edge_weight(G, 1e-8)

sm_fully = strat_models.StratifiedModel(bm, graph=G)

info = sm_fully.fit(data_train, **kwargs)
score = sm_fully.scores(data_test)
print("Fully")
print("\t", info)
print("\t", score)

strat_models.set_edge_weight(G, 15)
sm_strat = strat_models.StratifiedModel(bm, graph=G)

info = sm_strat.fit(data_train, **kwargs)
score = sm_strat.scores(data_test)
print("Strat")
print("\t", info)
print("\t", score)

G = nx.empty_graph(1)
sm_common = strat_models.StratifiedModel(bm, graph=G)

data_common_train = dict(X=X_train, Y=Y_train, Z=[0]*len(Y_train))
data_common_test = dict(X=X_test, Y=Y_test, Z=[0]*len(Y_test))

info = sm_common.fit(data_common_train, **kwargs)
score = sm_common.scores(data_common_test)
print("Common")
print("\t", info)
print("\t", score)

rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=1, n_jobs=-1)
rf.fit(df_train.drop(['log_price', 'lat_bin', 'long_bin'],
                     axis=1), df_train['log_price'])
score = rms(rf.predict(df_test.drop(
    ['log_price', 'lat_bin', 'long_bin'], axis=1)) - df_test['log_price'])
print("RF")
print("\t", score)
print("\t", np.sum([rf.estimators_[i].tree_.node_count for i in range(50)]))

# Visualize
latexify(fig_width=8)
params = np.array([sm_strat.G.node[node]['theta'] for node in sm_strat.G.nodes()])
params = params.reshape(bins, bins, 10)[::-1, :, :]
feats = ['bedrooms', 'bathrooms', 'sqft living', 'sqft lot', 'floors',
         'waterfront', 'condition', 'grade', 'yr built', 'intercept']
min_lat = df['lat'].min()
max_lat = df['lat'].max()
min_long = df['long'].min()
max_long = df['long'].max()
lat_labels = ["%.1f" % x for x in np.linspace(min_lat, max_lat, 6)]
long_labels = ["%.1f" % x for x in np.linspace(min_long, max_long, 6)]

fig, axes = plt.subplots(2, 5)
for i, feat in enumerate(feats):
    ax = axes[i // 5, i % 5]
    ax.imshow(params[:, :, i])
    ax.axis('off')
    ax.set_title(feat)
plt.tight_layout()
plt.subplots_adjust(wspace=.1, hspace=.1)
plt.savefig('./figs/house_price_coef.pdf')