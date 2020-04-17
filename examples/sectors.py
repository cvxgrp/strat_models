import strat_models

import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split

np.random.seed(0)

#Load data
nbins = 50
K = (nbins)**2
sectors = ['XLB', 'XLV', 'XLP', 'XLY', 'XLE', 'XLF', 'XLI', 'XLK', 'XLU']
df = pd.read_csv("data/sectors_data.csv", index_col="Unnamed: 0")

#Holdout for application
holdout_date = "2018-01-01"
df_heldout = df[holdout_date:]
df = df[:holdout_date]

#Split into train and test
df_train, df_ = train_test_split(df, test_size=0.3, random_state = 10)
df_test, df_val = train_test_split(df_, test_size=0.5, random_state = 10)

#Graph
G_vix = nx.path_graph(nbins) #vix quantiles (deciles)
G_vol = nx.path_graph(nbins) #volume quantiles
G = strat_models.cartesian_product([G_vix, G_vol])

#Put data into strat_models form
def make_data_dict(df1):
    Y, Z = [], []
    yshape = []
    for vix in df1["VIX_quantile_yesterday"].unique():
        for vol in df1["5_day_trail_vol_yesterday"].unique():
            Z += [(int(vix), int(vol))]
            y = np.array(df1[(df1.VIX_quantile_yesterday == vix)&
                         (df1["5_day_trail_vol_yesterday"] == vol)][sectors]).T 

            yshape += [y.shape]
            
            if y.shape[1] == 0:
                y = y.reshape(-1,1)
            if y.shape[0] == 0:
                y = np.zeros((9,1))

            Y += [y]
    
    for node in G.nodes():
        if node not in Z:
            Y += [np.zeros((9,1))]
            Z += [node]
            
    return dict(Y=Y, Z=Z, n=9), yshape

data_train, ys = make_data_dict(df1=df_train)
ys_num = [s[1] for s in ys]
ys_zero = [idx==0 for idx in ys_num]

for idx in np.where(ys_zero)[0]:
    data_train["Y"][idx] = np.zeros((9,1))
    
print("In the training set:")
print("\tMarket conditions average of {} data points.".format(np.mean(ys_num)))
print("\tThe most populated market condition has {} data points.".format(max(ys_num)))
print("\t{} market conditions have no data.".format(len(np.where(ys_zero)[0])))
    
data_val, _ = make_data_dict(df1=df_val)
data_test, _ = make_data_dict(df1=df_test)

#Train common model
theta_common = cp.Variable((9,9))
train_common = []
for y in data_train["Y"]:
    if np.sum(abs(y)) > 1e-7:
        train_common += [y]
        
train_common = np.hstack(train_common)

N = train_common.shape[1]
Y = train_common@train_common.T/N
local = cp.Parameter(nonneg=True)
obj = cp.trace(Y@theta_common) - cp.log_det(theta_common) + 5*cp.trace(theta_common)
prob = cp.Problem(cp.Minimize(obj), [theta_common==theta_common.T])

prob.solve(verbose=False)

logprobs = []
for y, z in zip(data_test["Y"], data_test["Z"]):
    n, nk = y.shape
    Y = (y@y.T)/nk
    if (np.zeros((n,n)) == Y).all():
        continue
    logprobs += [np.linalg.slogdet(theta_common.value)[1] - np.trace(Y@theta_common.value)]

print("Common model")
print("\t", -np.mean(np.array(logprobs))/np.hstack(data_test["Y"]).shape[1])

#Common model risk
w = np.ones(9)/9
Sigma_common = np.linalg.inv(theta_common.value)
risk_common = np.sqrt(w@Sigma_common@w)
print("Common model risk on uniform portfolio =", risk_common)

#Train stratified model

kwargs = dict(verbose=True, abs_tol=1e-3, maxiter=1000, rho=10)

wt_vix = 1500
wt_vol = 2500
local = .15

loss = strat_models.covariance_max_likelihood_loss()
reg = strat_models.trace_reg(lambd=local)

print("local, vix, vol =", local, wt_vix, wt_vol)

G_vix = nx.path_graph(nbins) #vix quantiles (deciles)
G_vol = nx.path_graph(nbins) #volume quantiles
strat_models.set_edge_weight(G_vix, wt_vix)
strat_models.set_edge_weight(G_vol, wt_vol)

G = strat_models.cartesian_product([G_vix, G_vol])

bm = strat_models.BaseModel(loss=loss,reg=reg)
sm = strat_models.StratifiedModel(BaseModel=bm, graph=G)

sm.fit(data=data_train, **kwargs)
print("Held-out test loss")
print("\t", sm.anll(data_test)/np.hstack(data_test["Y"]).shape[1])



#Stratified model risk on uniform portfolio
w = np.ones(9)/9
sm_risk = dict()

for node in G.nodes():
    sm_sigma = np.linalg.inv(sm.G._node[node]["theta"])
    sm_risk[node] = ((w @ sm_sigma @ w)**(1/2))

risk_mtx_sm = np.zeros((nbins, nbins))
for i in range(nbins):
    for j in range(nbins):
        risk_mtx_sm[i,j] = sm_risk[i,j]

lxy = [""]*nbins
lxy[0] = 0
lxy[-1] = 100

fig, ax = plt.subplots(1,1, figsize=(15,10))

sns.heatmap((risk_mtx_sm.T), cmap="nipy_spectral", cbar=True, ax=ax)

ax.set_title("Risk $\sqrt{w^T \Sigma_z w}$, stratified model", fontsize="xx-large")
ax.set_ylabel("Market volume quantile", fontsize="x-large")
ax.set_xlabel("Volatility index quantile", fontsize="x-large")
ax.set_xticklabels(labels=lxy, fontsize="large")
ax.set_yticklabels(labels=lxy, fontsize="large")
ax.set_aspect(1.0)
ax.invert_yaxis()
plt.tight_layout()
plt.show()


#Compute Markowitz portfolio leverage
ws_sm = dict()
for node in sm.G.nodes():
    w = cp.Variable(9)
    Sigma_z = np.linalg.inv(sm.G._node[node]["theta"])
    obj = cp.quad_form(w,Sigma_z)
    constraints = [sum(w) == 1, cp.norm(w,1) <= 1.5]
    cp.Problem(cp.Minimize(obj), constraints).solve()
    ws_sm[node] = w.value

leverage = np.nan * np.ones((50,50))
for node in sm.G.nodes():
    leverage[node] = np.linalg.norm(ws_sm[node],1)
    
fig, ax = plt.subplots(1,1, figsize=(15,10))
sns.heatmap(leverage.T, cmap="nipy_spectral", cbar=True, ax=ax)
ax.set_title("Leverage $||w_z||_1$, stratified model", fontsize="xx-large")
ax.set_ylabel("Market volume quantile", fontsize="x-large")
ax.set_xlabel("Volatility index quantile", fontsize="x-large")
ax.set_xticklabels(labels=lxy, fontsize="large")
ax.set_yticklabels(labels=lxy, fontsize="large")
ax.set_aspect(1.0)
ax.invert_yaxis()
plt.show()

#Application
df = df_heldout
RETURNS = np.array(df[sectors])/100

value_common = 1
vals_common = [value_common]

value_strat = 1
vals_strat = [value_strat]

SPY = pd.read_csv("data/SPY.csv", parse_dates=["caldt"])

SPY_value = 1
SPY_vals = [SPY_value]
SPY_RETURNS = np.array(SPY[(SPY.caldt > holdout_date)&(SPY.caldt < "2019-01-01")]["sprtrn"])

lev, lev_common = [], []
W, Wcommon = [], []

for date in range(1,df.shape[0]):
    vix = int(df.iloc[date]["VIX_quantile_yesterday"])
    vol = int(df.iloc[date]["5_day_trail_vol_yesterday"])
        
    w = cp.Variable(9)
    w_common = cp.Variable(9)

    thresh=1.5
    p = 1
    cons_sm = [sum(w)==1, cp.norm(w, p) <= thresh]
    cons_common = [sum(w_common)==1, cp.norm(w_common, p) <= thresh]
    
    obj = cp.quad_form(w, np.linalg.inv(sm.G._node[(vix, vol)]["theta"]))
    obj_common = cp.quad_form(w_common, np.linalg.inv(theta_common.value))

    
    prob = cp.Problem(cp.Minimize(obj), cons_sm)
    prob_common = cp.Problem(cp.Minimize(obj_common), cons_common)
    
    prob.solve(solver="MOSEK") 
    prob_common.solve(solver="MOSEK") 
        
    value_strat *= (1+RETURNS[date, :])@w.value
    vals_strat += [value_strat]
    lev += [np.linalg.norm(w.value,1)]
    W += [w.value.reshape(-1,1)]

    value_common *= (1+RETURNS[date, :])@w_common.value
    vals_common += [value_common]
    lev_common += [np.linalg.norm(w_common.value,1)]
    Wcommon += [w_common.value.reshape(-1,1)]

    SPY_value *= (1+SPY_RETURNS[date])
    SPY_vals += [SPY_value]

fig, ax = plt.subplots(1,1, figsize=(10,4))
ax.plot(df.index, vals_strat, label="Stratified Model Policy", color="black")
ax.plot(df.index, vals_common, label="Common Model Policy", color="blue")
ax.plot(df.index, SPY_vals, label="SPY", color="red")

ax.set_xlabel("Date", fontsize="x-large")
ax.set_ylabel("Overall return (% of portfolio)", fontsize="x-large")
ax.legend()
plt.show()