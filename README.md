# strat_models

`strat_models` is a Python package for fitting Laplacian regularized stratified models.

The implementation is based on our paper
[A distributed method for fitting Laplacian regularized stratified models](http://web.stanford.edu/~boyd/papers/strat_models.html).

## Installation

To install the latest version, clone the repository and run:
```
pip install .
```
## Usage
To fit a stratified model, one needs to specify a base model, a graph, and data.

### Base model
To specify a base model, one needs to specify a local loss function and a local regularization function.
Currently, we support fitting with the following local loss functions:
* Sum of squares loss
* Logistic loss
* Bernoulli negative log-likelihood
* Poisson negative log-likelihood
* Multivariate Gaussian negative log-likelihood (zero mean)
* Multivariate Gaussian negative log-likelihood (non-zero mean)

We also support fitting with the following local regularization functions:
* None
* Sum of squares 
* L1
* L2
* Elastic net
* Negative logarithm
* Non-negative indicator function
* Box indicator function

For example, here is how to specify a base model with a logistic regression local loss and
sum of squares local regularization with parameter 1:
```
bm = strat_models.BaseModel(loss=strat_models.logistic_loss(intercept=True), 
								reg=strat_models.sum_squares_reg(lambd=1))
```

### Graph
The graphs must be made using `networkx` Graphs. You can look more into how to build a `networkx`
Graph [here](https://networkx.github.io/documentation/stable/tutorial.html#).

### Data
Data must be specified as a dictionary with keys `X`, `Y`, and `Z` (`X` may be omitted if fitting a distribution.) `X` and `Y` must be Python lists with entries being the inputs/outputs, respectively.
`Z` must be a Python list with entries being tuples of the same form as the specified `networkx`
Graph.

Here is an example of an appropriate data dictionary with 500 samples consisting of 100 unique categorical parameter values:
```
K = 100
n = 10
num_samples = 500

X = np.random.randn(num_samples, n)
Z = np.random.randint(K, size=num_samples)
Y = np.random.randn(num_samples, 1)

data = dict(X=X, Y=Y, Z=Z)
```

### Stratified model
A stratified model is specified by a base model and a graph.
For example, here is how to specify a stratified model with a base model `bm` and a graph `G`:
```
sm = strat_models.StratifiedModel(bm, graph=G)
```

For example, we fit a ridge regression model with local regularization, 
stratified based on time and location:

```
import strat_models
import networkx as nx

# G is the cartesian product of a cycle graph and a path graph
G_time = nx.cycle_graph(10)
G_location = nx.path_graph(20)
G = strat_models.cartesian_product([G_time, G_location])

# Get the data
X, Y, Z = get_data()
Xtest, Ytest, Ztest = get_test_data()

data = dict(X=X, Y=Y, Z=Z)

# Construct the model
m = strat_models.RidgeRegression(lambd=0.05)
m.fit(X, Y, Z, G)

bm = strat_models.BaseModel(loss=strat_models.sum_squares_loss(intercept=True),
							reg=strat_models.sum_squares_reg(lambd=1))

sm = strat_models.StratifiedModel(bm, graph=G)

sm.fit(data)
```

## Examples
First, install the requirements in `requirements.txt`:
```
pip install -r requirements.txt
```

Next, navigate to the `examples` folder.
Unfortunately, we cannot include the data to run the examples in this repository.
At the top of each script are instructions on how to get the data to be able to run that script.
Once downloaded, place the data in a folder called `data` in the `examples` folder,
and then run the script; for example:
```
python house.py
```

## Running tests
First, install `pytest`:
```
pip install pytest
```

Then, in the main directory, run:
```
pytest test.py
```

## Citing `strat_models`

If you use `strat_models`, please cite the following paper:

```
@article{strat_models,
    author       = {Tuck, J. and Barratt, S. and Boyd, S.},
    title        = {A distributed method for fitting Laplacian regularized stratified models},
    journal      = {arXiv preprint arXiv:1904.12017},
    year         = {2019},
}