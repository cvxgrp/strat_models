# cvxstrat

`cvxstrat` is a Python package for fitting Laplacian regularized stratified models.

The implementation is based on our paper
[A distributed method for fitting Laplacian regularized stratified models](http://web.stanford.edu/~boyd/papers/strat_models.html).

## Installation

To install pytorch, follow instructions from [pytorch.org](https://pytorch.org/), e.g. (python 3.7 pip installation):
```
pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl
```

To install the latest version, clone the repository and run:
```
python setup.py install
```

## Usage

To fit a stratified model, one needs to specify a base model, a graph, and data.

### Base model
Currently, we support fitting the following base models:
* Bernoulli distribution
* Poisson distribution
* Logistic regression (binary or multi-class)
* Ridge regression

Each of these models has some number of optional arguments to specify, such as local regularization
parameters or the number of threads to use for parallelizing CPU operations.
For example, here is how to specify fitting a logistic regression model with sum-of-squares regularization
with parameter 1 and 4 threads:
```
m = cvxstrat.LogisticRegression(num_threads=4, lambd=1)
```

More base models will be coming soon, as well as the ability to mix-and-match losses and local
regularizations. It is also easy to develop your own base models; see `cvxstrat/models.py'
for some examples.

### Graph
The graphs must be made using `networkx` Graphs. You can look more into how to build a `networkx`
Graph [here](https://networkx.github.io/documentation/stable/tutorial.html#).

### Data
Data `X` and `Y` must be Python lists with entries being the inputs/outputs, respectively.
`Z` must be a Python list with entries being tuples of the same form as the specified `networkx`
Graph.

For example, we fit a ridge regression model with local regularization, 
stratified based on time and location:

```
import cvxstrat
import networkx as nx

# G is the cartesian product of a cycle graph and a path graph
G_time = nx.cycle_graph(10)
G_location = nx.path_graph(20)
G = cvxstrat.cartesian_product([G_time, G_location])

# Get the data
X, Y, Z = get_data()
Xtest, Ytest, Ztest = get_test_data()

# Construct the model
m = cvxstrat.RidgeRegression(lambd=0.05)
m.fit(X, Y, Z, G)

# Use the model to make predictions, 
# in this case the average negative log likelihood (ANLL)
predictions = m.anll(Xtest, Ytest, Ztest)
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

## Citing `cvxstrat`

If you use `cvxstrat`, please cite the following paper:

```
@article{tuck2019distributed,
    author       = {Tuck, J. and Barratt, S. and Boyd, S.},
    title        = {A distributed method for fitting Laplacian regularized stratified models},
    journal      = {arXiv preprint arXiv:1904.12017},
    year         = {2019},
}

```
