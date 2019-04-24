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

To fit, for example, a linear regression model stratified based on time and location:
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
m = cvxstrat.LinearRegression()
m.fit(X, Y, Z, G)

# Use the model to make predictions
predictions = m.score(Xtest, Ytest, Ztest)
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
