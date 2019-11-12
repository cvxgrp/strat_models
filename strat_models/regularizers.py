import numpy as np


class Regularizer:
    """
    Inputs:
            lambd (scalar > 0): regularization coefficient. Default value is 1.

    All regularizers implement the following functions:

    1. evaluate(theta). Evaluates the regularizer at theta.
    2. prox(t, nu, warm_start, pool): Evaluates the proximal operator of the regularizer at theta
    """

    def __init__(self, lambd=1):
        # if lambd < 0:
        # 	raise ValueError("Regularization coefficient must be a nonnegative scalar.")

        self.lambd = lambd

    def evaluate(self, theta):
        raise NotImplementedError(
            "This method is not implemented for the parent class.")

    def prox(self, t, nu, warm_start, pool):
        raise NotImplementedError(
            "This method is not implemented for the parent class.")

# Regularizers


class zero_reg(Regularizer):

    def __init__(self, lambd=0):
        super().__init__(lambd)
        self.lambd = lambd

    def evaluate(self, theta):
        return 0

    def prox(self, t, nu, warm_start, pool):
        return nu


class sum_squares_reg(Regularizer):

    def __init__(self, lambd=1):
        super().__init__(lambd)
        self.lambd = lambd

    def evaluate(self, theta):
        return (self.lambd / 2) * sum(theta**2)

    def prox(self, t, nu, warm_start, pool):
        if self.lambd == 0:
            return nu
        return nu / (1 + t * self.lambd)
        # return nu[:, :, :-1] / (1+t*self.lambd) #so you dont regularize the
        # intercept


class L1_reg(Regularizer):

    def __init__(self, lambd=1):
        super().__init__(lambd)

    def evaluate(self, theta):
        return self.lambd * sum(abs(theta))

    def prox(self, t, nu, warm_start, pool):
        return np.maximum(nu - t * self.lambd, 0) - np.maximum(-nu - t * self.lambd, 0)


class L2_reg(Regularizer):

    def __init__(self, lambd=1):
        super().__init__(lambd)

    def evaluate(self, theta):
        return self.lambd * np.linalg.norm(theta, 2)

    def prox(self, t, nu, warm_start, pool):
        nus = []
        for i in range(nu.shape[0]):
            nus += [nu[i] *
                    np.maximum(1 - t * self.lambd / np.linalg.norm(nu[i], 2), 0)]
        return np.rollaxis(np.dstack(nus), -1)


class elastic_net_reg(Regularizer):

    def __init__(self, lambd=1):
        super().__init__(lambd)

    def evaluate(self, theta):
        return sum(abs(theta)) + (self.lambd / 2) * sum(theta**2)

    def prox(self, t, nu, warm_start, pool):
        return (1 / (1 + t * self.lambd)) * np.maximum(nu - t, 0) - np.maximum(-nu - t, 0)


class neg_log_reg(Regularizer):

    def __init__(self, lambd=1):
        super().__init__(lambd)

    def evaluate(self, theta):

        if theta < 0:
            return np.inf

        return -self.lambd * np.log(theta)

    def prox(self, t, nu, warm_start, pool):
        return (nu + np.sqrt(nu**2 + 4 * t * self.lambd)) / 2


class nonnegative_reg(Regularizer):

    def __init__(self, lambd=1):
        super().__init__(lambd)

    def evaluate(self, theta):
        for theta_i in theta:
            if theta_i < 0:
                return np.inf
        return 0

    def prox(self, t, nu, warm_start, pool):
        nu[nu < 0] = 0
        return nu


class min_threshold_reg_one_elem(Regularizer):

    def __init__(self, lambd=1e-5):
        super().__init__(lambd)

    def evaluate(self, theta):
        if theta < self.lambd:
            return np.inf
        return 0

    def prox(self, t, nu, warm_start, pool):
        return np.maximum(nu, self.lambd)


class clip_reg(Regularizer):

    def __init__(self, lambd=(1e-5, 1 - 1e-5)):
        super().__init__(lambd)

    def evaluate(self, theta):
        if theta > self.lambd[1] or theta < self.lambd[0]:
            return np.inf
        return 0

    def prox(self, t, nu, warm_start, pool):
        return np.clip(nu, self.lambd[0], self.lambd[1])
