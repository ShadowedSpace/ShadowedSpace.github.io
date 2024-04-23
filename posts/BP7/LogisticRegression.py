import torch

class LinearModel:
    def __init__(self):
        self.w = None

    def score(self, X):
        return X@self.w

    def predict(self, X):
        return self.score(X) > 0

class LogisticRegression(LinearModel):
    def __init__(self, w):
        self.w = w

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def loss(self, X, y):
        s = self.score(X)
        sigma = self.sigmoid(s)

        first_half = -sigma.log()*y
        second_half = -(1 - sigma).log() * (1 - y)

        return (1 / X.size(0)) * (first_half + second_half).sum()

    def grad(self, X, y):
        s = self.score(X)
        sigma = self.sigmoid(s)
        sigma = sigma[:, None]

        y = y[:, None]

        grad = (sigma - y) * X

        return grad.mean(dim=0)

class GradientDescentOptimizer:
    def __init__(self, model, w, w_prev):
        self.model = model 
        self.w = w
        self.w_prev = w_prev

    def step(self, X, y, alpha, beta):
        temp = self.model.w - alpha * (self.model.grad(X, y)) + beta * (self.model.w - self.w_prev)
        self.w_prev = self.model.w
        self.model.w = temp
