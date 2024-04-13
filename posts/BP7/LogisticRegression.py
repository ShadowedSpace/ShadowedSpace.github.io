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

    def sigmoid(s):
        return 1 / (1 + torch.exp(s))

    def loss(self, X, y):
        s = self.score(X)
        sigma = self.sigmoid(s)

        first_half = sigma.log()*y
        second_half = (1 - sigma).log() * (1 - y)

        return (1 / X.size) * (first_half + second_half).sum()
    
    def grad(self, X, y):
        # v_ = v[:, None]
        s = self.score(X)
        sigma = self.sigmoid(s)

        return (1 / X.size) * (sigma - y) * X
    
class LogisticOptimizer:
    def __init__(self, model, w):
        self.model = model 
        self.w = None
        self.w_prev = None

    def step(self, X, y, alpha, beta):
        return self.w - alpha * (self.model.grad(X, y)) + beta * (self.w - self.w_prev)