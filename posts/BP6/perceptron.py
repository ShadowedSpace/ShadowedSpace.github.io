import torch

class LinearModel:
    def __init__(self):
        self.w = None 

    def score(self, X):
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        return X@self.w

    def predict(self, X):
        return self.score(X) > 0

class Perceptron(LinearModel):
    def loss(self, X, y):
        return (1.0 * ((self.score(X) * y) < 0)).mean()

    def grad(self, X, y):
        s = self.score(X)
        exp = ((s * y) < 0)
        grad = exp * y * X
        return grad.mean(dim=0)

class PerceptronOptimizer:
    def __init__(self, model, k = 1):
        self.model = model
        self.k = k
    
    def step(self, X, y):
        #self.model.loss(X, y)
        self.model.w += self.model.grad(X, y).squeeze()