import torch

class LinearModel:
    def __init__(self):
        self.w = None

    def score(self, X):
        return X@self.w

    def predict(self, X):
        return self.score(X) > 0

class LogisticRegression(LinearModel):

    # defines w as an instance variable
    def __init__(self, w):
        self.w = w

    # calculates sigmoid for a given score s
    def sigmoid(self, s):
        '''
        Takes in score s 
        Returns sigmoid function for given s
        '''
        return 1 / (1 + torch.exp(-s))

    # calculates empirical risk
    def loss(self, X, y):
        '''
        Takes in data set X and target y
        Returns empirical loss
        '''
        s = self.score(X)
        sigma = self.sigmoid(s)

        first_half = -sigma.log() * y
        second_half = -(1 - sigma).log() * (1 - y)

        return (1 / X.size(0)) * (first_half + second_half).sum()

    # calculates gradient of empirical risk
    def grad(self, X, y):
        '''
        Takes in data set X and target y
        Returns the gradient of empirical risk
        '''
        s = self.score(X)
        sigma = self.sigmoid(s)
        sigma = sigma[:, None]

        y = y[:, None]

        grad = (sigma - y) * X

        return grad.mean(dim=0)

class GradientDescentOptimizer:
    # initializes instance variables model, w, and w_prev
    def __init__(self, model, w, w_prev):
        self.model = model 
        self.w = w
        self.w_prev = w_prev

    # calculates one step of the logistic regression update
    def step(self, X, y, alpha, beta):
        '''
        Takes in feature matrix X, target vector y, learning rate alpha, and momentum beta
        Updates w and w_prev based on parameters
        '''
        temp = self.model.w - alpha * (self.model.grad(X, y)) + beta * (self.model.w - self.w_prev)
        self.w_prev = self.model.w
        self.model.w = temp
