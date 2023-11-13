import numpy as np

from src.utils import d_logloss, logloss

class NNmodel:
    def __init__(self, epochs):
        self.epochs = epochs
        self.layers = []
        self.costs = []

    def add(self, layer):
        self.layers.append(layer)
        
    def fit(self, x_train, y_train):
        for epoch in range(self.epochs):
            A = x_train
            for layer in self.layers:
                A = layer.forwardProp(A)
                
            cost = 1/y_train.shape[1] * np.sum(logloss(y=y_train,a=A))
            self.costs.append(cost)
            dA = d_logloss(y_train, A)
    
            for layer in reversed(self.layers):
                dA = layer.backProp(dA)
            if epoch%100==0:
                print(f'At epoch - {epoch}, cost - {cost:0.10f}')
    
    def predict(self, X):
        A = X
        for layer in self.layers:
            A = layer.forwardProp(A)
        A = np.where(A<0.5, 0, 1)
        return A