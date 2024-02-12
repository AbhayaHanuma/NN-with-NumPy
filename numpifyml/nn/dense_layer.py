import numpy as np
from numpifyml.nn.utils import *

#Layer class
class Layer:
    
    activationFunctions = {
            'tanh' : (tanh,d_tanh),
            'sigmoid' : (sigmoid,d_sigmoid),
            'relu' : (relu, d_relu)
        }
    np.random.seed(3)
    
    def __init__(self, inputs, neurons, activation, learning_rate=0.002):
        '''
        inputs = No of inputs to the current layer
        neurons = No of neurons in the current layer
        activation = Activation function for the currenrt layer
        '''
        self.W,self.b = initializeWeights(nx=inputs, ny=neurons)
        self.act, self.d_act = self.activationFunctions.get(activation)
        self.learning_rate = learning_rate
    
    def forwardProp(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.W, self.A_prev) + self.b
        self.A = self.act(self.Z)
        return self.A
    
    def backProp(self, dA):
        dZ = np.multiply(self.d_act(self.Z),dA)
        dW = 1/dZ.shape[1] * (np.dot(dZ,self.A_prev.T))
        db = 1/dZ.shape[1] * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T,dZ)
        
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        
        return dA_prev