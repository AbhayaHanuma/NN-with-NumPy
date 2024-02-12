import numpy as np

#Weights Initialization
def initializeWeights(nx,ny):
    '''
    nx = No of inputs or No of neurons in prev layer
    ny = No of neurons in current layer
    '''
    w = np.random.randn(ny,nx)*0.01
    b = np.zeros((ny,1))
    
    return w,b

#Activation Functions
def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.square(np.tanh(x))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))

def relu(x):
    return np.maximum(0,x)

def d_relu(x):
    return np.where(x > 0, 1, 0)

#Loss Functions
def logloss(y, a):
    return -(y*np.log(a) + (1 - y)*np.log(1 - a))

def d_logloss(y, a):
    return (a - y)/(a*(1 - a))

def getAccuracy(Y,predictions):
    return float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)