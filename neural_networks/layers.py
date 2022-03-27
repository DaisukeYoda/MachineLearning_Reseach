import numpy as np
from optimizers import Gradient_decent


class ReLU:
    def __init__(self,n_input,n_output,bias=True,initialization = "Xavier"):
        if initialization == "Xavier":
          self.weight = np.random.randn(n_input,n_output)/np.sqrt(n_input)
        elif initialization == "He":
          self.weight = 1.4142*np.random.randn(n_input,n_output)/np.sqrt(n_input)
        else:
          raise ValueError
          
        self.bias = np.ones(n_output)*bias
        
        self.optimizer = Gradient_decent({'eta':0.001})
        self.bn = False
        
    def set_batch_normalization(self):
    	from preprocessing import Batch_normalization
    	
    	self.bn = Batch_normalization()
    	
    	
    def set_optimizer(self,optimizer):
    	self.optimizer = optimizer
        
    def forward_propagation(self,dataX):
        self.input = dataX
        
        res = np.dot(dataX,self.weight) + self.bias
        if self.bn:
        	res = self.bn.forward_propagation(res)
        self.output = res.clip(0)
        
        return self.output
        
    def back_propagation(self,delta,eta=0.001):
        res = (self.output > 0).astype(np.float)
        if self.bn:
        	res = self.bn.back_propagation(res)
        delta *= res
        weight = np.copy(self.weight)
        grad = np.dot(self.input.T,delta)
        self.weight = self.optimizer(weight,grad)
        self.bias -= eta * np.sum(delta, axis=0)
        
        return np.dot(delta,weight.T)
        
        
class Tanh_Layer:
    def __init__(self,n_input,n_output,bias=True,initialization = "Xavier"):
        self.weight = np.random.randn(n_input,n_output)/np.sqrt(n_input)
        
        self.bias = np.ones(n_output)*bias
        self.optimizer = Gradient_decent({'eta':0.001})
        
    def set_optimizer(self,optimizer):
    	self.optimizer = optimizer
    	
    def forward_propagation(self,dataX):
        self.input = dataX
        res = np.dot(dataX,self.weight) + self.bias
        self.output = np.tanh(res)
        
        return self.output
        
    def back_propagation(self,delta,eta=0.001):
        res = 1./(np.cosh(self.output))**2
        delta *= res
        weight = np.copy(self.weight)

        self.weight -= eta*np.dot(self.input.T,delta)
        self.bias -= eta*np.sum(delta,axis=0)
        
        return np.dot(delta,weight.T)
        
class Sigmoid_Layer:
    def __init__(self,n_input,n_output,bias=True,initialization = "Xavier"):
        self.weight = np.random.randn(n_input,n_output)/np.sqrt(n_input)
        
        self.bias = np.ones(n_output)*bias
        self.optimizer = Gradient_decent({'eta':0.001})
 
    def set_optimizer(self,optimizer):
    	self.optimizer = optimizer
        
    def forward_propagation(self,dataX):
        self.input = dataX
        res = np.dot(dataX,self.weight) + self.bias
        self.output = 1./(1. + np.exp(-res))
        
        return self.output
        
    def back_propagation(self,delta,eta=0.001):
        res = delta * (1. - self.output) * self.output
        delta *= res
        
        weight = np.copy(self.weight)
        self.weight -= eta*np.dot(self.input.T,delta)
        self.bias -= self.eta*np.sum(delta,axis=0)
        
        return np.dot(delta,weight.T)
