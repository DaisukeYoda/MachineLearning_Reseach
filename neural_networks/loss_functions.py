import numpy as np


class SumSquaresError:
    def __call__(self,X,Y):   
        return 0.5*np.sum((X - Y)**2)
        
    def delta(self,X,Y):
        return X - Y
        
        
class SumAbsoluteError(object):
    def __call__(self,X,Y):
        return np.sum(np.abs(X-Y))
        
    def delta(self):
        raise ValueError
        
        
class CrossEntropy:
    def __call__(self,X,Y):
        res = 1./(1. + np.exp(-X))
        return np.sum(-Y*np.log(res) - (1-Y)*np.log(1 - res))
        
    def delta(self,X,Y):
        res = 1./(1. + np.exp(-X))
        return res - Y
        

