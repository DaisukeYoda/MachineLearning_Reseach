import numpy as np

"""
INPUTとDELTAが渡される。
DELTA=phiL/phiW

パラメータを辞書で保存
params={'eta':0.001,'momentum':0.7,'beta':0.9,'alpha':0.001,'beta1':0.9,'beta2':0.999}
__init__(self,params={eta:0.0001, momentum:0.7})
"""

class Gradient_decent:
	def __init__(self,params={'eta':0.001}):
		self.eta = params['eta']
		self.momentum = 1.
		
	def __call__(self,weight,grad):
		weight = self.momentum * weight - self.eta*grad
		
		return weight
		
		
class Momentum(Gradient_decent):
	def __init__(self,params={'eta':0.001,'momentum':0.9}):
		Gradient_decent.__init__(self,params)
		
	def __del__(self):
		return True
		
		
class AdaGrad:
	def __init__(self,params={'eta':0.001}):
		self.eta = params['eta']
		self.h = None
				
	def __call__(self,weight,grad):
		if self.h == None:
			self.h = np.zeros_like(weight)
		
		self.h += grad * grad
		weight -= self.eta * grad/np.sqrt(self.h + 1e-7)
		
		return weight 
		
class RMSprop:
	def __init__(self,params={'beta':0.9,'eta':0.001}):
			self.h = None
			self.beta = params['beta']
			self.eta = params['eta']
			
	def __call__(self,weight,grad):
		if self.h == None:
			self.h = np.zeros_like(weight)
		
		self.h = self.beta * self.h + (1 - self.beta) * (grad * grad)
		weight -= self.eta * grad/np.sqrt(self.h + 1e-7)
		
		return weight 
		
class Adadelta:
	def __init__(self,params={'beta':0.9}):
		self.beta = params['beta']
		self.h = None
		self.s = None
		
	def __call__(self,weight,grad):
		if self.h == None:
			self.h = np.zeros_like(weight)
			self.s = np.zeros_like(weight)
			
		self.h = self.beta * self.h + (1 - self.beta) * (grad * grad)
		v = ((np.sqrt(self.s) + 1e-7)/(np.sqrt(self.h) + 1e-7)) * grad
		self.s = self.beta * self.s + (1 - self.beta) * (v * v)
		weight -= v
		
		return weight				

class Adam:
	def __init__(self,params={'alpha':0.001,'beta1':0.9,'beta2':0.999}):
		self.m = 0
		self.v = 0
		self.t = 0
		self.alpha = params['alpha']
		self.beta1 = params['beta1']
		self.beta2 = params['beta2']
				
	def __call__(self,weight,grad):
		self.t += 1
		self.m = self.beta1 * self.m + (1 - self.beta1) * grad
		self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
		m_hat = self.m / (1 - self.beta1**self.t)
		v_hat = self.v / (1 - self.beta2**self.t)
		weight -= self.alpha * m_hat / (np.sqrt(v_hat) + 1e-8)
		
		return weight		
		
	

