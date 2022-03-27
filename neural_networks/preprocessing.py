
import numpy as np

def get_minibatch(dataX,dataY,batch_size=10):
    data_size = dataX.shape[0]
    idx = np.random.choice(data_size,batch_size)
    
    batchX = dataX[idx]
    batchY = dataY[idx]
    
    return batchX,batchY
    
    
def one_hot_encoding(figures,n_class):
    return np.eye(n_class)[figures]
    
    
def normalize(data):                                                                                                                                                                                                                        
    vmax = np.max(data)
    vmin = np.min(data)
    
    return (data - vmin)/(vmax-vmin)
    
def standard(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    z = (x-xmean)/xstd
    
    return z
    
class Dropout:
	def __init__(self,ratio=0.5):
		self.ratio = ratio
	
class Batch_normalization:
	def __init__(self):
		self.gamma = 1.
		self.beta = 0.
	
	def forward_propagation(self,affine):
		self.affine = affine
		xmean = affine.mean(axis=0)
		xstd = np.std(affine,axis=0)
		z = (affine - xmean)/xstd
		
		return z
		
	def back_propagation(self,delta):
		N = self.affine.shape[0]
		xmean = self.affine.mean(axis=0)
		xstd = np.std(self.affine,axis=0)
		xvar = np.var(self.affine,axis=0)		
		f3 = self.affine - xmean
		d12a = delta * self.gamma	
		d3a = (1/xstd) * (d12a - f3 * (1/xstd) * (1/N) * np.sum(f3 * d12a,axis=0))
		
		dx = d3a - np.sum(d3a,axis=0)/N
		self.gamma -= np.sum(((self.affine - xmean)/xstd) * delta ,axis=0)
		self.beta -= np.sum(delta,axis=0)
		
		return dx
	
if __name__ == '__main__':
	"""
	bn = Batch_normalization()
	inputX = np.random.random([3,5])	
	bn.forward_propagation(inputX)
	delta = np.random.random([3,5])
	dx = bn.back_propagation(delta)
	print(dx)
	"""
	
	from neural_network import Neural_Network
	from layers import *
	from loss_functions import SumSquaresError
	
	cost = SumSquaresError()
	net = Neural_Network(cost)
	
	
	l1 = ReLU(2,10)
	l2 = Batch_normalization()
	l3 = ReLU(10,1)
	net.connect(l1,l3)
	net.set_optimizers("Adam")
	
	x = np.array([[1,1],[1,0],[0,1],[0,0]])
	y = np.array([[1],[1],[1],[0]])
	
	for i in range(5000):
		net.fit(x,y)
		
	print(net.predict([1,1]))
	print(net.predict([1,0]))
	print(net.predict([0,1]))
	print(net.predict([0,0]))
