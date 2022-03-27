import numpy as np
from layers import *
from loss_functions import *
from neural_network import Neural_Network
from datasets.mnist import load_mnist

        
            
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

if __name__ == '__main__':
            
    (x_train, t_train), (x_test,t_test) = load_mnist(flatten=True,normalize=False)
    
    x_train,x_test = normalize(x_train),normalize(x_test)
    t_train,t_test = one_hot_encoding(t_train,10),one_hot_encoding(t_test,10)
    
    cost_fnc = SumSquaresError()
    structure = (784,50,10)
    net = Neural_Network(cost_fnc)
    net.build(structure,'ReLU')
    net.set_optimizers("Adadelta")
    #net.set_batch_normalization()
    
    for i in range(10):
      trainX,trainY = get_minibatch(x_train,t_train,100)
      for i in range(5):
      	delta = net.fit(trainX,trainY)
          	
    
    acc = 0
    for X,Y in zip(x_test,t_test):
    	res = net.predict(X)
    	if np.argmax(res) == np.argmax(Y):
    		acc += 1
    	
    print(100*acc/len(x_test))
    	
    
