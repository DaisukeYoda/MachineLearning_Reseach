from layers import *
from optimizers import *

class Neural_Network:
    def __init__(self,cost_fnc,initialization="Xavier"):
        self.layers = []
        self.cost_fnc = cost_fnc
        self.init = initialization
    
    def connect(self,*layers):
        for layer in layers:
            self.layers.append(layer)
        
    def build(self,structure,fnc):
        n_connect = len(structure) - 1 
        
        if fnc == 'ReLU':
            for i in range(n_connect):
                layer = ReLU(structure[i],structure[i+1],bias=True,initialization=self.init)
                self.layers.append(layer)
        elif fnc == 'Tanh':
            for i in range(n_connect):
                layer = Tanh_Layer(structure[i],structure[i+1])
                self.layers.append(layer)
        elif fnc == 'Sigmoid':
        	for i in range(n_connect):
        		layer = Sigmoid_Layer(structure[i],structure[i+1])
        		self.layers.append(layer)
        else:
            raise ValueError
                
        return True
        
    def set_optimizers(self,method,params={'eta':0.001,'momentum':0.9,'beta':0.9,'alpha':0.001,'beta1':0.9,'beta2':0.999}):
    	
    	if method == "AdaGrad":
    		for layer in self.layers:
    			optimizer = AdaGrad(params)
    			layer.set_optimizer(optimizer)
    			
    	elif method == "RMSprop":
    		for layer in self.layers:
    			optimizer = RMSprop(params)
    			layer.set_optimizer(optimizer)  
    			
    	elif method == "Adadelta":
    		for layer in self.layers:
    			optimizer = Adadelta(params)
    			layer.set_optimizer(optimizer)
    	
    	elif method == "Adam":
    		for layer in self.layers:
    			optimizer = Adam(params)
    			try:
    				layer.set_optimizer(optimizer)
    			except:
    				pass
    			
    	else:
    		return False
    		
    def set_batch_normalization(self):
    	for layer in self.layers:
    		layer.set_batch_normalization()
        
    def fit(self,trainX,trainY):
    
        for layer in self.layers:
            trainX = layer.forward_propagation(trainX)

        delta = self.cost_fnc.delta(trainX,trainY)
        for layer in reversed(self.layers):
            delta = layer.back_propagation(delta)
            
            
        return delta
        
    def predict(self,dataX):
        for layer in self.layers:
            dataX = layer.forward_propagation(dataX)
            
        return dataX
        
    def calc_accuracy(self,dataX,dataY):
        pred = self.predict(dataX)
        return np.sum(np.argmax(pred)==np.argmax(dataY))/np.float(pred.shape[0])
       
    def calc_error(self,trainX,trainY):
        pred = self.predict(trainX)
        ans = trainY
        
        return  self.cost_fnc(pred,ans)
