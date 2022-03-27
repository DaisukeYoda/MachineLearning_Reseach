

import numpy as np


def hold_out(datasetX,datasetY=None,rate=0.8):
	"""
	this method create training_data and test_data.
	dataset: numpy array
	rate: the rate of training_data to entire_data.
	"""
	
	if datasetY is not None and datasetX.shape[0] != datasetY.shape[0]:
		raise AssertionError
	else:
		num_data = datasetX.shape[0]
		
	trainX,testX = np.split(datasetX,[int(num_data*rate)])
	
	if datasetY is not None:
		trainY,testY = np.split(datasetY,[int(num_data*rate)])
		return trainX,testX,trainY,testY
	else:
		return trainX,testX
		

def cross_validation(datasetX,datasetY,k_folds):
	if datasetX.shape[0] != datasetY.shape[0]:
		raise AssertionError
	else:
		num_data = datasetX.shape[0]
		
	x_dim = datasetX.shape[1]
	try:
		y_dim = datasetY.shape[1]
	except:
		datasetY = np.vstack(datasetY)
		y_dim = 1
		
	x_sample = np.split(datasetX,k_folds)
	y_sample = np.split(datasetY,k_folds)
	
	i = 0
	for testX,testY in zip(x_sample,y_sample):
		trainX = np.delete(x_sample,i)
		trainY = np.delete(y_sample,i)
		
	
	
	return x_sample,y_sample
	
	
if __name__ == "__main__":
	dataX = np.array([[1,1,1],[1,1,0],[1,0,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]])
	dataY = np.array([1,1,1,0,1,1,1,0])
	X,Y = cross_validation(dataX,dataY,4)
	print(X)
	print(Y)
