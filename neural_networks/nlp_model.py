__author__ = 'Daisuke Yoda'


import numpy as np
from layers import *
from loss_functions import *
from neural_network import Neural_Network
from datasets.mnist import load_mnist

def one_hot_encoding(figures,n_class):
    return np.eye(n_class)[figures]
    
def char2vec(character):
	char_num = ord(character) - 97
	return one_hot_encoding(char_num,26)
	
def vec2char(vector):
	vec_num = np.argmax(vector) + 97
	return chr(vec_num) 


if __name__ == '__main__':
	
	w = char2vec('w')
	o = char2vec('o')
	r = char2vec('r')
	k = char2vec('k')
	
	trainX = np.vstack([w,o,r])
	trainY = np.vstack([o,r,k])
	cost_fnc = SumSquaresError()
	structure = (26,50,26)
	net = Neural_Network(cost_fnc)
	net.build(structure,'ReLU')
	net.set_optimizers("Adam")
	for i in range(500):
		net.fit(trainX,trainY)
		
	res = net.predict(o)
	print(vec2char(res))
	
	
