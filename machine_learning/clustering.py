# -*- coding: utf-8 -*-
"""
Created on Fri Aug 04 02:26:53 2017

@author: daisuke yoda
"""
import numpy as np


class Kmeans_Clustering:
    
    def __init__(self,data,n_cluster):
        self.data = np.array(data)
        self.N = n_cluster
        
        n_data = len(data)
        label = np.random.randint(0,self.N,n_data)       
        self.label = label
        
    def calc_centroid(self):
        centroid = [np.mean(self.data[self.label==i]) for i in range(self.N)]
                    
        return np.array(centroid)
        
    def allocate(self):
        centroid = self.calc_centroid()
        new_label = []
        for X in self.data:
            err = np.abs(centroid - X)
            
            label = np.argmin(err)
            new_label.append(label)
            
        return np.array(new_label)
        
    def activate(self,epoch):
        for i in range(epoch):
            new_label = self.allocate()
            self.label = new_label
            
    def output(self):
        for i in range(self.N):
            print('Label{}:{}'.format(i+1,self.data[self.label==i]))
            
if __name__ == '__main__':      
    
    import seaborn as sns
    iris = sns.load_dataset("iris")
    
    from sklearn import datasets
    iris = datasets.load_iris()

    
    data = iris.data
    data = data[:,:2]
    KC = Kmeans_Clustering(data,2)
    KC.activate(10000)
    KC.output()
    
    import matplotlib.pyplot as plt
      
    plt.plot(KC.data[KC.label==0],'ro')  
    plt.plot(KC.data[KC.label==1],'bo')
    #plt.plot(KC.data[KC.label==2],'b+')
    #plt.plot(KC.data[KC.label==3],'r+')
    
    
    
