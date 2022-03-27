# -*- coding: utf-8 -*-
"""
Created on Mon May 01 23:30:15 2017

@author: daisuke yoda
"""

import numpy as np


def unit_step_function(x):
    """単位ステップ関数
    f(x) = 1 if x >= 0
    f(x) = 0 if x < 0
    """
    
    if x >= 0:
        return 1
    else:
        return 0

        
class Perceptron:
    def __init__(self,trainX,trainY):
        self.weight = np.zeros(1 + trainX.shape[1])#初期パラメーター（誤差項に注意）
        self.trainX = trainX#学習させたいデータ
        self.trainY = trainY#教師データ
    
    def activate(self,eta=0.01):       
        """パーセプトロンの重み付けを行うメソッド
        Widorow-Hoffの学習規則を用いて重みを更新
        eta : 学習率
        """
   
        for X,Y in zip(self.trainX,self.trainY):        
            X = np.append(X,1) #誤差項を加える
            Z = np.dot(X,self.weight) #重みを掛け合わせる（内積）
            Z = unit_step_function(Z)#単位ステップ関数で分類
            
            self.weight += eta*(Y - Z)*X #重みを更新
        
    def train(self,train_epoch=10):
        #学習回数の数だけ重みの更新を行う。
        for i in xrange(train_epoch):
            self.activate()           
            
    def predict(self,data):        
        #データに対する予測
        X = np.append(data,1)
        Y = np.dot(X,self.weight)
        
        return unit_step_function(Y)
            
        
if __name__ == '__main__':
    """今回はAND,OR問題を学習させる。
    f(1,1) = 1
    f(1,0) = 1
    f(0,1) = 1
    f(0,0) = 0
    """
    trainX = np.array([[1,1],[1,0],[0,1],[0,0]])
    trainY = np.array([[1],[1],[1],[0]])
    
    ppn = Perceptron(trainX,trainY)   #パーセプトロンを構築   
    ppn.train(1000) #学習
    result = ppn.predict([1,1])#適当に予測させる
    
    print result
    