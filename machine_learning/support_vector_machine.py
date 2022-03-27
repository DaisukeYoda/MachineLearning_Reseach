# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:55:21 2017

@author: daisuke yoda
"""

import numpy as np

def unit_step_function(x):
    """単位ステップ関数
    f(x) = 1 if x >= 0
    f(x) = -1 if x < 0
    """
    
    if x >= 0:
        return 1
    else:
        return -1

class SVM:
    def __init__(self,trainX,trainY):
        self.n_data = trainX.shape[0]#データ数
        self.L = np.zeros(trainX.shape[0])#初期パラメーター
        self.trainX = trainX#学習させたいデータ
        self.trainY = trainY#教師データ
        
    
    def Lagrange_dash(self,i):
        """ラグランジェ乗数の最適化
        通常は線型計画問題を解いてサポートベクトルを決定。
        今回は最急降下法を用いる。
        →ラグランジュ乗数を偏微分して学習量を決定   
        """
        
        X = self.trainX
        Y = self.trainY
        
        gL = 0
        for j in range(self.n_data):
            gL += self.L[j]*Y[i]*Y[j]*np.dot(X[i],X[j])      

        return 1 - gL
        
    def activate(self,eta):
        C = 10
        
        for i in range(self.n_data):
            self.L[i] +=  eta*self.Lagrange_dash(i)
            
            """ラグランジュ問題の制約条件について
            本来は∑λy=0の制約条件を学習の更新の度に満すようにしなければ
            ならないが、各々のラグランジュ乗数に上限を設けつつ、疑似的に正規化すること
            によって、制約条件を表現
            """
            if self.L[i] > C:
                self.L[i] = C
                
            elif self.L[i] < 0:
                self.L[i] = 0        
                
        self.L = self.L*(1.0/np.dot(self.L,self.trainY))
        
        
    def train(self,train_epoch=100,eta=0.01):
        for i in range(train_epoch):
            self.activate(eta)
        
    def calculate_param(self):
        """
        ラグランジュ乗数が0以外の値をとるものがサポートベクトル
        ラグランジュの偏微分を計算することでw（重み）とb（定数項）を決定
        なお、定数項に関しては、ラグランジュ乗数が収束していれば、
        任意のサポートベクトルに対して同じ値をとことを利用
        s：任意のサポートベクトル
        """
        support_vector = np.where(self.L != 0)[0]
        
        w = 0
        for i in range(self.n_data):
            w += self.L[i]*self.trainY[i]*self.trainX[i]
            
        s = support_vector[0]
        b = self.trainY[s] - np.dot(w,self.trainX[s])
    
        self.w = w
        self.b = b
        
        return support_vector
        
    def predict(self,X):
        
        return unit_step_function(np.dot(self.w,X) + self.b)
    
 
if __name__ == '__main__':
            
    trainX = np.array([[1,1],[1,0],[0,1],[0,0]])
    trainY = np.array([[1],[1],[1],[-1]])           
    svm = SVM(trainX,trainY)
    svm.train(100,0.01)
    svm.calculate_param()
    result = svm.predict([0,0])
    
    print(result)