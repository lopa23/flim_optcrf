 #Code from Chapter 14 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
import torch
class kmeans:
    """The k-Means Algorithm implemented as a neural network"""
    def __init__(self,k,nEpochs=100,eta=0.25):
        self.nData = 100
        self.nDim = 5
       
        self.k = k
        self.nEpochs = nEpochs
        self.weights = np.random.rand(self.nDim,self.k)
        self.eta = eta
        
    def kmeanstrain(self,data):
        self.nData = np.shape(data)[0]
        self.nDim = np.shape(data)[1]
        self.weights = np.random.rand(self.nDim,self.k)
        # Preprocess data (won't work if (0,0,...0) is in data)
        #data=data.numpy()
        print(data.shape)
        normalisers = np.sqrt(np.sum(data**2,axis=1))*np.ones((1,np.shape(data)[0]))
        normalisers[normalisers==0]=1
        data = np.transpose(np.transpose(data)/normalisers)

        for i in range(self.nEpochs):
            #print(i)
            for j in range(self.nData):
                #print(j)
                activation = np.sum(self.weights*np.transpose(data[j:j+1,:]),axis=0)
                winner = np.argmax(activation)
                self.weights[:,winner] += self.eta * data[j,:] - self.weights[:,winner]            
            
    def kmeansfwd(self,data):
        self.nData = np.shape(data)[0]
        self.nDim = np.shape(data)[1]
        best = np.zeros(np.shape(data)[0])
        for i in range(np.shape(data)[0]):
            activation = np.sum(self.weights*np.transpose(data[i:i+1,:]),axis=0)
            best[i] = np.argmax(activation)
        return best

def main():
    dtype = 'float32' 
    torchtype = {'float32': torch.float32, 'float64': torch.float64}
    N, D, K = 100, 2, 5



    x = np.random.rand(N, D) / 6 + .5
    Km=kmeans(K);
    Km.kmeanstrain(x);
    sol=Km.kmeansfwd(x)
    print(sol)
#main();
