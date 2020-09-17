#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
# X = (smear width, middle smear color density, upper and lower color density avg), y = should be protein concentration hm
X = np.array(([25, 25, 12], [50, 50,24], [100, 100,48]), dtype=float)
y = np.array(([24], [49], [98]), dtype=float)

X = X/np.amax(X, axis=0) #maximum of X array
y = y/100 # maximum protein conc. is 100ng in scale

class NeuralNetwork(object):
    def __init__(self):
        #parameters
        self.inputSize = 3
        self.outputSize = 1
        self.hiddenSize = 81
        
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
        
    def feedForward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2) 
        output = self.sigmoid(self.z3)
        return output
    
    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))
    
    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)
        
        self.z2_error = self.output_delta.dot(self.W2.T) 
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True) 
        
        self.W1 += X.T.dot(self.z2_delta) #
        self.W2 += self.z2.T.dot(self.output_delta) 
        
    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)
        
NN = NeuralNetwork()

for i in range(1000): #trains repetition
    if (i % 100 == 0): #show loss everi x times
        print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
    NN.train(X, y)
        

dt1 = str(input("User Input One: "))
dt2 = str(input("User Input Two: "))
dt3 = str(input("User Input Three: "))
print("Input: ", str(X))
print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
print("\n")
print("Predicted Output: " + str(NN.feedForward(X)))


# In[ ]:





# In[ ]:




