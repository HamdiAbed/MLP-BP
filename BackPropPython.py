#!/usr/bin/env python
# coding: utf-8

# # Copyright¶
# 
# Jelen iPython notebook a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott "Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült. 
# A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
# Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning
# Jelen notebook Nicolas P. Rougier munkája alapján készült, melyet BSD licensz véd: http://www.labri.fr/perso/nrougier/downloads/mlp.py
# 
# A notebook bármely részének újra felhasználása, publikálása csak a szerzők írásos beleegyezése esetén megegengedett.
# 
# 2018 (c) Gyires-Tóth Bálint (toth.b kukac tmit pont bme pont hu), Császár Márk
# 
# 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 

# locations of true values
true_x = [1, 0]
true_y = [0, 1]

# Locations of false values
false_x = [0, 1]
false_y = [0, 1]

# Plotting trues in green , and falses in red
plot1 = plt.plot(true_x, true_y,'gs')
plot2 = plt.plot(false_x, false_y, 'rs')

# setting the axises of x and y
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

# Showing the grid of the plot
plt.grid(True)

# chosing the dashed lines for the axises
plt.axhline(0, linestyle='dashed')
plt.axvline(0, linestyle='dashed')

# showin the figure
plt.show()


# In[4]:


import numpy as np
from sklearn import preprocessing
import copy


# In[5]:


# writing the activation function formula
def activation(x):
    return 1 / (1 + np.exp(-x))

#plotting the activation function
segedX=np.linspace(-6,6,200)
# axis limits
plt.xlim(-6, 6)
plt.ylim(-0.5, 1.5)
# dashed lines
plt.axhline(0, linestyle='dashed')
plt.axvline(0, linestyle='dashed')
plt.plot(segedX,activation(segedX))


# In[6]:


# definingt the derivative of the activation function
def dactivation(x):
    return np.exp(-x)/((1+np.exp(-x))**2)
plt.plot(segedX,dactivation(segedX))

""" Here is all the magic, defining the MLP
- the class MLP should have 4 functions:
    - Initialization
    - Reset the weights
    - forward step
    - Back propagations step
"""
# In[7]:


class MLP:
    #initialization of the network and setting the number of layers
    def __init__(self, *args):
        #random seeds for weights generation
        np.random.seed(123)
        #number of layers
        self.shape = args
        n = len(args)
        # Creating the layers
        self.layers = []
        # creating input layer + 1 for bias 
        self.layers.append(np.ones(self.shape[0]+1))
        # Creating hidden layer and output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))
        # creating weights matrix
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))
        # dw is the change in the weights we will use it later for momentum method
        self.dw = [0,]*len(self.weights)
        # Re-initializing the weights
        self.reset()

    # This function is to re-initialize the weights
    def reset(self):
        for i in range(len(self.weights)):
            # each weight is a random number with 0.1 difference
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            # the range of weights is from -1 to 1
            self.weights[i][...] = (2*Z-1)*1

    # This function is for feed forward
    def propagate_forward(self, data):
        # Input layer's set up (teaching layer)
        self.layers[0][0:-1] = data
        # Feeding the data in the direction from input to output layer
        # Sigmoid function is the activation function of the neurons
        # In the theoritical part, the activation function was denoted with "a"
        for i in range(1,len(self.shape)):
            self.layers[i][...] = activation(np.dot(self.layers[i-1],self.weights[i-1]))
        # Returns the network estimated results
        return self.layers[-1]

    # Defining backpropagation.
    # The learning rate parameter affects how the network weights
    # modify to the gradient. If this value is too high, then the net
    # "oscillate" around a local or global minimum. If you select too small a value,
    # then it takes considerably more time to reach the best solution or leak out a local
    # minimum and never reach it.

    def propagate_backward(self, target, lrate=0.1):
        deltas = []
        # Calculate the error at the output layer
        error = -(target-self.layers[-1]) # y-y_kalap
        # error*dactivation(s(3))
        delta = np.multiply(error,dactivation(np.dot(self.layers[-2],self.weights[-1])))
        deltas.append(delta)
        # Calculate the gradient for the hidden layer
        for i in range(len(self.shape)-2,0,-1):
            # output to hidden layer delta(2) = delta(3)*(W(2).T)*dactivation(s(2)) (check the lecture please)
            delta=np.dot(deltas[0],self.weights[i].T)*dactivation(np.dot(self.layers[i-1],self.weights[i-1]))
            deltas.insert(0,delta)            
        # Calculate the change in the weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            # pl. last layer's weights : delta(3)*a(2) (check the lecture please)
            dw = -lrate*np.dot(layer.T,delta)
            # weights change
            self.weights[i] += dw 

            # storing the weights change
            self.dw[i] = dw
            # Return the error
        return (error**2).sum()

Now the learning process
# In[8]:


def learn(network, X, Y, valid_split, test_split, epochs=20, lrate=0.1):

        # train-validation-test splitting
        X_train = X[0:int(nb_samples*(1-valid_split-test_split))]
        Y_train = Y[0:int(nb_samples*(1-valid_split-test_split))]
        X_valid = X[int(nb_samples*(1-valid_split-test_split)):int(nb_samples*(1-test_split))]
        Y_valid = Y[int(nb_samples*(1-valid_split-test_split)):int(nb_samples*(1-test_split))]
        X_test  = X[int(nb_samples*(1-test_split)):]
        Y_test  = Y[int(nb_samples*(1-test_split)):]
    
        # Standardization
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test  = scaler.transform(X_test)
    
        # data shuffling in the same order for data and targets to perform batch grad later
        randperm = np.random.permutation(len(X_train))
        X_train, Y_train = X_train[randperm], Y_train[randperm]
        randperm = np.random.permutation(len(X_valid))
        X_valid, Y_valid = X_valid[randperm], Y_valid[randperm]
        randperm = np.random.permutation(len(X_test))
        X_test, Y_test = X_test[randperm], Y_test[randperm]
        
        best_valid_err = np.inf #because accessing the constants is faster than variables
        es_counter = 0 # early stopping counter
        best_model = network
    
        # Now is the training, we will feed epochs to the network with randomly selected batches.
        for i in range(epochs):
            # This solution uses the method that you specified.
            # we go through the training data and send all the items first
            # on the network and then distribute the resulting difference
            # expected results. This is called stochastic gradient descent.
            train_err = 0
            for k in range(X_train.shape[0]):
                network.propagate_forward( X_train[k] )
                train_err += network.propagate_backward( Y_train[k], lrate )
            train_err /= X_train.shape[0]

            # Validation phase
            valid_err = 0
            o_valid = np.zeros(X_valid.shape[0])
            for k in range(X_valid.shape[0]):
                o_valid[k] = network.propagate_forward(X_valid[k])
                valid_err += (o_valid[k]-Y_valid[k])**2
            valid_err /= X_valid.shape[0]

            print("%d epoch, train_err: %.4f, valid_err: %.4f" % (i, train_err, valid_err))

        # Testing phase
        print("\n--- Testing ---\n")
        test_err = 0
        o_test = np.zeros(X_test.shape[0])
        for k in range(X_test.shape[0]):
            o_test[k] = network.propagate_forward(X_test[k])
            test_err += (o_test[k]-Y_test[k])**2
            print(k, X_test[k], '%.2f' % o_test[k], ' (Retrieved: %.2f)' % Y_test[k])
        test_err /= X_test.shape[0]

        fig1=plt.figure()
        plt.scatter(X_test[:,0], X_test[:,1], c=np.round(o_test[:]), cmap=plt.cm.cool)
        


# In[9]:


# Creating the NN with 2 inputs , 10 hidden neurons and 1 output
network = MLP(2,10,1)


# In[10]:


# Training, validating and testing the neural net on the dataset (Noise loaded XOR data)
nb_samples=1000
X = np.zeros((nb_samples,2))
Y = np.zeros(nb_samples)
for i in range(0,nb_samples,4):
    noise = np.random.normal(0,1,8)
    X[i], Y[i] = (-2+noise[0],-2+noise[1]), 0
    X[i+1], Y[i+1] = (2+noise[2],-2+noise[3]), 1
    X[i+2], Y[i+2] = (-2+noise[4],2+noise[5]), 1
    X[i+3], Y[i+3] = (2+noise[6],2+noise[7]), 0

# and finally plotting the results
fig1=plt.figure()
plt.scatter(X[:,0],X[:,1],c=Y[:], cmap=plt.cm.cool)


# In[11]:


#training, and start testing
network.reset()
learn(network, X, Y, 0.2, 0.1)


# In[ ]:




