#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:45:05 2020

@author: nick
"""





import numpy as np


import matplotlib.pyplot as plt
from dnn_app_utils_v3 import *
import random

np.random.seed(1)






#write_data() DONE!  
    
    
train_x, train_y, test_x, test_y, dev_x, dev_y= load_data()





train_x_flatten = train_x.reshape(train_x.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x.reshape(test_x.shape[0], -1).T
dev_x_flatten = dev_x.reshape(dev_x.shape[0], -1).T
# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
dev_x = dev_x_flatten/255




layers_dims = [27360, 20, 15, 4, 7, 5, 1]

def L_layer_model(X, Y, DEV, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    print(X.shape)
    print(Y.shape)
    print(layers_dims)
    print(learning_rate)

    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ##
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###

        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            dev_AL, caches = L_model_forward(DEV["x"], parameters)
            devcost = compute_cost(dev_AL, DEV["y"])
            print("Dev cost afer iteration %i: %f" %(i, devcost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
      
            
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
 
    
    return parameters
DEV = {
   "x": dev_x,
   "y": dev_y    
}
parameters = L_layer_model(train_x, train_y, DEV, layers_dims, num_iterations = 2500, print_cost = True)
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)
predictions_dev = predict(dev_x, dev_y, parameters)
print_mislabeled_images(["female", "male"], test_x, test_y, predictions_test)
#write_image_data()
