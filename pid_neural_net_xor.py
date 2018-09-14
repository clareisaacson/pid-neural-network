import numpy as np
import random

class NeuralNet:
  def __init__(self, num_hidden1, num_hidden2, learning_rate):
    """
    Initialize the Neural Network.

    :param num_hidden1: number of hidden units in the first hidden layer
    :type num_hidden1: int
    :param num_hidden2: number of hidden units in the second hidden layer
    :type num_hidden2: int
    :param learning_rate: float that should be used as the learning
        rate coefficient in training
    :type learning_rate: float 

    Initialize weights to random values in range [-0.05, 0.05]
    Initialize global variables
    """
    self.num_in = 2 + 1 # +1 for bias
    self.num_out = 1
    self.num_hid1 = num_hidden1 
    self.num_hid2 = num_hidden2 
    self.learn = learning_rate
    self.input = []
    self.output = []
    self.hidden1 = []
    self.hidden2 = []
    self.weights1 = ((np.random.rand(self.num_in, self.num_hid1)) - 0.5) /10
    self.weights2 = ((np.random.rand(self.num_hid1, self.num_hid2)) - 0.5) /10
    self.weights3 = ((np.random.rand(self.num_hid2, self.num_out)) - 0.5) /10

  def sigmoid(self, unit):
    """
    Sigmoid activation function

    :param unit: value(s) to be activated.
    :type unit: float, scalar, or vector
    :return type: float
    """
    return 1/(1+np.exp(-unit))
    
  def forward_propagate(self, x):
    """
    Push the input 'x' through the network

    :param x: vecor of inputs to the network
    :type x: vector 
    :return type: vector
    :return: activation on the output nodes.
    """
    x = np.append(x,[1.0])
    #calulate hidden layer 1 vals
    hidden_row1 = np.dot(x, self.weights1)
    hidden_row1_activated = self.sigmoid(hidden_row1)
    #calulate hidden layer 2 vals
    hidden_row2 = np.dot(hidden_row1_activated, self.weights2)
    hidden_row2_activated = self.sigmoid(hidden_row2)
    #calulate output layer vals
    output_row = np.dot(hidden_row2_activated, self.weights3)
    output_final = self.sigmoid(output_row)

    # update global variables for backpropagation
    self.input = x
    self.hidden1 = hidden_row1
    self.hidden2 = hidden_row2
    self.output = output_final

    return output_final

  def back_propagate(self, target):
    """
    Updates the weights of the network based on the last forward propagate

    :param target: the (correct) label of the last forward_propagate call
    :type target: vector
    :return: None
    """
    # Change weights between output layer and hidden layer 2
    first_step_errors = []
    for i in range(self.num_hid2):
        unit_error = 0
        for j in range(self.num_out):
            y = self.output[j]
            x = self.hidden2[i]
            t = target[j]
            E = (y-t)
            unit_error += (E * (y * (1 - y))) * self.weights3[i][j]
            de_dw = (E * (y * (1 - y))) * x
            self.weights3[i][j] = self.weights3[i][j] - self.learn * de_dw # change weight
        first_step_errors.append(unit_error) # activation errors of hidden 2

    # Change weights between hidden layer 2 and hidden layer 1
    second_step_errors = []
    for k in range(self.num_hid1):
        unit_error = 0
        for l in range(self.num_hid2):
            y = self.hidden2[l]
            x = self.hidden1[k]
            E = first_step_errors[l]
            unit_error += (E * (y * (1 - y))) * self.weights2[k][l]
            de_dw = (E * (y * (1 - y))) * x
            self.weights2[k][l] -= self.learn * de_dw # change weight
        second_step_errors.append(unit_error) #activation errors of hidden 1

    # Change weights bewteen hidden layer 1 and input layer
    for m in range(self.num_in):
        for n in range(self.num_hid1):
            y = self.hidden1[n]
            x = self.input[m]
            E = second_step_errors[n]
            de_dw = (E * (y * (1 - y))) * x
            self.weights1[m][n] -= self.learn * de_dw # change weight


  def train(self, X, Y, iterations=100):
    """
    Trains the network on observations X with labels Y.

    :param X: matrix corresponding to a series of observations - each row is
        one observation
    :type X: numpy matrix
    "param Y: matrix corresponding to the correct labels for the observations
    :type Y: numpy matrix
    :return: None
    """
    for it in range(iterations):
        for idx in range(len(X)):
            x = X[idx]
            y = Y[idx]
            self.forward_propagate(x)
            self.back_propagate(y)

  def test(self, X, Y):
    """
    Tests the network on observations X with labels Y.

    :param X: matrix corresponding to a series of observations - each row is
        one observation
    :type X: numpy matrix
    "param Y: matrix corresponding to the correct labels for the observations
    :type Y: numpy matrix
    :return: mean squared error, float
    """
    total_error = 0
    for i in range(len(X)):
        output = self.forward_propagate(X[i])
        target = Y[i]
        diff = 0
        for j in range(len(output)):
            diff +=  (output[j]-target[j])**2
        total_error += (diff/len(output))
    return total_error/len(Y)