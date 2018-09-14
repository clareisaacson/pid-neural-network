import numpy as np
import random

class NeuralNet:
  def __init__(self, structure, learning_rate):
    """
    Initialize the Neural Network.

    - structure is a dictionary with the following keys defined:
        num_inputs
        num_outputs
        num_hidden
    - learning rate is a float that should be used as the learning
        rate coefficient in training

    Initialize weights to random values in the range [-0.05, 0.05].
    Initialize global variables
    """
    self.learn = learning_rate
    self.num_in = structure['num_inputs'] + 1
    self.num_out = structure['num_outputs']
    self.num_hid = structure['num_hidden']
    self.input = []
    self.output = []
    self.hidden = []
    self.weights1 = ((np.random.rand(self.num_in, self.num_hid)) - 0.5) / 10.0
    self.weights2 = ((np.random.rand(self.num_hid, self.num_out)) - 0.5) / 10.0
    

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
    x = np.append(x,[1.0]) # add bias
    hidden_row = np.dot(x, self.weights1)
    hidden_row_activated = self.sigmoid(hidden_row)
    output_row = np.dot(hidden_row_activated, self.weights2)
    output_final = self.sigmoid(output_row)

    self.input = x
    self.hidden = hidden_row_activated
    self.output = output_final

    return output_final

    #old iterative code
    '''
    hidden_row = []
    for hidden1 in range(self.num_hid):
        total = 0
        for inp in range(self.num_in):
            total = total + (self.weights1[inp][hidden1]* x[inp])
        sig = 1/(1+np.exp(-total))
        hidden_row.append(sig)

    output_row = []
    for out in range(self.num_out):
        total = 0
        for hidden2 in range(self.num_hid):
            total = total + (self.weights2[hidden2][out] * hidden_row[hidden2])
        sig = 1/(1+np.exp(-total))
        output_row.append(sig)
    self.input = x
    self.hidden = hidden_row
    self.output = output_row

    return output_row'''

  def back_propagate(self, target):
    """
    Updates the weights of the network based on the last forward propagate

    :param target: the (correct) label of the last forward_propagate call
    :type target: vector
    :return: None
    """

    # calculate error between output layer and hidden layer
    first_step_errors = []
    for i in range(self.num_hid):
        unit_error = 0
        for j in range(self.num_out):
            y = self.output[j]
            x = self.hidden[i]
            t = target[j]
            E = (y-t)
            unit_error += (E*(y*(1-y)))*self.weights2[i][j]
            de_dw = (E * (y * (1-y))) * x
            self.weights2[i][j] = self.weights2[i][j] - self.learn * de_dw # change weights
        first_step_errors.append(unit_error)

    # Calculate error between hidden layer and input layer
    for k in range(self.num_in):
        for l in range(self.num_hid):
            y = self.hidden[l]
            x = self.input[k]
            E = first_step_errors[l]
            de_dw = (E * (y * (1-y))) * x
            self.weights1[k][l] -= self.learn * de_dw # change weights
            
  def train(self, X, Y, iterations=1000):
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
        x = self.forward_propagate(X[i])
        t = Y[i]
        vector_diff = 0
        for j in range(len(x)):
            vector_diff +=  (x[j]-t[j])**2
        total_error += (vector_diff/len(x))
    return total_error/len(Y)
