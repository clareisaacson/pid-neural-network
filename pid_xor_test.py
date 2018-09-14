import random
import numpy as np

from pid_neural_net_xor import NeuralNet

def test_my_xor():
    learning_rate = 0.2
    num_hidden1 = 3
    num_hidden2 = 3
    candidate = NeuralNet(num_hidden1, num_hidden2, learning_rate)

    labeled_data = [
        (np.array([0,0]), np.array([0])),
        (np.array([0,1]), np.array([1])),
        (np.array([1,0]), np.array([1])),
        (np.array([1,1]), np.array([0]))
    ]

    iterations = 1000

    trainX, trainY = zip(*labeled_data)
    trainX = np.array(trainX)
    trainY = np.array(trainY)

    candidate.train(trainX, trainY, iterations)

    cand_error = candidate.test(trainX, trainY)
    print "XOR Error: ", cand_error