import random
import numpy as np

from neural_net_base import NeuralNet

def test_xor():
    learning_rate = 0.2
    structure = {'num_inputs': 2, 'num_hidden': 2, 'num_outputs': 1}
    candidate = NeuralNet(structure, learning_rate)

    labeled_data = [
        (np.array([0,0]), np.array([0])),
        (np.array([0,1]), np.array([1])),
        (np.array([1,0]), np.array([1])),
        (np.array([1,1]), np.array([0]))
    ]

    iterations = 15000

    trainX, trainY = zip(*labeled_data)
    trainX = np.array(trainX)
    trainY = np.array(trainY)

    candidate.train(trainX, trainY, iterations)

    cand_error = candidate.test(trainX, trainY)
    print "XOR Error: ", cand_error