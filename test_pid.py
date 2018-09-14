import random
import numpy as np
from convert_set import parse
from test_set import test_set_str
from train_set import train_set_str

from pid_neural_net import NeuralNet

def run():
  (X,Y) = parse(train_set_str)
  X = np.array(X,dtype=np.float128)
  Y = np.array(Y,dtype=np.float128)
  learning_rate = 0.2
  num_hidden1 = 3
  num_hidden2 = 3
  candidate = NeuralNet(num_hidden1, num_hidden2, learning_rate)

  (X2,Y2) = parse(test_set_str)
  X2 = np.array(X2,dtype=np.float128)
  Y2 = np.array(Y2,dtype=np.float128)

  iterations = 15
  candidate.train(X,Y,iterations)
  cand_error = candidate.test(X2,Y2)
  print "PID Error: ", cand_error
