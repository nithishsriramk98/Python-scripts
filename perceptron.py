#binary classifier
import numpy as np      #importing numpy library
import pandas as pd      #importing pandas library
import seaborn as sns     # importing seaborn library

class Neuron:           # define a class named Neuron
  def __init__(self, n_inputs, bias = 0., weights = None):
    self.b = bias       # store the bias value inside the object
    if weights: self.ws = np.array(weights)
    else: self.ws = np.what dorandom.rand(n_inputs)     # details about weights

  def __call__(self, xs):      # to call Neuron like a function
    return self._f(xs @ self.ws + self.b)

  def _f(self, x):    # activation function (Leaky ReLU)
    return max(x*.2, x)  # if x < 0 -- scale it down (0.2x), else keep x

test_data = np.random.rand(1000, 4)
perceptron = Neuron(n_inputs = 4, bias = 0.05, weights = [0.5, 1.2, -0.3, 0.8])    # create a neuron
test_predictions = [perceptron(text) for test in test_data]      # use model to predict random test data

#plot using seaborn

sns.scatterplot(x="x", y="y", hue="class", data=dp.DataFrame({'x':[x for x,_ in test_data], 'y':[y for _,y in test_data], 'class':['cats' if p>=0.0 else 'dogs' for p in test_predictions]}))

