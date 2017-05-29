import numpy as np

class Perceptron:

	def __init__(self):
		self.bias = -1
		self.weights = np.array(np.random.random_sample((3,)))

	def fit(self, X, y):
		pass
		

perceptron = Perceptron()