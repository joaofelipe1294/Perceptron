import numpy as np

class Perceptron:

	def __init__(self):
		self.bias = -1
		self.weights = np.array(np.random.random_sample((3,)))
		
	def aggregation_function(self, values):
		return (values * self.weights).sum()

	def fit(self, X, y):
		pass
		

perceptron = Perceptron()
f = perceptron.aggregation_function(np.array([3, 1, -1]))
print(f)