import numpy as np

class Perceptron:

	def __init__(self):
		self.bias = -1
		self.weights = np.array(np.random.random_sample((3,)))
		
	def aggregation_function(self, values):
		return (values * self.weights).sum()

	def activation_function(self, u):
		if u >= 1:
			return 1
		else:
			return 0

	def fit(self, X, y):
		pass
		

perceptron = Perceptron()
u = perceptron.aggregation_function(np.array([3, 1, -1]))
print(perceptron.activation_function(u))