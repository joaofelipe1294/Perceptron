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

	def insert_bias(self, values):
		bias_array = np.ones(values.shape[0]) * self.bias
		bias_array = np.reshape(bias_array, (bias_array.shape[0], 1))
		prepared_values = np.concatenate((values, bias_array), axis = 1)
		return prepared_values

	def fit(self, X, y):
		pass




perceptron = Perceptron()
perceptron.insert_bias(np.array(([3, 1], [2, -1], [1, 1], [-1, -1])))
#u = perceptron.aggregation_function(np.array([3, 1, -1]))
#print(perceptron.activation_function(u))