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

	def back_propagation(self, x_values, label_finded, label_correct):
		error = label_correct - label_finded
		self.weights = self.weights + error * x_values

	def fit(self, X, y):
		X_with_bias = self.insert_bias(X)
		while True:
			error_count = 0
			for line_index in range(0, X_with_bias.shape[0]):
				x_values = X_with_bias[line_index, :]
				u = self.aggregation_function(x_values)
				output = self.activation_function(u)
				if output != y[line_index]:
					self.back_propagation(x_values, output, y[line_index])
					error_count += 1

			print(self.weights)
			if input('Continue : ') != "s" or error_count == 0:
				break

	


X = np.array([[1,1], [3, 1], [-1, 2], [2, -1], [-1, -1], [1, -3]])
y = [1,0,1,0,1,0]

perceptron = Perceptron()
perceptron.fit(X, y)

#perceptron.insert_bias(np.array(([3, 1], [2, -1], [1, 1], [-1, -1])))
#u = perceptron.aggregation_function(np.array([3, 1, -1]))
#print(perceptron.activation_function(u))