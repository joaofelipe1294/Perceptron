import numpy as np
import matplotlib as plt


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

	def back_propagation(self, x_values, label_finded, label_correct):
		x_values = np.append(x_values, -1)
		error = label_correct - label_finded
		self.weights = self.weights + error * x_values

	def fit(self, X, y):
		while True:
			error_count = 0
			for line_index in range(0, X.shape[0]):
				x_values = X[line_index, :]
				output = self.find_signal(x_values)
				if output != y[line_index]:
					self.back_propagation(x_values, output, y[line_index])
					error_count += 1

			print(self.weights)
			if error_count == 0:
				break

	def find_signal(self, x_values):
		x_with_bias = np.append(x_values, self.bias)
		u = self.aggregation_function(x_with_bias)
		output = self.activation_function(u)
		return output

	def predict(self, x_values):
		output = self.find_signal(x_values)
		return output


X = np.array([[1,1], [3, 1], [-1, 2], [2, -1], [-1, -1], [1, -3]])
y = [1,0,1,0,1,0]

perceptron = Perceptron()
perceptron.fit(X, y)
label = perceptron.predict(np.array([-1, 4]))
print(label)

#perceptron.insert_bias(np.array(([3, 1], [2, -1], [1, 1], [-1, -1])))
#u = perceptron.aggregation_function(np.array([3, 1, -1]))
#print(perceptron.activation_function(u))