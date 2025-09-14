import numpy as np

class FCLayer():
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.Eg_w = np.zeros(( input_size ,output_size))
        self.Eg_b = np.zeros(( 1, output_size))

    def forward(self, inputs):
        self.x = inputs
        z = np.dot(self.x, self.weights) + self.biases

        if self.activation == "relu":
            self.output = np.maximum(0, z)
        elif self.activation == "softmax":
            exp_vals = np.exp(z - np.max(z, axis=-1, keepdims=True))
            self.output = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

        return self.output

    def backward(self, d_values):
        if self.activation == "relu":
            d_values = d_values * (self.output > 0)

        elif self.activation == "softmax":
            for i, single_output in enumerate(self.output):
                single_output = single_output.reshape(-1, 1)
                jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
                d_values[i] = np.dot(jacobian_matrix, d_values[i])

        d_weights = np.dot(self.x.T, d_values)
        d_biases = np.sum(d_values, axis=0, keepdims=True)
        d_inputs = np.dot(d_values, self.weights.T)
        self.Eg_w = self.Eg_w * 0.9 + 0.1 * (d_weights ** 2)
        self.Eg_b = self.Eg_b * 0.9 + 0.1 * (d_biases ** 2)

        return d_weights, d_biases, d_inputs
