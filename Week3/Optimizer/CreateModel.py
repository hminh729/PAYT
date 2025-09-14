import numpy as np
from Layer import FCLayer
from Optimizer import Adam

class InitModel():
    def __init__(self, input_size, output_size, hidden_size):
        self.layer1 = FCLayer(input_size=input_size, output_size = hidden_size[0], activation='relu')
        self.layer2 = FCLayer(input_size=hidden_size[0], output_size=hidden_size[1], activation='relu')
        self.layer3 = FCLayer(input_size=hidden_size[1], output_size=output_size, activation='softmax')

        self.adam_layer1 = Adam(beta1=0.9, beta2=0.99)
        self.adam_layer2 = Adam(beta1=0.9, beta2=0.99)
        self.adam_layer3 = Adam(beta1=0.9, beta2=0.99)

    def forward(self, inputs):
        output1 = self.layer1.forward(inputs)
        output2 = self.layer2.forward(output1)
        output3 = self.layer3.forward(output2)
        return output3

    def backward(self, inputs, target, learning_rate):
        epsilon = 1e-8
        forward_value = self.forward(inputs = inputs)
        loss = -np.mean(np.sum(target * np.log(forward_value + epsilon), axis=1))
        # l2_lambda = 0.01
        # l2_penalty = (l2_lambda / 2) * (np.sum(self.layer1.weights) + np.sum(self.layer2.weights) + np.sum(self.layer3.weights))
        # loss = loss + l2_penalty

        predict = np.argmax(forward_value,axis=1)
        true_lable = np.argmax(target, axis = 1)
        accuracy = np.mean(predict == true_lable)

        d_values = (forward_value - target) / forward_value.shape[0]
        grad_w3, grad_b3, d_inputs3 = self.layer3.backward(d_values)
        grad_w2, grad_b2, d_inputs2 = self.layer2.backward(d_inputs3)
        grad_w1, grad_b1, d_inputs1 = self.layer1.backward(d_inputs2)


        self.layer1.weights, self.layer1.biases = self.adam_layer1.update(self.layer1.weights, self.layer1.biases, learning_rate, grad_w1, grad_b1, self.layer1.Eg_w, self.layer1.Eg_b)
        self.layer2.weights, self.layer2.biases = self.adam_layer2.update(self.layer2.weights, self.layer2.biases, learning_rate, grad_w2, grad_b2 ,self.layer2.Eg_w, self.layer2.Eg_b)
        self.layer3.weights, self.layer3.biases = self.adam_layer3.update(self.layer3.weights, self.layer3.biases, learning_rate, grad_w3, grad_b3,  self.layer3.Eg_w, self.layer3.Eg_b)

        return loss, accuracy
