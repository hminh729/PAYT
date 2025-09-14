import numpy as np


class Adam():
    def __init__(self,beta1, beta2):

        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.m_weights = None
        self.v_weights = None
        self.m_biases = None
        self.v_biases = None
        self.epsilon = 1e-8



    def update(self, weights, biases,learning_rate, grad_w, grad_b, Eg_w, Eg_b):
        if self.m_weights is None:
            self.m_weights = np.zeros_like(weights)
            self.v_weights = np.zeros_like(weights)
            self.m_biases = np.zeros_like(biases)
            self.v_biases = np.zeros_like(biases)

        self.t += 1

        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * grad_w
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (grad_w **2)
        m_hat_weights = self.m_weights / (1 - self.beta1 ** self.t)
        v_hat_weights = self.v_weights / (1 - self.beta2 ** self.t)
        weights -= (learning_rate * (m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)) + learning_rate * 0.01 * weights)
        # weights -= (learning_rate * (m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)))
        # weights -= (learning_rate * grad_w / (np.sqrt(Eg_w) + self.epsilon))

        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * grad_b
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (grad_b ** 2)
        m_hat_biases = self.m_biases / (1 - self.beta1 ** self.t)
        v_hat_biases = self.v_biases / (1 - self.beta2 ** self.t)
        biases -= (learning_rate * (m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)) + learning_rate * 0.01 * biases)
        # biases -= (learning_rate * (m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)))
        # biases -= (learning_rate * grad_b / (np.sqrt(Eg_b) + self.epsilon))

        return weights, biases