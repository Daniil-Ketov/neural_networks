from numpy import dot
from numpy.linalg import norm
import random


class NeuronHebb:
    weights: list[float]

    def __init__(self, _inputs):
        self.weights = [random.random() for _ in range(_inputs)]

    def out(self, _input):
        return dot(self.weights, _input)


class Network:
    __neurons: list[NeuronHebb]
    __theta: float

    def __init__(self, _inputs_num, _neurons_num, _theta):
        self.__neurons = [NeuronHebb(_inputs_num) for _ in range(_neurons_num)]
        self.__theta = _theta

    def forward(self, _input):
        return [n.out(_input) for n in self.__neurons]

    def get_weights(self):
        return [n.weights for n in self.__neurons]

    def learn(self, _inputs, _epoch):
        for _ in range(_epoch):
            for i in _inputs:
                out = self.forward(i)
                m = max(out)
                mi = out.index(m)
                for wi in range(len(self.__neurons[mi].weights)):
                    self.__neurons[mi].weights[wi] += self.__theta * out[mi] * i[wi]


with open('input.txt') as f:
    inputs = [tuple(map(float, line.split())) for line in f]

n_neurons = 2
n_inputs = 2
theta = 0.5

network = Network(n_inputs, n_neurons, theta)
network.learn(inputs, 100)

for w in network.get_weights():
    print(w)
