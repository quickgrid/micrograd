import random
from typing import List

from .scalar import Scalar


class Module:
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Attenmpt to emulate `set_to_none` of pytorch. Not sure yet if it is the right way.
        """
        for p in self.parameters():
            if set_to_none:
                p.grad = None
            else:
                p.grad = 0

    @staticmethod
    def parameters():
        return []


class Neuron(Module):
    def __init__(self, input_count: int):
        super(Neuron, self).__init__()
        self.w = [Scalar(data=random.uniform(-1, 1)) for _ in range(input_count)]
        self.b = Scalar(data=random.uniform(-1, 1))

    def forward(self, x):
        """Upon input calculates single neuron activation with non-linearity.

        Returns:
             w * x + b.
        """
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return out.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, input_count: int, neuron_count: int):
        super(Layer, self).__init__()
        self.neurons = [Neuron(input_count=input_count) for _ in range(neuron_count)]

    def forward(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, input_count: int, neuron_count: List[int]):
        """Input layer is merged with combiend hidden and output layer to get total layers in network.

        Args:
            input_count: Neurons in input layer.
            neuron_count: List of neurons per layer.
        """
        super(MLP, self).__init__()
        all_layers = [input_count] + neuron_count
        self.layers = [
            Layer(input_count=all_layers[i], neuron_count=all_layers[i + 1]) for i in range(len(neuron_count))
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
