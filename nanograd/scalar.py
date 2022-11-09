import math
from typing import Tuple

import rich.repr


class Scalar:
    _compact_print: bool = True

    def __init__(self, data, label: str = None, _children: Tuple = (), _op: str = None):
        """Gradient should be removed from leaf node.
        """
        self.data = data
        self.grad = 0.0
        self.label = label
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    def __repr__(self):
        """If `_compact_print` is enabled it print detailed information of whole network including child in order.
        This may take long for large network so disabled by default.
        """
        if self._compact_print:
            return f'{self.__class__.__name__}(data={self.data})'
        return f'{self.__class__.__name__}(data={self.data}, children={self._prev}, op={self._op})'

    def __rich_repr__(self) -> rich.repr.Result:
        """Print with rich library. Use `from rich import print as rprint`. May take long for large network.
        Meant to be used for smaller to medium size function for understanding operations without diagram.

        Examples:
            >>> from rich import print as rprint
            >>> rprint(Scalar(2))
            Scalar(data=2, grad=0.0, label=None, children=(), op=None)
        """
        yield "data", self.data
        yield "grad", self.grad
        yield "label", self.label
        yield "children", self._prev
        yield "op", self._op

    def __add__(self, other):
        """Forward pass of adding two Scalar with backward pass implemented as closure. Internal `_backward`
        function is not meant to be called by user other than debugging.
        """
        other = other if isinstance(other, Scalar) else Scalar(data=other)
        out = Scalar(data=self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            """For add operation out node gradient flows backward to self and other node with same value. `.data` is
             ommited in below example to simplify expression.

            Local derivate of `out` with respect to first position in add expression which is `self` is,
            d(out) / d(self) = d(self + other) / d(self) = 1.0 + 0.0 = 1.0 based on calculus derivative rule.
            Similarly, d(out) / d(other) = d(self + other) / d(other) = 0.0 + 1.0 = 1.0.

            Here, `out` refers to current add operation output. For getting derivative of self and grad with respect
            to whole network output it the local gradient needs to be multiplied with the calculated output gradient
            of the out node as per chain rule.

            If `L` is the network output where, d(L) / d(out) is the calulated gradient of L with respect to out then,
            for calculating gradient of self with respect to `L` is,
            d(L) / d(self) = (d(L) / d(out)) * (d(out) / d(self)). Here, d(L) / d(out) is already calulated via back
            propagation as `out.grad`. There may be more nodes between d(L) / d(out) but `out.grad` has calculated
            changes through all those nodes to `L`.
            Similarly, for `other` same process is followed.

            Gradients are summed with previous one for cases when multiple nodes calculate and backpropagate gradients
            they can overwrite each other instead there contributions are added.
            """
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(data=other)
        out = Scalar(data=self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            """Local derivative of `out` with respect self and other are,
            d(out) / d(self) = d(self * other) / d(self) = other.
            d(out) / d(other) = d(self * other) / d(other) = self.
            `.data` part is ommited in above example to simplify expression.

            As explained above to get derivative of network output with respect `self` and `other` local gradient is
            mutliplied with output gradient based on chain rule to get the grad values.
            """
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __pow__(self, power, modulo=None):
        assert isinstance(power, (int, float)), 'Only int or float supported for power.'
        out = Scalar(data=self.data ** power, _children=(self,), _op=f'**{power}')

        def _backward():
            """Calculus derivative formula, 3x^4 = 3*4*x^(4-1) = 12x^3.
            """
            self.grad += ((self.data * power) ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """Formula for `tanh` forward and backward pass used based on mathematical formula.
        """
        out = math.exp(2 * self.data)
        out = (out - 1) / (out + 1)
        out = Scalar(data=out, _children=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - (out.data ** 2)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Scalar(data=math.exp(self.data), _children=(self,), _op='tanh')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topological_order = []
        visited = set()

        def build_topological_order(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topological_order(child)
                topological_order.append(v)

        build_topological_order(self)

        self.grad = 1.0
        for v in reversed(topological_order):
            v._backward()

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return self * (other ** -1)

    def __rsub__(self, other):
        return self + (-other)
