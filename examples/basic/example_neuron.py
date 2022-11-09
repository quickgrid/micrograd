from nanograd.scalar import Scalar
from nanograd.visualization import save_plot


# Forward pass.
x1 = Scalar(2.0, label='x1')
x2 = Scalar(-4.0, label='x2')

w1 = Scalar(-3.0, label='w1')
w2 = Scalar(1.0, label='w2')

b = Scalar(6.875, label='b')

x1w1 = x1 * w1
x1w1.label = 'x1w1'
x2w2 = x2 * w2
x2w2.label = 'x2w2'

sum_wx = x1w1 + x2w2
sum_wx.label = 'sum_wx'

n = sum_wx + b
n.label = 'n'

out = n.tanh()
out.label = 'out'


# Topological ordering. It can be verfied looking at plotted model.
topological_order = []
visited = set()


def build_topological_order(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topological_order(child)
        topological_order.append(v)


build_topological_order(out)
print(topological_order)


# Manual backward pass.
out.grad = 1.0
out._backward()
n._backward()
sum_wx._backward()
x1w1._backward()
x2w2._backward()


save_plot(expression=out, filename='example_neuron', directory='../../output')
