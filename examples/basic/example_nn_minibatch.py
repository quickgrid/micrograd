from nanograd.nn import MLP
from nanograd.visualization import save_plot


# Model definition.
n = MLP(input_count=3, neuron_count=[2, 3, 2, 1])


# Multiple sample input as a minibatch and loss calculation against target.
xs = [
    [2.3, 4.5, -1],
    [3, -4.5, 2],
    [-2.3, 2.1, 3.5],
    [0.3, 0.5, 2],
]

ys = [-1.0, 1.0, 1.0, -1.0]

# Run forward pass and get results.
y_preds = [n(i) for i in xs]
print(y_preds)

# MSE Loss.
loss = sum((((y_pred - y_real) ** 2) for y_real, y_pred in zip(ys, y_preds)))
loss.label = 'loss'
print(loss)

# Backward pass.
n.zero_grad()
loss.backward()


print(n.layers[1].neurons[0].w[0].grad)

save_plot(directory='../../output', filename='example_nn_minibatch', expression=loss)
