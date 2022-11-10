from nanograd.nn import MLP
from nanograd.optimizer import GradientDescent
from nanograd.loss import MSELoss
from nanograd.visualization import save_plot


# Model definition.
n = MLP(input_count=3, neuron_count=[2, 3, 1])


# In this example minibatch is equivalent to whole batch as it all is passed at once.
# Multiple sample input as a minibatch and loss calculation against target.
xs = [
    [2.3, 4.5, -1],
    [3, -4.5, 2],
    [-2.3, 2.1, 3.5],
    [0.3, 0.5, 2],
]

ys = [-1.0, 1.0, 1.0, -1.0]


# Training hyperparameters and setup.
epochs = 50
lr = 0.01
plot_graph_at_epoch = [10, 30]

optimizer = GradientDescent(params=n.parameters(), lr=lr)
mse_loss = MSELoss()


for epoch in range(epochs):
    # Run forward pass and get results.
    y_preds = [n(i) for i in xs]

    # Zero previous grads. Called here to not draw another plot just for zero grad.
    optimizer.zero_grad()

    # Calculate loss. Here, mse loss is used.
    loss = mse_loss(pred=y_preds, real=ys)
    loss.label = 'loss'
    if epoch in plot_graph_at_epoch:
        save_plot(directory='../../output', filename=f'example_nn_minibatch_epoch_loss_{epoch}', expression=loss)

    # Backward pass.
    loss.backward()
    if epoch in plot_graph_at_epoch:
        save_plot(directory='../../output', filename=f'example_nn_minibatch_epoch_backward_{epoch}', expression=loss)

    # Optimize, update weights based on calculated gradient.
    optimizer.step()

    if epoch in plot_graph_at_epoch:
        save_plot(directory='../../output', filename=f'example_nn_minibatch_epoch_step_{epoch}', expression=loss)

    # Print details.
    print(f'epoch = {epoch} loss = {loss.data}')
