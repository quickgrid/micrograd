from nanograd.nn import MLP
from nanograd.visualization import save_plot


# Single example input.
x = [5.3, -4.5, -1]
y = 1.0

n = MLP(input_count=3, neuron_count=[4, 2, 3, 1])
out = n(x)


n.zero_grad()
loss = (y - out) ** 2
loss.label = 'loss'

print(loss)

loss.backward()

save_plot(directory='../../output', filename='example_nn', expression=loss)
