from rich import print as rprint

from nanograd.visualization import plot_model, save_plot
from nanograd.engine import ScalarValue


a = ScalarValue(5.0, label='a')
b = ScalarValue(2, label='b')
c = ScalarValue(3, label='c')

d = a * b
d.label = 'd'

e = d + c
e.label = 'e'

f = ScalarValue(2.0, label='k') + b
f.label = 'f'

out = e * f
out.label = 'out'

out.backward()


print(out)
print(out._prev)
print(out._op)

rprint(out)

t = ScalarValue(4.5)
print(1 + t)

# dot = plot_model(out)

save_plot(expression=out, filename='example_vis', directory='../../output')
