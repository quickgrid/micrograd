import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, symbols


def f1(x):
    return 4 * x ** 3 - 4 * x + 5


def symbolic_derivative(func, args):
    print(func(*args))
    return diff(func(*args), *args)


def main():
    h = 0.0001
    x = 3
    print((f1(x + h) - f1(x)) / h)

    x = symbols('x')
    x_prime = symbolic_derivative(func=f1, args=(x, ))
    print(x_prime)

    xs = np.arange(-5, 5, 0.25)
    ys = f1(xs)

    print(xs)
    print(ys)

    plt.plot(xs, ys)
    plt.show()


if __name__ == '__main__':
    main()
