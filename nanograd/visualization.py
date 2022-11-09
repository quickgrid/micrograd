from typing import Set, Tuple

from graphviz import Digraph

from .scalar import Scalar


def trace(root: Scalar) -> [Set[Scalar], Set[Tuple[Scalar, Scalar]]]:
    nodes = set()
    edges = set()

    def build(v: Scalar):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def plot_model(root: Scalar, img_format: str = 'png', rankdir: str = 'LR') -> Digraph:
    """ Outputs an image from graph with chosen format.

    Args:
        root: Value class.
        img_format: png, svg etc.
        rankdir: TB (top to bottom graph), LR (left to right).
    """
    assert rankdir in ['LR', 'TB'], 'Please insert TB or LB.'

    nodes, edges = trace(root)
    dot = Digraph(format=img_format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        dot.node(name=str(id(n)), label=f'{n.label} | data = {n.data:.4f} | grad = {n.grad:.4f}', shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def save_plot(filename: str, directory: str, expression: Scalar) -> None:
    dot = plot_model(expression)
    dot.render(filename=filename, directory=directory)
