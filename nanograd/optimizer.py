class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError

    def zero_grad(self, set_to_none: bool = False) -> None:
        for p in self.params:
            if set_to_none:
                p.grad = None
            else:
                p.grad = 0


class GradientDescent(Optimizer):
    def __init__(self, params, lr: float = 0.001):
        super(GradientDescent, self).__init__(params=params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data += -self.lr * p.grad
