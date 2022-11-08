from .optimizer import Optimizer


class _LRScheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def step(self):
        raise NotImplementedError
