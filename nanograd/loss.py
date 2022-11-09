from typing import List, Union

from .nn import Module
from .scalar import Scalar


class _Loss(Module):
    def __init__(self):
        super(_Loss, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class MSELoss(_Loss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(
            self,
            pred: Union[List[float], Scalar, List[Scalar]],
            real: Union[List[float], Scalar, List[Scalar]]
    ):
        if isinstance(pred, Scalar):
            return (pred - real) ** 2
        else:
            return sum((((y_pred - y_real) ** 2) for y_real, y_pred in zip(real, pred)))
