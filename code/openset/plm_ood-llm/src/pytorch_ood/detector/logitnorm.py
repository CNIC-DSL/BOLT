from typing import TypeVar
from torch import Tensor
from torch.nn import Module
import torch

from ..api import Detector, ModelNotSetException

Self = TypeVar("Self")


class LogitNorm(Detector):
    """
    Implements Logit Normalization for OOD Detection.

    .. math::

        - \\max_y \\alpha \\cdot \\frac{f_y(x)}{\\|f(x)\\|_2}

    where :math:`f(x)` are the logits and :math:`\\alpha` is a scaling parameter.

    Based on ideas from:
    - *ReAct: Out-of-distribution Detection With Rectified Activations* (extended)
    - *Improving Calibration via Logit Normalization*

    :param model: PyTorch classification model
    :param alpha: scaling factor after normalization (default: 10.0)
    """

    def __init__(self, model: Module, alpha: float = 10.0):
        super(LogitNorm, self).__init__()
        self.model = model
        self.alpha = alpha

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: model input tensor
        """
        if self.model is None:
            raise ModelNotSetException
        return self.score(self.model(x))

    def predict_features(self, logits: Tensor) -> Tensor:
        """
        :param logits: raw model output (logits)
        """
        return LogitNorm.score(logits, self.alpha)

    def fit(self: Self, *args, **kwargs) -> Self:
        return self

    def fit_features(self: Self, *args, **kwargs) -> Self:
        return self

    @staticmethod
    def score(logits: Tensor, alpha: float = 10.0) -> Tensor:
        """
        Compute logit normalization score.

        :param logits: model output logits
        :param alpha: scaling factor
        :return: -max(normalized logits)
        """
        # L2 normalization per sample
        norm = torch.norm(logits, p=2, dim=1, keepdim=True) + 1e-12
        norm_logits = alpha * logits / norm
        return -norm_logits.max(dim=1).values