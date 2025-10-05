"""
Parts of this code are taken from
 code snippet from https://github.com/deeplearning-wisc/vos/blob/a449b03c7d6e120087007f506d949569c845b2ec/classification/CIFAR/train_virtual.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..loss.crossentropy import cross_entropy
from ..utils import apply_reduction

class NPOSRegLoss(nn.Module):
    """
    Implements the uncertainty regularization loss from 
    *NPOS: Non-Parametric Outlier Synthesis* (ICLR 2023).

    Adds a regularization term to the classification loss that aims to
    distinguish between ID embeddings and synthesized outlier embeddings
    via a binary classifier :math:`\\phi`.

    The regularization term is defined as:

    .. math::
        L_{\\text{uncertainty}} =
        \\mathbb{E}_{v \\sim V} [-\\log(1 - \\sigma(\\phi(v)))] +
        \\mathbb{E}_{x \\sim D} [-\\log(\\sigma(\\phi(h(x))))]

    where :math:`\\phi` is a possibly non-linear function and
    :math:`V` are synthesized outliers (via non-parametric sampling), 
    and :math:`D` is the in-distribution training data.

    :see Paper:
        `ArXiv <https://arxiv.org/pdf/2210.14070.pdf>`__

    Example::

        phi = torch.nn.Linear(hidden_size, 1)
        reg_loss = NPOSRegLoss(phi)

    """

    def __init__(
        self,
        phi: nn.Module,
        alpha: float = 0.1,
        sigma: float = 0.5,
        k: int = 50,
        p: int = 10,
        device: str = "cpu",
        reduction: str = "mean",
    ):
        """
        :param phi: Binary classifier applied to embeddings.
        :param alpha: Weight of the regularization loss.
        :param sigma: Std-dev for Gaussian sampling around boundary embeddings.
        :param k: k-NN value for identifying boundary points.
        :param p: Number of boundary samples used to generate outliers.
        :param device: Device (e.g., 'cuda', 'cpu')
        :param reduction: Loss reduction method.
        """
        super().__init__()
        self.phi = phi
        self.alpha = alpha
        self.sigma = sigma
        self.k = k
        self.p = p
        self.device = device
        self.reduction = reduction
        self.nll = cross_entropy

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        :param embeddings: Tensor of shape [B, D] (typically last-layer representations).
        :param labels: Tensor of shape [B] with ground truth labels.
        """
        reg_loss = self._regularization(embeddings)
        return apply_reduction(self.alpha * reg_loss, self.reduction)

    def _regularization(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the regularization term based on synthesized outliers and ID embeddings.
        :param h: [B, D] L2-normalized ID embeddings
        """
        with torch.no_grad():
            Z = F.normalize(h.detach(), dim=-1)
            dists = torch.cdist(Z, Z, p=2)
            dists.fill_diagonal_(float('inf'))  # mask self-distance

            # Step 1: k-NN distance as proxy for density
            knn_dists = dists.topk(self.k, largest=False).values[:, -1]
            boundary_indices = torch.topk(knn_dists, k=min(self.p, len(knn_dists)), largest=True).indices
            boundary_embeddings = Z[boundary_indices]

            # Step 2: sample outliers from Gaussian around boundary
            v = boundary_embeddings.unsqueeze(1) + self.sigma * torch.randn(
                (len(boundary_embeddings), 1, Z.size(-1)), device=self.device
            )
            v = v.view(-1, Z.size(-1))

        # Step 3: binary labels
        id_logits = self.phi(Z)              # label = 1
        ood_logits = self.phi(v)             # label = 0

        loss_id = F.binary_cross_entropy_with_logits(id_logits.squeeze(), torch.ones_like(id_logits.squeeze()))
        loss_ood = F.binary_cross_entropy_with_logits(ood_logits.squeeze(), torch.zeros_like(ood_logits.squeeze()))

        return loss_id + loss_ood