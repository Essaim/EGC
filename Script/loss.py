import sys

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F


def get_loss(name: str, **kwargs):
    try:
        return getattr(nn, name)(**kwargs)
    except AttributeError:
        try:
            return getattr(sys.modules[__name__], name)(**kwargs)
        except AttributeError:
            raise ValueError(f'{name} is not defined.')

class HuberMSELoss(nn.Module):
    def __init__(self, rho=1):
        super(HuberMSELoss, self).__init__()
        self.rho = rho
    def forward(self, predict, truth):
        x = torch.abs(truth - predict)
        loss = torch.where(x>self.rho, x - 0.5*self.rho, (0.5 / self.rho)*torch.pow(x,2))
        return loss.mean()


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, truth, predict):
        return self.mse_loss(truth, predict) ** 0.5

class MaskedLoss(nn.Module):
    def __init__(self, null_val=0.0):
        super(MaskedLoss, self).__init__()
        self.null_val = null_val

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError()

    def compute_mask(self, targets: Tensor):
        if np.isnan(self.null_val):
            mask = ~torch.isnan(targets)
        else:
            mask = torch.ne(targets, self.null_val)
        mask = mask.to(torch.float32)
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.tensor(0., device=mask.device), mask)
        return mask

class MaskedMSELoss(MaskedLoss):
    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        mask = self.compute_mask(targets)
        loss = F.mse_loss(outputs, targets, reduction='none') * mask
        loss = torch.where(torch.isnan(loss), torch.tensor(0., device=loss.device), loss)
        return loss.mean()

class MaskedRMSELoss(MaskedLoss):
    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        mask = self.compute_mask(targets)
        loss = F.mse_loss(outputs, targets, reduction='none') * mask
        loss = torch.where(torch.isnan(loss), torch.tensor(0., device=loss.device), loss)
        return loss.mean()

class MaskedMAELoss(MaskedLoss):
    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        mask = self.compute_mask(targets)
        loss = F.l1_loss(outputs, targets, reduction='none') * mask
        loss = torch.where(torch.isnan(loss), torch.tensor(0., device=loss.device), loss)
        return loss.mean()