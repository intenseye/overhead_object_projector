import torch
import torch.nn as nn


class Criterion_mse_loss(nn.Module):
    """
    MSE loss class
    """
    def __init__(self):
        """
        Initialize MSE loss class.
        """
        super(Criterion_mse_loss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        The forward pass function of the MSE loss.

        Parameters
        ----------
        output: torch.Tensor
            Model output
        target: torch.Tensor
            Target value

        Returns
        ----------
        loss: torch.Tensor
            The loss value
        """
        loss = self.loss(output, target)
        return loss


class Criterion_nth_power_loss(nn.Module):
    """
    n^th power loss class (e.g. for MSE n=2)
    """
    def __init__(self, power_term: int = 2):  # It is advised to use positive even integer for power_term
        """
        Initialize nth power loss class.

        Parameters
        ----------
        power_term: int
            The power term
        """

        super(Criterion_nth_power_loss, self).__init__()
        self.power_term = power_term

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        The forward pass function of the nth power loss.

        Parameters
        ----------
        output: torch.Tensor
            Model output
        target: torch.Tensor
            Target value

        Returns
        ----------
        loss: torch.Tensor
            The loss value
        """
        loss = self.loss(output, target)
        return loss

    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        The forward pass function of the nth power loss.

        Parameters
        ----------
        output: torch.Tensor
            Model output
        target: torch.Tensor
            Target value

        Returns
        ----------
        power_loss: torch.Tensor
            The loss value
        """
        dist = output - target
        power_loss = dist.pow(self.power_term).mean(1).pow(1/self.power_term)
        return power_loss

