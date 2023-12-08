import dataclasses
from torch import nn
from typing import List


@dataclasses.dataclass
class TrainResult:
    r"""
    A collection containing everything we need to know about the training results
    """

    num_epochs: int
    lr: float

    # The trained model
    model: nn.Module

    # Training loss (saved at each iteration in `train_epoch`)
    train_losses: List[float]

    # Training accuracies, before training and after each epoch
    train_accs: List[float]


def compute_accuracy(output, targets):
    # Placeholder accuracy computation
    # Replace this with your actual accuracy computation logic
    correct = (output == targets).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy
