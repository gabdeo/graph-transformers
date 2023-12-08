from matplotlib import pyplot as plt
import torch
from graph_transformers.utils.traintools import TrainResult


def learning_curve(result: TrainResult, *, title: str = "Learning Curve"):
    r"""
    Plot the training loss and training accuracy versus epochs taken.
    """
    fig, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.set_title(title, fontsize=16)
    ax_loss.set_xlabel("Epoch", fontsize=12)

    l_trloss = ax_loss.plot(
        torch.arange(len(result.train_losses))
        / len(result.train_losses)
        * result.num_epochs,
        result.train_losses,
        label="Train loss",
        color="C0",
    )
    ax_loss.set_ylim(0, max(result.train_losses))
    ax_loss.set_ylabel("Train loss", color="C0", fontsize=12)
    ax_loss.tick_params(axis="y", labelcolor="C0")

    ax_acc = ax_loss.twinx()
    if len(result.train_accs):
        l_trainacc = ax_acc.plot(result.train_accs, label="Val", color="C1")
    else:
        l_trainacc = ()
    ax_acc.set_ylim(0, max(max(result.train_accs), 1))
    ax_acc.set_ylabel("Validation", color="C1", fontsize=12)
    ax_acc.tick_params(axis="y", labelcolor="C1")

    lines = l_trloss + l_trainacc

    ax_loss.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=13)
