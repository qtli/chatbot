import matplotlib.pyplot as plt

from data import VOCAB_SIZE
from models import Seq2Seq


def plot_loss(train_losses, val_losses):
    plt.plot(train_losses, color='red', label='train')
    plt.plot(val_losses, color='blue', label='val')
    plt.legend(loc='upper right', frameon=False)
    plt.show()


model = Seq2Seq(VOCAB_SIZE, 256)
train_losses, val_losses = model.train(iters=1000)
plot_loss(train_losses, val_losses)
