import time

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl


def plot_train_val_loss(train_loss, val_loss, path: str, model_name: str):
    named_tuple = time.localtime()
    time_string = time.strftime("%d.%m.%Y,%H:%M:%S", named_tuple)

    # Save the data for further plotting
    # It is a bit ugly at the moment, but because of PL lightning
    # the first val_loss value is actually the train_loss value

    df = pd.DataFrame({"train_loss": train_loss})
    df.to_csv(f"{path}/{model_name}_train_loss_{time_string}.csv", index_label="epoch")
    df = pd.DataFrame({"val_loss": val_loss})
    df.to_csv(f"{path}/{model_name}_validation_loss_{time_string}.csv", index_label="epoch")

    plt.figure(figsize=(8, 8))
    plt.title('Training and Validation loss')
    plt.semilogy(train_loss, label='Train Loss')
    plt.semilogy(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE loss value')
    plt.legend()
    plt.savefig(f'{path}/{model_name}_{time_string}.png')


class MetricsCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.train_loss = []
        self.val_loss = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_loss.append(trainer.logged_metrics['validation_loss'].item())

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss.append(trainer.logged_metrics['train_loss'].item())
