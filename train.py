import pyrallis
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split

from config import TrainConfig
from dataloader import RandomKappaDataset
from utils import MetricsCallback, plot_train_val_loss


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str | list[str], val_percent: float, seed: int, batch_size: int, workers: int,
                 switch_datasets: bool = True) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.val_percent = val_percent
        self.seed = seed
        self.workers = workers
        self.dataset_path = dataset_path if type(dataset_path) == list else [dataset_path]

        self.switch_datasets = switch_datasets
        self.curr_dataloader_idx = 0

    def setup(self, stage: str = "") -> None:
        datasets = [RandomKappaDataset(dataset)
                    for dataset in self.dataset_path]
        self.train_loaders = []
        self.val_loaders = []
        for dataset in datasets:
            n_val = int(len(dataset) * self.val_percent)
            n_train = len(dataset) - n_val
            train_set, val_set = random_split(
                dataset, [n_train, n_val], generator=torch.Generator().manual_seed(self.seed))

            self.train_loaders.append(train_set)
            self.val_loaders.append(val_set)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.switch_datasets and self.trainer.current_epoch % 20 == 0:
            self.curr_dataloader_idx = (self.curr_dataloader_idx + 1) % len(self.train_loaders)
            print("Switched train_loader")
        return DataLoader(self.train_loaders[self.curr_dataloader_idx], batch_size=self.batch_size,
                          num_workers=self.workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.switch_datasets and self.trainer.current_epoch % 20 == 0:
            self.curr_dataloader_idx = (self.curr_dataloader_idx + 1) % len(self.val_loaders)
            print("Switched val_loader")
        return DataLoader(self.val_loaders[self.curr_dataloader_idx], batch_size=self.batch_size,
                          num_workers=self.workers)


def train_network(exp_name: str, model: pl.LightningModule, dataset_path: str | list[str], graph_path: str,
                  val_percent: float, batch_size: int, epochs: int, workers: int, seed: int,
                  evaluation_callback: pl.Callback = None):
    dm = DataModule(dataset_path, val_percent, seed, batch_size, workers)
    metrics = MetricsCallback()
    mntr_ckpt = ModelCheckpoint(monitor="train_loss")
    lr_monitor = LearningRateMonitor("epoch")
    callbacks = [metrics, mntr_ckpt, lr_monitor]
    if evaluation_callback is not None:
        callbacks.append(evaluation_callback)

    trainer = pl.Trainer(max_epochs=epochs, callbacks=callbacks, accelerator='cpu',
                         reload_dataloaders_every_n_epochs=1)
    trainer.fit(model, datamodule=dm)

    train_loss, val_loss = metrics.train_loss, metrics.val_loss
    plot_train_val_loss(train_loss, val_loss, graph_path, exp_name)
    return model


@pyrallis.wrap()
def main(cfg: TrainConfig) -> None:
    pl.seed_everything(cfg.seed)
    print(cfg)

    train_network(cfg.exp_name, cfg.architecture.model, cfg.dataset_path,
                  cfg.graph_path, cfg.val_percent, cfg.batch_size, cfg.epochs,
                  cfg.workers, cfg.seed)


if __name__ == "__main__":
    main()
