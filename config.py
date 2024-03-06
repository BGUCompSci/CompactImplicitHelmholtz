from dataclasses import dataclass
from typing import Union

import pytorch_lightning as pl
from pyrallis import field

from unet.implicit_unet import EncoderSolver
from unet.original_unet import FeaturesUNet


@dataclass
class TrainConfig:
    dataset_path: Union[list, str] = field()
    results_path: str = field()
    exp_name: str = field(default='default_exp')
    batch_size: int = field(default=32)
    val_percent: float = field(default=0.2)
    seed: int = field(default=42)
    devices: int = field(default=1)
    epochs: int = field(default=100)
    workers: int = field(default=4)

    # Architecture to select. Options are:
    #   ExplicitEncoderSolver, ImplicitEncoderSolver, OriginalEncoderSolver
    architecture: str = field(default='ImplicitEncoderSolver')
    input_channels: int = field(default=3)
    output_channels: int = field(default=2)
    small: bool = field(default=False)

    @property
    def model(self) -> pl.LightningModule:
        factory = {'ExplicitEncoderSolver': EncoderSolver(self.input_channels, self.output_channels,
                                                          implicit=False, small=self.small),
                   'ImplicitEncoderSolver': EncoderSolver(self.input_channels, self.output_channels,
                                                          implicit=False, small=self.small),
                   'OriginalEncoderSolver': FeaturesUNet(self.input_channels, self.output_channels,
                                                         kernel=3)}
        return factory[self.architecture]
