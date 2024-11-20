from magnet_pinn.models import UNet3D
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.utils import StandardNormalizer
from magnet_pinn.data.utils import worker_init_fn
import einops

import pytorch_lightning as pl

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


class MAGNETPINN(pl.LightningModule):
    def __init__(self, net: torch.nn.Module):
        super(MAGNETPINN, self).__init__()
        self.net = net
        self.target_normalizer = StandardNormalizer()
        self.input_normalizer = StandardNormalizer()
        self.target_normalizer.load_params("data/processed/train/grid_voxel_size_4_data_type_float32/normalization/target_normalization.json")
        self.input_normalizer.load_params("data/processed/train/grid_voxel_size_4_data_type_float32/normalization/input_normalization.json")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        properties, phase, field, subject_mask = batch['input'], batch['coils'], batch['field'], batch['subject']

        x = torch.cat([properties, phase], dim=1)
        y = einops.rearrange(field, 'b he reim xyz ... -> b (he reim xyz) ...')

        x = self.input_normalizer(x)
        y = self.target_normalizer(y)

        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
iterator = MagnetGridIterator(
    "data/processed/train/grid_voxel_size_4_data_type_float32",
    phase_samples_per_simulation=100,
)
dataloader = DataLoader(iterator, batch_size=2, worker_init_fn=worker_init_fn)

net = UNet3D(5, 12, is_segmentation=False, f_maps=16)
model = MAGNETPINN(net)

trainer = pl.Trainer(max_epochs=10, devices=[0], accelerator="gpu")

trainer.fit(model, dataloader)