from magnet_pinn.models import UNet3D
from magnet_pinn.data.dataset import PhaseAugmentedMagnetIterator
from magnet_pinn.utils import StandardNormalizer

import pytorch_lightning as pl

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MAGNETPINN(pl.LightningModule):
    def __init__(self, net: torch.nn.Module):
        super(MAGNETPINN, self).__init__()
        self.net = net
        self.input_normalizer = StandardNormalizer(key="input", axis=1, ndims=5)
        self.efield_normalizer = StandardNormalizer(key="efield", axis=1, ndims=5)
        self.hfield_normalizer = StandardNormalizer(key="hfield", axis=1, ndims=5)
        self.input_normalizer.load_from_numpy("data/processed/train/grid_voxel_size_4/input_normalization.npy")
        self.efield_normalizer.load_from_numpy("data/processed/train/grid_voxel_size_4/efield_normalization.npy")
        self.hfield_normalizer.load_from_numpy("data/processed/train/grid_voxel_size_4/hfield_normalization.npy")


    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        properties, phase, efield, hfield, subject_mask = batch['input'], batch['coils_real'], batch['efield'], batch['hfield'], batch['subject']
        properties = self.input_normalizer(properties)
        efield = self.efield_normalizer(efield)
        hfield = self.hfield_normalizer(hfield)
        x = torch.cat([properties, phase], dim=1)
        y = torch.cat([efield, hfield], dim=1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
dataset = PhaseAugmentedMagnetIterator(data_dir="data/processed/batch_1/grid_processed_voxel_size_4/")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

net = UNet3D(6, 12, is_segmentation=False, f_maps=16)
model = MAGNETPINN(net)

trainer = pl.Trainer(max_epochs=10, devices=[0], accelerator="gpu")

trainer.fit(model, dataloader)