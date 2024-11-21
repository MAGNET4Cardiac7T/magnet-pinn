from magnet_pinn.models import UNet3D
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.utils import StandardNormalizer
from magnet_pinn.data.utils import worker_init_fn
import einops

import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader


class MAGNETPINN(pl.LightningModule):
    def __init__(self, net: torch.nn.Module,
                 target_normalizer: StandardNormalizer,
                 input_normalizer: StandardNormalizer,
                 subject_lambda: float = 10.0,
                 space_lambda: float = 0.01):
        super(MAGNETPINN, self).__init__()
        self.net = net
        self.target_normalizer = target_normalizer
        self.input_normalizer = input_normalizer

        self.subject_lambda = subject_lambda
        self.space_lambda = space_lambda

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        properties, phase, field, subject_mask = batch['input'], batch['coils'], batch['field'], batch['subject']

        x = torch.cat([properties, phase], dim=1)
        y = einops.rearrange(field, 'b he reim xyz ... -> b (he reim xyz) ...')

        x = self.input_normalizer(x)
        y = self.target_normalizer(y)

        y_hat = self(x)
        
        # calculate loss
        mse = torch.mean((y_hat - y) ** 2, dim=1)
        subject_loss = torch.mean(mse * subject_mask)
        space_loss = torch.mean(mse * (~subject_mask))
        loss = subject_loss*self.subject_lambda + space_loss*self.space_lambda

        self.log('train_loss', loss, prog_bar=True)
        self.log('subject_loss', subject_loss, prog_bar=True)
        self.log('space_loss', space_loss, prog_bar=True)

        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Set the base directory where the preprocessed data is stored
BASE_DIR = "data/processed/train/grid_voxel_size_4_data_type_float32"

# Create the data loader
iterator = MagnetGridIterator(
    BASE_DIR,
    phase_samples_per_simulation=100,
)
dataloader = DataLoader(iterator, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)

# Create the model
net = UNet3D(5, 12, f_maps=64)

target_normalizer = StandardNormalizer.load_from_json()
input_normalizer = StandardNormalizer()
target_normalizer.load_params(f"{BASE_DIR}/normalization/target_normalization.json")
input_normalizer.load_params(f"{BASE_DIR}/normalization/input_normalization.json")

model = MAGNETPINN(net, target_normalizer, input_normalizer)

# Train the model
trainer = pl.Trainer(max_epochs=20, devices=[0], accelerator="gpu")
trainer.fit(model, dataloader)