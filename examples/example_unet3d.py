#############################################
# Example Script for Training a 3D UNet Model
#
# Prerequisites:
# - Ensure you have the necessary libraries installed: torch, einops, magnet_pinn.
# - The dataset should be preprocessed and stored in the specified BASE_DIR.
# - (optionally run the normalization script first to create normalization files)
#
#############################################

import torch
import einops
from torch.utils.data import DataLoader
from magnet_pinn.losses import MSELoss
from magnet_pinn.utils import StandardNormalizer
from magnet_pinn.data.utils import worker_init_fn
from magnet_pinn.models import UNet3D

from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.transforms import Compose, Crop, GridPhaseShift

# Set the base directory where the preprocessed data is stored
BASE_DIR = "data/processed/train/grid_voxel_size_4_data_type_float32"
target_normalizer = StandardNormalizer.load_from_json(f"{BASE_DIR}/normalization/target_normalization.json")
input_normalizer = StandardNormalizer.load_from_json(f"{BASE_DIR}/normalization/input_normalization.json")

# Create a DataLoader for the preprocessed data
augmentation = Compose(
    [
        Crop(crop_size=(100, 100, 100)),
        GridPhaseShift(num_coils=8)
    ]
)

iterator = MagnetGridIterator(
    BASE_DIR,
    transforms=augmentation,
    num_samples=100
)
train_loader = DataLoader(iterator, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)

# Create the model
model = UNet3D(5, 12, f_maps=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = MSELoss()
subject_lambda = 10.0
space_lambda = 0.01

for epoch in range(10):
    model.train()
    for i, batch in enumerate(train_loader):
        properties, phase, field, subject_mask = batch['input'], batch['coils'], batch['field'], batch['subject']
        x = torch.cat([properties, phase], dim=1)
        y = einops.rearrange(field, 'b he reim xyz ... -> b (he reim xyz) ...')
        x = input_normalizer(x)
        y = target_normalizer(y)
        optimizer.zero_grad()
        y_hat = model(x)
        # calculate loss
        subject_loss = criterion(y_hat, y, subject_mask)
        space_loss = criterion(y_hat, y, ~subject_mask)
        loss = subject_loss*subject_lambda + space_loss*space_lambda

        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")