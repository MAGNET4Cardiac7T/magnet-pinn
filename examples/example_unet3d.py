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
from pathlib import Path
from torch.utils.data import DataLoader
from magnet_pinn.losses import MSELoss
from magnet_pinn.utils import StandardNormalizer
from magnet_pinn.data.utils import worker_init_fn
from magnet_pinn.models import UNet3D

from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.transforms import Compose, Crop, GridPhaseShift
import tqdm

# Configuration
BASE_DIR = Path("data/processed/train/grid_voxel_size_4_data_type_float32")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load normalizers
target_normalizer = StandardNormalizer.load_from_json(f"{BASE_DIR}/normalization/target_standardnormalizer_log_after.json")
input_normalizer = StandardNormalizer.load_from_json(f"{BASE_DIR}/normalization/input_standardnormalizer_identity_after.json")

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
    num_samples=10
)
train_loader = DataLoader(iterator, batch_size=4, num_workers=2, worker_init_fn=worker_init_fn)

# Create the model
model = UNet3D(5, 12, f_maps=32)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = MSELoss()
subject_lambda = 10.0
space_lambda = 0.01

# Training configuration
num_epochs = 1
best_loss = float('inf')

print("\nStarting training...")
print(f"{len(train_loader)}")
print("=" * 60)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for i, batch in enumerate(train_loader):
        properties, phase, field, subject_mask = batch['input'], batch['coils'], batch['field'], batch['subject']

        # Move data to device
        properties = properties.to(device)
        phase = phase.to(device)
        field = field.to(device)
        subject_mask = subject_mask.to(device)

        # Prepare inputs and targets
        x = torch.cat([properties, phase], dim=1)
        y = einops.rearrange(field, 'b he reim xyz ... -> b (he reim xyz) ...')
        x = input_normalizer(x)
        y = target_normalizer(y)

        # Forward pass
        optimizer.zero_grad()
        y_hat = model(x)

        # Calculate loss
        subject_loss = criterion(y_hat, y, subject_mask)
        space_loss = criterion(y_hat, y, ~subject_mask)
        loss = subject_loss*subject_lambda + space_loss*space_lambda

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        epoch_loss += loss.item()
        num_batches += 1

        if i % 10 == 0:
            print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {i}, Loss: {loss.item():.6f}")

    # Calculate average epoch loss
    avg_epoch_loss = epoch_loss / num_batches
    print(f"\nEpoch {epoch+1}/{num_epochs} - Average Loss: {avg_epoch_loss:.6f}")

    # Save best model
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        best_model_path = MODEL_DIR / "unet3d_best.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"✓ Saved best model to {best_model_path} (loss: {best_loss:.6f})")

    # Save latest checkpoint
    latest_checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss,
    }
    checkpoint_path = MODEL_DIR / "unet3d_latest.pth"
    torch.save(latest_checkpoint, checkpoint_path)
    print(f"✓ Saved checkpoint to {checkpoint_path}")
    print("-" * 60)

print("\n" + "=" * 60)
print("Training complete!")
print(f"Best loss: {best_loss:.6f}")
print(f"Best model saved to: {MODEL_DIR / 'unet3d_best.pth'}")
print(f"Latest checkpoint saved to: {MODEL_DIR / 'unet3d_latest.pth'}")
print("\nYou can now use this model for predictions with:")
print("  python examples/example_unet3d_predictions.py")
print("=" * 60)