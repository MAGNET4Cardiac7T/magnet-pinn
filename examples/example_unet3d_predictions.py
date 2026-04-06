#############################################
# Example Script for Generating Predictions with a Trained 3D UNet Model
#
# Prerequisites:
# - Trained model saved to MODEL_PATH
# - Test dataset preprocessed and stored in BASE_DIR
# - Normalization files created and stored in BASE_DIR/normalization/
#
#############################################

import torch
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np

from magnet_pinn.models import UNet3D
from magnet_pinn.utils import StandardNormalizer
from magnet_pinn.data.utils import worker_init_fn
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.transforms import Compose, Crop, CoilEnumeratorPhaseShift
from magnet_pinn.prediction import GridPredictionWriter

# Paths configuration
BASE_DIR = Path("data/processed/test/grid_voxel_size_4_data_type_float32")
OUTPUT_DIR = Path("data/predictions/test")
MODEL_PATH = Path("models/unet3d_best.pth")

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load trained model
print("Loading model...")
model = UNet3D(5, 12, f_maps=32)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
print(f"Model loaded successfully")


# Load normalizers
print("Loading normalizers...")
target_normalizer = StandardNormalizer.load_from_json(
    f"{BASE_DIR}/normalization/target_standardnormalizer_log_after.json"
)
input_normalizer = StandardNormalizer.load_from_json(
    f"{BASE_DIR}/normalization/input_standardnormalizer_identity_after.json"
)

# Create dataloader with coil enumeration
# IMPORTANT: Use CoilEnumeratorPhaseShift and set num_samples=8 to enumerate all coils
print("Creating dataloader...")
augmentation = Compose([
    Crop(crop_size=(100, 100, 100)),
    CoilEnumeratorPhaseShift(num_coils=8)
])

iterator = MagnetGridIterator(
    BASE_DIR,
    transforms=augmentation,
    num_samples=8  # Must equal num_coils to process each coil once
)

dataloader = DataLoader(
    iterator,
    batch_size=1,  # Must be 1 for coil enumeration
    shuffle=False,  # Don't shuffle to maintain coil order
    num_workers=0,  # Use 0 for debugging, increase for production
    worker_init_fn=worker_init_fn
)

print(f"Dataloader created with {len(iterator)} samples ({len(iterator) // 8} simulations)")

# Create prediction writer
print("Creating prediction writer...")
writer = GridPredictionWriter(
    output_dir=OUTPUT_DIR,
    source_data_dir=BASE_DIR,
    target_normalizer=target_normalizer,
    input_normalizer=input_normalizer,
    model=model,
    device=device,
    field_dtype=np.float32,
    num_coils=8,
)

# Generate and write predictions
print("\nGenerating predictions...")
print("=" * 60)
written_files = writer.write_predictions(dataloader)
print("=" * 60)

# Report results
print(f"\nSuccessfully wrote {len(written_files)} prediction files:")
for sim_name, path in written_files.items():
    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  {sim_name:40s} → {path.name:30s} ({file_size_mb:.1f} MB)")

print(f"\nPredictions saved to: {OUTPUT_DIR}")
print("\nTo verify the output format matches preprocessing:")
print("  1. Check keys: input, efield, hfield, subject, positions")
print("  2. Check efield/hfield shape: (8, 3, x, y, z)")
print("  3. Check dtype: structured with 're' and 'im' fields")
