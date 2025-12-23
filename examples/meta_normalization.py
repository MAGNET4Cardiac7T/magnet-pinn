####################################
# Example Script for Normalization #
####################################
# This script demonstrates how to use the MetaNormalizer to fit multiple normalizers
# (e.g., StandardNormalizer) in a single pass over the dataset.
# The MetaNormalizer is particularly useful for reducing the number of iterations
# over the dataset when fitting multiple normalizers.

from magnet_pinn.utils import StandardNormalizer, MetaNormalizer
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.transforms import Crop, GridPhaseShift, Compose

import numpy as np
import einops

class Iterator:
    def __init__(self, path):
        self.path = path
        augmentation = Compose(
            [
                Crop(crop_size=(100, 100, 100)),
                GridPhaseShift(num_coils=8)
            ]
        )
        self.iterator = MagnetGridIterator(
            path,
            transforms=augmentation,
            num_samples=10
        )

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            input = np.concatenate([batch['input'], batch['coils']], axis=0)
            target = einops.rearrange(batch['field'], 'he reim xyz ... -> (he reim xyz) ...')
            yield {
                'input': input,
                'target': target,
            }

# Path to the dataset
path = "data/processed/train/grid_voxel_size_4_data_type_float32"

# Create the iterator
iterator = Iterator(path)

# Define normalizers for input and target
input_normalizer = StandardNormalizer()
target_normalizer = StandardNormalizer()

# Use MetaNormalizer to fit both normalizers in one pass
meta_normalizer = MetaNormalizer([input_normalizer, target_normalizer])
meta_normalizer.fit_params(iterator, keys=["input", "target"], axis=0)

# Save the fitted normalizers using MetaNormalizer's save_as_json method
file_names = ["input_normalization.json", "target_normalization.json"]
meta_normalizer.save_as_json(file_names, base_path=f"{path}/normalization")
