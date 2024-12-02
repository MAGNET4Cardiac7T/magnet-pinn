from magnet_pinn.utils import MinMaxNormalizer, StandardNormalizer
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.augmentations import CropAugmentation, GridPhaseAugmentation, ComposeAugmentation

import numpy as np
import einops

class Iterator:
    def __init__(self, path):
        self.path = path
        augmentation = ComposeAugmentation(
            [
                CropAugmentation(crop_size=(100, 100, 100)),
                GridPhaseAugmentation(num_coils=8)
            ]
        )



        self.iterator = MagnetGridIterator(
            path,
            augmentation=augmentation,
            num_augmentations=1
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

iterator = Iterator("data/processed/train/grid_voxel_size_4_data_type_float32")

input_normalizer = StandardNormalizer()
input_normalizer.fit_params(iterator, key='input', axis=0)
input_normalizer.save_as_json("data/processed/train/grid_voxel_size_4_data_type_float32/normalization/input_normalization.json")