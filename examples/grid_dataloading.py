from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.transforms import Crop, GridPhaseShift, Compose
import tqdm


augmentation = Compose(
    [
        Crop(crop_size=(100, 100, 100)),
        GridPhaseShift(num_coils=8)
    ]
)



iterator = MagnetGridIterator(
    "data/processed/train/grid_voxel_size_4_data_type_float32",
    augmentation=augmentation,
    num_augmentations=100
)


for item in tqdm.tqdm(iterator):
    pass

