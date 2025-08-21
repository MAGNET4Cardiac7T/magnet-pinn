#########################################
# Example Script for Grid Preprocessing #
#########################################

from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.transforms import Crop, GridPhaseShift, Compose, DefaultTransform
import tqdm


augmentation = Compose(
    [
        Crop(crop_size=(100, 100, 100)),
        GridPhaseShift(num_coils=8)
    ]
)

#augmentation = Crop(crop_size=(100, 100, 100))



iterator = MagnetGridIterator(
    "/home/andi/coding/data/magnet/processed/train/grid_voxel_size_4_data_type_float32",
    transforms=augmentation,
    num_samples=1
)


for item in tqdm.tqdm(iterator, smoothing=0):
    print(item['field'].shape)
    print(item['coils'].shape)

