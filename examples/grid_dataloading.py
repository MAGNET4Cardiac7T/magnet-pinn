#########################################
# Example Script for Grid Preprocessing #
#########################################

from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.transforms import Crop, GridPhaseShift, Compose, DefaultTransform
import tqdm

BASE_DIR = "data/processed/train/grid_voxel_size_4_data_type_float32"

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


for item in tqdm.tqdm(iterator, smoothing=0):
    print(item['field'].shape)
    print(item['coils'].shape)

