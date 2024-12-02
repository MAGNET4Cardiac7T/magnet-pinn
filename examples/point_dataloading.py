from magnet_pinn.data.point import MagnetPointIterator
from magnet_pinn.data.transforms import PointSampling, PointPhaseShift, Compose
import tqdm

augmentation = Compose(
    [
        PointSampling(points_sampled=1000),
        PointPhaseShift(num_coils=8)
    ]
)

iterator = MagnetPointIterator(
    "data/processed/train/point_data_type_float32",
    augmentation=augmentation,
    num_augmentations=100
)

for item in tqdm.tqdm(iterator):
    pass

