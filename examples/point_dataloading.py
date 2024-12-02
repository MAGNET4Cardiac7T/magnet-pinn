from magnet_pinn.data.point import MagnetPointIterator
from magnet_pinn.data.augmentations import PointSamplingAugmentation, PointPhaseAugmentation, ComposeAugmentation
import tqdm

augmentation = ComposeAugmentation(
    [
        PointSamplingAugmentation(points_sampled=1000),
        PointPhaseAugmentation(num_coils=8)
    ]
)

iterator = MagnetPointIterator(
    "data/processed/train/point_data_type_float32",
    augmentation=augmentation,
    num_augmentations=100
)

for item in tqdm.tqdm(iterator):
    pass

