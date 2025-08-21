#########################################
# Example Script for Point Data Loading #
#########################################
from magnet_pinn.data.point import MagnetPointIterator
from magnet_pinn.data.transforms import PointSampling, PointPhaseShift, Compose, PointFeatureRearrange
import tqdm

augmentation = Compose(
    [
        PointSampling(points_sampled=1000),
        PointPhaseShift(num_coils=8),
        PointFeatureRearrange(num_coils=8)
    ]
)

iterator = MagnetPointIterator(
    "data/processed/train/point_data_type_float32",
    transforms=augmentation,
    num_samples=100
)

for item in tqdm.tqdm(iterator):
    print(item['field'].shape)

