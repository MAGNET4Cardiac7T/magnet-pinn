from magnet_pinn.data.point import MagnetPointIterator
import tqdm

iterator = MagnetPointIterator(
    "data/processed/train/point_data_type_float32",
    phase_samples_per_simulation=100,
)

for item in tqdm.tqdm(iterator):
    pass

