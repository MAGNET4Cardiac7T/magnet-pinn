#########################################
# Example Script for plotting grid data #
#########################################

from magnet_pinn.data.grid import MagnetGridIterator
import numpy as np
import matplotlib.pyplot as plt
from magnet_pinn.data.transforms import Crop, GridPhaseShift, Compose, DefaultTransform

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

item = next(iter(iterator))
abs_efield = np.linalg.norm(item['field'][0], axis=(0, 1))*item['subject']
coils_real = item['coils'][0, :, :, :]
coils_imag = item['coils'][1, :, :, :]

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(abs_efield[:, :, 60], norm='log')
ax[0].set_title('abs_efield')

ax[1].imshow(coils_real[:, :, 60])
ax[1].set_title('coils_real')

ax[2].imshow(coils_imag[:, :, 60])
ax[2].set_title('coils_imag')

plt.savefig('grid_data_example.png')