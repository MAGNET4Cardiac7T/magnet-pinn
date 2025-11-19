import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.utils import worker_init_fn
from magnet_pinn.losses.physics import DivergenceLoss
from magnet_pinn.losses.utils import mask_padding
from magnet_pinn.data.transforms import GridPhaseShift, Compose, Crop


# Set the base directory where the preprocessed data is stored
BASE_DIR = "data/processed/train/grid_voxel_size_4_data_type_float32"
transforms = Compose(
    [
        Crop(crop_size=(100, 100, 100)),
        GridPhaseShift(num_coils=8)
    ]
)
iterator = MagnetGridIterator(
    BASE_DIR,
    transforms=transforms,
    num_samples=100,
)

dataloader = DataLoader(iterator, batch_size=1, num_workers=1, worker_init_fn=worker_init_fn)

item = next(iter(dataloader))

loss_fn = DivergenceLoss()


properties, phase, field, subject_mask = item['input'], item['coils'], item['field'], item['subject']
efield_real = field[:, 1, 0]
efield_imag = field[:, 1, 1]
hfield_real = field[:, 0, 0]
hfield_imag = field[:, 0, 1]
subject_padded = mask_padding(subject_mask, padding=2)

div = loss_fn(hfield_imag, None, subject_padded)
div = div * subject_padded

fig, ax = plt.subplots(1, 3)
ax[0].imshow(subject_mask[0, :, :, 60].detach().cpu().numpy())
ax[1].imshow(subject_padded[0, :, :, 60].detach().cpu().numpy())
im = ax[2].imshow(div[0, :, :, 50].detach().cpu().numpy())
fig.colorbar(im)
plt.savefig("divergence.png")
quit()
# div_space = loss_fn(efield_real, None, ~subject_mask).item()
# print(f"Divergence subject: {div_subject}, divergence space: {div_space}")
