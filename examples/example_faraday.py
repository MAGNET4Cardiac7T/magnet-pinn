####################################################
# Example Script calculating divergence of a field #
####################################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import einops

from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.transforms import Crop, GridPhaseShift, Compose
from magnet_pinn.losses.physics import DivergenceLoss, FaradaysLawLoss
from magnet_pinn.losses.utils import ObjectMaskCropping


def prepare_data():

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

    dataloader = torch.utils.data.DataLoader(
        iterator,
        batch_size=1,
        num_workers=1,
    )
    
    mask_cropper = ObjectMaskCropping(padding=1)

    item = next(iter(dataloader))
    item['subject'] = mask_cropper(item['subject'].type(torch.bool))
    return item

item = prepare_data()


faradayslaw_loss_fn = FaradaysLawLoss(reduction=None, dx=0.004)
faradayslaw_loss_fn2 = FaradaysLawLoss(reduction=None, dx=1)

field = item['field']
field_dict = {
    'efield_real': field[:, 0, 0],
    'efield_imag': field[:, 0, 1],
    'hfield_real': field[:, 1, 0],
    'hfield_imag': field[:, 1, 1]
    
}
efield_abs = torch.linalg.norm(field_dict['efield_real'] + 1j*field_dict['efield_imag'], dim=1)
hfield_abs = torch.linalg.norm(field_dict['hfield_real'] + 1j*field_dict['hfield_imag'], dim=1)

faradayslaw_residual = faradayslaw_loss_fn(tuple(field_dict.values()), None)
faradayslaw_residual_cropped = faradayslaw_residual * item['subject']

faradayslaw_residual2 = faradayslaw_loss_fn2(tuple(field_dict.values()), None)
faradayslaw_residual_cropped2 = faradayslaw_residual2 * item['subject']

fig, ax = plt.subplots(1, 3, figsize=(15, 45))

im0 = ax[0].imshow(efield_abs[0, :, :, 60], norm='log')
cax0 = make_axes_locatable(ax[0]).append_axes("right", size="5%", pad=0.05)
fig.colorbar(im0, cax=cax0)
ax[0].set_title("E-field Magnitude")

im1 = ax[1].imshow(hfield_abs[0, :, :, 60], norm='log')
cax1 = make_axes_locatable(ax[1]).append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1)
ax[1].set_title("H-field Magnitude")

im2 = ax[2].imshow(faradayslaw_residual_cropped[0, :, :, 60], norm='log')
cax2 = make_axes_locatable(ax[2]).append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2)
ax[2].set_title('Faraday\'s Law Residual')


plt.tight_layout()
plt.savefig('faraday_example.png', bbox_inches='tight')