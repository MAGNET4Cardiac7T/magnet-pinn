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
        [Crop(crop_size=(100, 100, 100)), GridPhaseShift(num_coils=8)]
    )

    iterator = MagnetGridIterator(BASE_DIR, transforms=augmentation, num_samples=100)

    dataloader = torch.utils.data.DataLoader(
        iterator,
        batch_size=1,
        num_workers=1,
    )

    mask_cropper = ObjectMaskCropping(padding=1)

    item = next(iter(dataloader))
    item["subject"] = mask_cropper(item["subject"].type(torch.bool))
    return item


item = prepare_data()


div_loss_fn = DivergenceLoss(reduction=None, dx=0.004)

field = item["field"]
field_dict = {
    "efield_real": field[:, 0, 0],
    "efield_imag": field[:, 0, 1],
    "hfield_real": field[:, 1, 0],
    "hfield_imag": field[:, 1, 1],
}
efield_abs = torch.linalg.norm(
    field_dict["efield_real"] + 1j * field_dict["efield_imag"], dim=1
)
hfield_abs = torch.linalg.norm(
    field_dict["hfield_real"] + 1j * field_dict["hfield_imag"], dim=1
)

divergence_dict = {
    key: div_loss_fn(value, mask=item["subject"]) for key, value in field_dict.items()
}

divergence_cropped_dict = {
    key: divergence_dict[key] * item["subject"] for key in divergence_dict
}
hfield_divergence = (
    divergence_cropped_dict["hfield_real"] + divergence_cropped_dict["hfield_imag"]
)
efield_divergence = (
    divergence_cropped_dict["efield_real"] + divergence_cropped_dict["efield_imag"]
)


fig, ax = plt.subplots(2, 2, figsize=(10, 10))

im00 = ax[0, 0].imshow(efield_abs[0, :, :, 60], norm="log")
cax00 = make_axes_locatable(ax[0, 0]).append_axes("right", size="5%", pad=0.05)
fig.colorbar(im00, cax=cax00)
ax[0, 0].set_title("E-field Magnitude")

im01 = ax[0, 1].imshow(efield_divergence[0, :, :, 60], norm="log")
cax01 = make_axes_locatable(ax[0, 1]).append_axes("right", size="5%", pad=0.05)
fig.colorbar(im01, cax=cax01)
ax[0, 1].set_title("E-Field Divergence")

im10 = ax[1, 0].imshow(hfield_abs[0, :, :, 60], norm="log")
cax10 = make_axes_locatable(ax[1, 0]).append_axes("right", size="5%", pad=0.05)
fig.colorbar(im10, cax=cax10)
ax[1, 0].set_title("H-field Magnitude")

im11 = ax[1, 1].imshow(hfield_divergence[0, :, :, 60], norm="log")
cax11 = make_axes_locatable(ax[1, 1]).append_axes("right", size="5%", pad=0.05)
fig.colorbar(im11, cax=cax11)
ax[1, 1].set_title("H-Field Divergence")

plt.tight_layout()
plt.savefig("divergence_example.png", bbox_inches="tight")
