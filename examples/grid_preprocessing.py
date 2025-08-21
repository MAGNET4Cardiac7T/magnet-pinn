#########################################
# Example Script for Grid Preprocessing #
#########################################

from magnet_pinn.preprocessing.preprocessing import GridPreprocessing
import numpy as np

preprocessor = GridPreprocessing(
    ["data/raw/batches/batch_6"],
    "data/raw/antenna",
    "data/processed/train",
    field_dtype=np.float32,
    x_min=-240,
    x_max=240,
    y_min=-220,
    y_max=220,
    z_min=-250,
    z_max=250
)

preprocessor.process_simulations()