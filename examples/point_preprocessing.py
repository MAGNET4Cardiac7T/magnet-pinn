from magnet_pinn.preprocessing.preprocessing import PointPreprocessing
import numpy as np

point_preprocessor = PointPreprocessing(
    ["data/raw/batches/batch_1", "data/raw/batches/batch_2"],
    "data/raw/antenna",
    "data/processed/train",
    field_dtype=np.float32
)

point_preprocessor.process_simulations()