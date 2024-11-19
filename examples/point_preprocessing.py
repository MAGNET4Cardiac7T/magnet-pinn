from magnet_pinn.preprocessing.preprocessing import PointPreprocessing
import numpy as np

point_preprocessor = PointPreprocessing(
    "data/raw/train",
    "data/processed/train",
    field_dtype=np.float32
)

point_preprocessor.process_simulations()