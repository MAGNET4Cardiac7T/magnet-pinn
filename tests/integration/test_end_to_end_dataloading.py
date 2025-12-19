import numpy as np

from magnet_pinn.data._base import MagnetBaseIterator
from magnet_pinn.data.transforms import (
    Compose,
    Crop,
    GridPhaseShift,
    PointPhaseShift,
    PointSampling,
)
from magnet_pinn.preprocessing.preprocessing import (
    GridPreprocessing,
    PointPreprocessing,
)
from tests.preprocessing.helpers import CENTRAL_SPHERE_SIM_NAME


def test_grid_preprocess_then_iterate(raw_central_batch_dir_path, raw_antenna_dir_path, processed_dir_path):
    p = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_dir_path,
        field_dtype=np.dtype(np.float32),
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4,
        voxel_size=1,
    )
    p.process_simulations([CENTRAL_SPHERE_SIM_NAME])

    case_dir = p.out_simulations_dir_path.parent

    augmentation = Compose([
        Crop(crop_size=(4, 4, 4)),
        GridPhaseShift(num_coils=4)
    ])

    it = MagnetBaseIterator(case_dir, transforms=augmentation, num_samples=1)

    assert len(it) == 1
    items = list(it)
    assert len(items) == 1

    result = items[0]

    assert result["positions"].ndim == 4 and result["positions"].shape[0] == 3
    spatial = result["positions"].shape[1:]

    assert result["input"].shape == (3, *spatial)

    assert result["field"].shape == (2, 2, 3, *spatial)

    assert result["subject"].shape == spatial

    assert result["coils"].shape == (2, *spatial)

    num_coils = it.num_coils
    assert result["phase"].shape == (num_coils,)
    assert result["mask"].shape == (num_coils,)

    assert result["truncation_coefficients"].shape == (3,)


def test_point_preprocess_then_iterate(raw_central_batch_dir_path, raw_antenna_dir_path, processed_dir_path):
    p = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_dir_path,
        field_dtype=np.dtype(np.float32),
    )
    p.process_simulations([CENTRAL_SPHERE_SIM_NAME])

    case_dir = p.out_simulations_dir_path.parent

    augmentation = Compose(
        [
            PointSampling(points_sampled=10),
            PointPhaseShift(num_coils=4)
        ]
    )

    it = MagnetBaseIterator(case_dir, transforms=augmentation, num_samples=1)

    assert len(it) == 1
    items = list(it)
    assert len(items) == 1

    result = items[0]

    assert result["positions"].ndim == 2 and result["positions"].shape[0] == 3
    points = result["positions"].shape[1]

    assert result["input"].shape == (3, points)

    assert result["field"].shape == (2, 2, 3, points)

    assert result["subject"].shape == (points,)

    assert result["coils"].shape == (2, points)

    num_coils = it.num_coils
    assert result["phase"].shape == (num_coils,)
    assert result["mask"].shape == (num_coils,)

    assert result["truncation_coefficients"].shape == (3,)
