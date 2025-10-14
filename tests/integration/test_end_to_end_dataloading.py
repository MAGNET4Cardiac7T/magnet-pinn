import numpy as np

from magnet_pinn.preprocessing.preprocessing import (
    GridPreprocessing,
    PointPreprocessing,
)
from magnet_pinn.data._base import MagnetBaseIterator
from magnet_pinn.data.transforms import Compose, Crop, GridPhaseShift, PointSampling, PointPhaseShift
from tests.preprocessing.helpers import (
    CENTRAL_SPHERE_SIM_NAME,
    create_central_batch,
    create_antenna_test_data,
)



def test_grid_preprocess_then_iterate(data_dir_path, processed_dir_path):
    # Prepare raw inputs
    raw_central_batch_dir_path = create_central_batch(data_dir_path)
    raw_antenna_dir_path = create_antenna_test_data(data_dir_path)

    # Preprocess a single simulation into a grid case directory
    p = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_dir_path,
        field_dtype=np.float32,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4,
        voxel_size=1,
    )
    p.process_simulations([CENTRAL_SPHERE_SIM_NAME])

    # Iterator expects the case directory (parent of "simulations")
    case_dir = p.out_simulations_dir_path.parent

    augmentation = Compose(
    [
        Crop(crop_size=(4, 4, 4)),
        GridPhaseShift(num_coils=4)
    ]
)

    it = MagnetBaseIterator(case_dir, transforms=augmentation, num_samples=1)

    # Exactly one processed simulation yields exactly one item
    assert len(it) == 1
    items = list(it)
    assert len(items) == 1

    result = items[0]

    # Shapes relationship checks for grid
    # positions: (3, x, y, z)
    assert result["positions"].ndim == 4 and result["positions"].shape[0] == 3
    spatial = result["positions"].shape[1:]

    # input: (3, x, y, z)
    assert result["input"].shape == (3, *spatial)

    # field after DefaultTransform: (2 [e/h], 2 [re/im], 3 [x/y/z], x, y, z)
    assert result["field"].shape == (2, 2, 3, *spatial)

    # subject reduced to (x, y, z)
    assert result["subject"].shape == spatial

    # coils after DefaultTransform: (2 [re/im], x, y, z)
    assert result["coils"].shape == (2, *spatial)

    # phase/mask length equals number of coils in the dataset
    num_coils = it.num_coils
    assert result["phase"].shape == (num_coils,)
    assert result["mask"].shape == (num_coils,)

    # truncation coefficients are length-3
    assert result["truncation_coefficients"].shape == (3,)


def test_point_preprocess_then_iterate(data_dir_path, processed_dir_path):
    # Prepare raw inputs
    raw_central_batch_dir_path = create_central_batch(data_dir_path)
    raw_antenna_dir_path = create_antenna_test_data(data_dir_path)

    # Preprocess a single simulation into a point case directory
    p = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_dir_path,
        field_dtype=np.float32,
    )
    p.process_simulations([CENTRAL_SPHERE_SIM_NAME])

    # Iterator expects the case directory (parent of "simulations")
    case_dir = p.out_simulations_dir_path.parent

    augmentation = Compose(
    [
        PointSampling(points_sampled=10),
        PointPhaseShift(num_coils=4)
    ]
)

    it = MagnetBaseIterator(case_dir, transforms=augmentation, num_samples=1)

    # Exactly one processed simulation yields exactly one item
    assert len(it) == 1
    items = list(it)
    assert len(items) == 1

    result = items[0]

    # Shapes relationship checks for pointcloud
    # positions: (3, points)
    assert result["positions"].ndim == 2 and result["positions"].shape[0] == 3
    points = result["positions"].shape[1]

    # input: (3, points)
    assert result["input"].shape == (3, points)

    # field after DefaultTransform: (2 [e/h], 2 [re/im], 3 [x/y/z], points)
    assert result["field"].shape == (2, 2, 3, points)

    # subject reduced to (points,)
    assert result["subject"].shape == (points,)

    # coils after DefaultTransform: (2 [re/im], points)
    assert result["coils"].shape == (2, points)

    # phase/mask length equals number of coils in the dataset
    num_coils = it.num_coils
    assert result["phase"].shape == (num_coils,)
    assert result["mask"].shape == (num_coils,)

    # truncation coefficients are length-3
    assert result["truncation_coefficients"].shape == (3,)
