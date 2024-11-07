import pytest
import numpy as np
from h5py import File

from tests.conftest import FIRST_SIM_NAME


from magnet_pinn.preprocessing.preprocessing import (
    GridPreprocessing, PROCESSED_ANTENNA_DIR_PATH,
    PROCESSED_SIMULATIONS_DIR_PATH, TARGET_FILE_NAME,
    MASKS_DATABASE_KEY
)


def test_valid_preprocessing(raw_batch_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_batch_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(
        [FIRST_SIM_NAME]
    )

    case_name = f"grid_voxel_size_1_data_type_complex64"
    out_dir = processed_batch_dir_path / case_name
    out_antenna_dir = out_dir / PROCESSED_ANTENNA_DIR_PATH
    out_antenna_file = out_antenna_dir / TARGET_FILE_NAME.format(name="antenna")
    assert out_antenna_dir.exists()
    assert out_antenna_dir.exists()
    assert out_antenna_file.exists()

    with File(out_antenna_file) as f:
        assert f.keys() == [MASKS_DATABASE_KEY]
        masks = f[MASKS_DATABASE_KEY][:]
        assert masks.shape == (9, 9, 9, 4)

    def test_shifted_box_preprocessing_case(raw_batch_dir_path, processed_batch_dir_path):
        grid_preprocessor = GridPreprocessing(
            raw_batch_dir_path,
            processed_batch_dir_path,
            field_dtype=np.float32,
            x_min=-4,
            x_max=4,
            y_min=-4,
            y_max=4,
            z_min=-4,
            z_max=4, 
            voxel_size=1
        )
        grid_preprocessor.process_simulations(
            [FIRST_SIM_NAME]
        )

        case_name = f"grid_voxel_size_2_data_type_float32"
        out_dir = processed_batch_dir_path / case_name
        out_antenna_dir = out_dir / PROCESSED_ANTENNA_DIR_PATH
        out_antenna_file = out_antenna_dir / TARGET_FILE_NAME.format(name="antenna")
        assert out_antenna_dir.exists()
        assert out_antenna_dir.exists()
        assert out_antenna_file.exists()

        with File(out_antenna_file) as f:
            assert f.keys() == [MASKS_DATABASE_KEY]
            masks = f[MASKS_DATABASE_KEY][:]
            assert masks.shape == (9, 9, 9, 4)


    def test_box_object_preprocessing_case(raw_batch_dir_path, processed_batch_dir_path):
        grid_preprocessor = GridPreprocessing(
            raw_batch_dir_path,
            processed_batch_dir_path,
            field_dtype=np.float64,
            x_min=-4,
            x_max=4,
            y_min=-4,
            y_max=4,
            z_min=-4,
            z_max=4, 
            voxel_size=1
        )
        grid_preprocessor.process_simulations(
            [FIRST_SIM_NAME]
        )

        case_name = f"grid_voxel_size_3_data_type_float64"
        out_dir = processed_batch_dir_path / case_name
        out_antenna_dir = out_dir / PROCESSED_ANTENNA_DIR_PATH
        out_antenna_file = out_antenna_dir / TARGET_FILE_NAME.format(name="antenna")
        assert out_antenna_dir.exists()
        assert out_antenna_dir.exists()
        assert out_antenna_file.exists()

        with File(out_antenna_file) as f:
            assert f.keys() == [MASKS_DATABASE_KEY]
            masks = f[MASKS_DATABASE_KEY][:]
            assert masks.shape == (9, 9, 9, 4)


    def test_shifted_object_preprocessing_case(raw_batch_dir_path, processed_batch_dir_path):
        grid_preprocessor = GridPreprocessing(
            raw_batch_dir_path,
            processed_batch_dir_path,
            field_dtype=np.int32,
            x_min=-4,
            x_max=4,
            y_min=-4,
            y_max=4,
            z_min=-4,
            z_max=4, 
            voxel_size=1
        )
        grid_preprocessor.process_simulations(
            [FIRST_SIM_NAME]
        )

        case_name = f"grid_voxel_size_4_data_type_int32"
        out_dir = processed_batch_dir_path / case_name
        out_antenna_dir = out_dir / PROCESSED_ANTENNA_DIR_PATH
        out_antenna_file = out_antenna_dir / TARGET_FILE_NAME.format(name="antenna")
        assert out_antenna_dir.exists()
        assert out_antenna_dir.exists()
        assert out_antenna_file.exists()

        with File(out_antenna_file) as f:
            assert f.keys() == [MASKS_DATABASE_KEY]
            masks = f[MASKS_DATABASE_KEY][:]
            assert masks.shape == (9, 9, 9, 4)