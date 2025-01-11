from os import listdir
from pathlib import Path
from shutil import rmtree

import pytest
import numpy as np
from h5py import File

from tests.preprocessing.conftest import (
    CENTRAL_SPHERE_SIM_NAME, CENTRAL_BOX_SIM_NAME,
    SHIFTED_BOX_SIM_NAME, SHIFTED_SPHERE_SIM_NAME,
    ALL_SIM_NAMES
)
from magnet_pinn.preprocessing.preprocessing import (
    GridPreprocessing, PointPreprocessing, 
    PROCESSED_ANTENNA_DIR_PATH, PROCESSED_SIMULATIONS_DIR_PATH, TARGET_FILE_NAME,
    ANTENNA_MASKS_OUT_KEY, E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, 
    SUBJECT_OUT_KEY, COORDINATES_OUT_KEY, DTYPE_OUT_KEY,
    TRUNCATION_COEFFICIENTS_OUT_KEY, COORDINATES_OUT_KEY,
    MIN_EXTENT_OUT_KEY, MAX_EXTENT_OUT_KEY, VOXEL_SIZE_OUT_KEY
)


def test_grid_out_dir_structure(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    p = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
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
    p.process_simulations()

    case_name = "grid_voxel_size_1_data_type_complex64"
    expected_batch_case_dir = processed_batch_dir_path / case_name

    assert expected_batch_case_dir == p.out_antenna_dir_path.parent
    assert expected_batch_case_dir == p.out_simulations_dir_path.parent
    assert expected_batch_case_dir.exists()

    expected_antenna_dir = expected_batch_case_dir / PROCESSED_ANTENNA_DIR_PATH
    assert p.out_antenna_dir_path == expected_antenna_dir
    assert expected_antenna_dir.exists()

    expected_simulations_dir = expected_batch_case_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert p.out_simulations_dir_path == expected_simulations_dir
    assert expected_simulations_dir.exists()

    for simulation_dir in expected_simulations_dir.iterdir():
        assert simulation_dir.is_file()


def test_grid_antenna(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    p = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
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
    p.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    out_antenna_file = Path(p.out_antenna_dir_path) / TARGET_FILE_NAME.format(name="antenna")
    assert out_antenna_file.exists()

    with File(out_antenna_file) as f:
        assert list(f.keys()) == [ANTENNA_MASKS_OUT_KEY]
        assert list(f.attrs.keys()) == []
        masks = f[ANTENNA_MASKS_OUT_KEY][:]
        assert masks.shape == (9, 9, 9, 4)
        assert masks.dtype == np.bool_

        received_mask = np.sum(masks[:, :, 3, :], axis=-1)
        expected_mask = np.zeros((9, 9))
        expected_mask[3:6, 0:3] = 1
        expected_mask[3:6, 6:9] = 1
        expected_mask[0:3, 3:6] = 1
        expected_mask[6:9, 3:6] = 1
        assert np.equal(received_mask, expected_mask).all()


def test_grid_out_simulation_structure(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    p = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
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
    p.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    sim_file = p.out_simulations_dir_path /TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)

    with File(sim_file) as f:
        assert set(f.keys()) == {E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY}
        assert set(f.attrs.keys()) == {DTYPE_OUT_KEY, TRUNCATION_COEFFICIENTS_OUT_KEY, MIN_EXTENT_OUT_KEY, MAX_EXTENT_OUT_KEY, VOXEL_SIZE_OUT_KEY}


def test_grid_central_complex_one_simulation_valid_preprocessing(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
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
        [CENTRAL_SPHERE_SIM_NAME]
    )

    case_name = f"grid_voxel_size_1_data_type_complex64"
    out_dir = processed_batch_dir_path / case_name

    out_simulations_dir = out_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert out_simulations_dir.exists()

    out_simulation_file = out_simulations_dir / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    assert out_simulation_file.exists()

    assert len(list(listdir(out_simulations_dir))) == 1

    with File(out_simulation_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]
        check_complex_fields(f)
        check_central_features(f)
        check_central_subject_mask(f)


def test_grid_central_complex_multiple_simulations_valid_preprocessing(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
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
        [CENTRAL_SPHERE_SIM_NAME, CENTRAL_BOX_SIM_NAME]
    )

    case_name = f"grid_voxel_size_1_data_type_complex64"
    out_dir = processed_batch_dir_path / case_name

    out_simulations_dir = out_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert out_simulations_dir.exists()

    out_simulation_file = out_simulations_dir / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    assert out_simulation_file.exists()

    assert len(list(listdir(out_simulations_dir))) == 2

    with File(out_simulation_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]
        check_complex_fields(f)
        check_central_features(f)
        check_central_subject_mask(f)

    next_sim_file = out_simulations_dir / TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)
    assert next_sim_file.exists()

    with File(next_sim_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]
        check_complex_fields(f)
        check_central_features(f)
        check_central_subject_mask(f)


def check_antenna(out_dir: str):
    out_antenna_dir = out_dir / PROCESSED_ANTENNA_DIR_PATH
    assert out_antenna_dir.exists()

    out_antenna_file = out_antenna_dir / TARGET_FILE_NAME.format(name="antenna")
    assert out_antenna_file.exists()

    with File(out_antenna_file) as f:
        assert list(f.keys()) == [ANTENNA_MASKS_OUT_KEY]
        masks = f[ANTENNA_MASKS_OUT_KEY][:]
        assert masks.shape == (9, 9, 9, 4)
        assert masks.dtype == np.bool_

        received_mask = np.sum(masks[:, :, 3, :], axis=-1)
        expected_mask = np.zeros((9, 9))
        expected_mask[3:6, 0:3] = 1
        expected_mask[3:6, 6:9] = 1
        expected_mask[0:3, 3:6] = 1
        expected_mask[6:9, 3:6] = 1
        assert np.equal(received_mask, expected_mask).all()


def test_grid_central_float_one_simulation_valid_preprocessing(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
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
        [CENTRAL_SPHERE_SIM_NAME]
    )

    case_name = f"grid_voxel_size_1_data_type_float32"
    out_dir = processed_batch_dir_path / case_name
    assert out_dir.exists()

    out_simulations_dir = out_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert out_simulations_dir.exists()

    assert len(list(listdir(out_simulations_dir))) == 1

    out_simulation_file = out_simulations_dir / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    assert out_simulation_file.exists()

    with File(out_simulation_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]
        check_float_fields(f)
        check_central_features(f)
        check_central_subject_mask(f)


def test_grid_central_float_multiple_simulations_valid_preprocessing(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
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
        [CENTRAL_SPHERE_SIM_NAME, CENTRAL_BOX_SIM_NAME]
    )

    case_name = f"grid_voxel_size_1_data_type_float32"
    out_dir = processed_batch_dir_path / case_name
    assert out_dir.exists()

    out_simulations_dir = out_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert out_simulations_dir.exists()

    assert len(list(listdir(out_simulations_dir))) == 2

    out_simulation_file = out_simulations_dir / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    assert out_simulation_file.exists()

    with File(out_simulation_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]
        check_float_fields(f)
        check_central_features(f)
        check_central_subject_mask(f)

    another_sim_file = out_simulations_dir / TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)
    assert another_sim_file.exists()

    with File(another_sim_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]
        check_float_fields(f)
        check_central_features(f)
        check_central_subject_mask(f)


def test_grid_shifted_one_simulation_valid_preprocessing(raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_shifted_batch_dir_path,
        raw_antenna_dir_path,
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
        [SHIFTED_SPHERE_SIM_NAME]
    )

    case_name = f"grid_voxel_size_1_data_type_complex64"
    out_dir = processed_batch_dir_path / case_name

    out_simulations_dir = out_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert out_simulations_dir.exists()

    out_simulation_file = out_simulations_dir / TARGET_FILE_NAME.format(name=SHIFTED_SPHERE_SIM_NAME)
    assert out_simulation_file.exists()

    with File(out_simulation_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]
        check_shifted_subject_mask(f)
        check_shifted_features(f)


def check_complex_fields(f: File):
    expected_field = np.concatenate(
        [
            np.zeros((3, 9, 9, 9, 1), dtype=np.complex64), 
            np.ones((3, 9, 9, 9, 1), dtype=np.complex64),
            np.full(fill_value=2, shape=(3, 9, 9, 9, 1), dtype=np.complex64)
        ],
        axis=-1
    )

    e_field = f[E_FIELD_OUT_KEY][:]
    assert e_field.shape == (3, 9, 9, 9, 3)
    assert e_field.dtype == np.complex64
    assert np.equal(e_field, expected_field).all()

    h_field = f[H_FIELD_OUT_KEY][:]
    assert h_field.shape == (3, 9, 9, 9, 3)
    assert h_field.dtype == np.complex64
    assert np.equal(h_field, expected_field).all()


def check_float_fields(f: File):
    expected_re_field = np.concatenate(
        [
            np.zeros((3, 9, 9, 9, 1), dtype=np.float32), 
            np.ones((3, 9, 9, 9, 1), dtype=np.float32),
            np.full(fill_value=2, shape=(3, 9, 9, 9, 1), dtype=np.float32)
        ],
        axis=-1
    )

    expected_im_field = np.zeros((3, 9, 9, 9, 3), dtype=np.float32)

    e_field = f[E_FIELD_OUT_KEY][:]
    assert e_field.shape == (3, 9, 9, 9, 3)
    assert e_field.dtype == np.dtype([("re", np.float32), ("im", np.float32)])
    re_e_field = e_field["re"]
    assert np.equal(re_e_field, expected_re_field).all()
    im_e_field = e_field["im"]
    assert np.equal(im_e_field, expected_im_field).all()

    h_field = f[H_FIELD_OUT_KEY][:]
    assert h_field.shape == (3, 9, 9, 9, 3)
    assert h_field.dtype == np.dtype([("re", np.float32), ("im", np.float32)])
    re_h_field = h_field["re"]
    assert np.equal(re_h_field, expected_re_field).all()
    im_h_field = h_field["im"]
    assert np.equal(im_h_field, expected_im_field).all()


def check_central_subject_mask(f: File):
    subject_mask = f[SUBJECT_OUT_KEY][:]
    assert subject_mask.shape == (9, 9, 9, 1)
    assert subject_mask.dtype == np.bool_
    expected_subject_mask = np.zeros((9, 9), dtype=np.bool_)
    expected_subject_mask[3:6, 3:6] = 1
    assert np.equal(subject_mask[:, :, 3, 0], expected_subject_mask).all()
    assert np.equal(subject_mask[:, :, 4, 0], expected_subject_mask).all()
    assert np.equal(subject_mask[:, :, 5, 0], expected_subject_mask).all()

def check_shifted_subject_mask(f: File):
    subject_mask = f[SUBJECT_OUT_KEY][:]
    assert subject_mask.shape == (9, 9, 9, 1)
    assert subject_mask.dtype == np.bool_
    expected_subject_mask = np.zeros((9, 9), dtype=np.bool_)
    expected_subject_mask[4:7, 4:7] = 1
    assert np.equal(subject_mask[:, :, 3, 0], expected_subject_mask).all()
    assert np.equal(subject_mask[:, :, 4, 0], expected_subject_mask).all()
    assert np.equal(subject_mask[:, :, 5, 0], expected_subject_mask).all()


def check_central_features(f: File):
    features = f[FEATURES_OUT_KEY][:]
    assert features.shape == (3, 9, 9, 9)
    assert features.dtype == np.float32
    expected_features = np.zeros((9, 9), dtype=np.float32)
    expected_features[0:3, 3:6] = 1
    expected_features[3:6, :] = 1
    expected_features[6:9, 3:6] = 1
    assert np.equal(features[0, :, :, 3], expected_features).all()


def check_shifted_features(f: File):
    features = f[FEATURES_OUT_KEY][:]
    assert features.shape == (3, 9, 9, 9)
    assert features.dtype == np.float32
    expected_features = np.zeros((9, 9), dtype=np.float32)
    expected_features = np.zeros((9, 9), dtype=np.float32)
    expected_features[0:3, 3:6] = 1
    expected_features[3:6, 0:3] = 1
    expected_features[3:6, 6:9] = 1
    expected_features[4:7, 4:7] = 1
    expected_features[6:9, 3:6] = 1
    expected_features[6, 4:6] = 2
    expected_features[4:6, 6] = 2
    assert np.equal(features[0, :, :, 3], expected_features).all()
    assert np.equal(features[0, :, :, 4], expected_features).all()
    assert np.equal(features[0, :, :, 5], expected_features).all()


def test_pointcloud_float_out_dirs(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    out_case_dir = processed_batch_dir_path / "point_data_type_float32"
    assert out_case_dir.exists()


def test_pointcloud_complex_out_dirs(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    out_case_dir = processed_batch_dir_path / "point_data_type_complex64"
    assert out_case_dir.exists()


def test_pointcloud_antenna(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    This testcase checks creation antenna milestones and coils mask correction.
    In details it sequentially checks the directory structure, the existence of 
    the antenna file, after opening the h5 file it checks the databases keys,
    shgapes, datatypes and the correctness of the masks.
    """
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    out_case_dir = processed_batch_dir_path / "point_data_type_float32"
    out_antenna_dir = out_case_dir / PROCESSED_ANTENNA_DIR_PATH
    supposed_antenna_dir = preprop.out_antenna_dir_path
    assert supposed_antenna_dir == out_antenna_dir
    assert out_antenna_dir.exists()

    antenna_file = out_antenna_dir / TARGET_FILE_NAME.format(name="antenna")
    assert antenna_file.exists()

    with File(antenna_file) as f:
        assert set(f.keys()) == set([ANTENNA_MASKS_OUT_KEY])
        masks = f[ANTENNA_MASKS_OUT_KEY][:]
        assert masks.shape == (729, 4)
        assert masks.dtype == np.bool_

        assert len(np.where(masks)[0]) == 108


def test_pointcloud_general_structure_for_one_float_simulation(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    This testcase checks creating general structure of directories and files for float data type.
    """
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    out_case_dir = processed_batch_dir_path / "point_data_type_float32"
    out_sim_dir = out_case_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert preprop.out_simulations_dir_path == out_sim_dir
    assert out_sim_dir.exists()
    assert len(list(listdir(out_sim_dir))) == 1

    exact_sim_file = out_sim_dir / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    assert exact_sim_file.exists()


def test_pointcloud_general_structure_for_one_complex_simulation(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    This testcase checks creating general structure of directories and files for complex data type.
    """
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    out_case_dir = processed_batch_dir_path / "point_data_type_complex64"
    out_sim_dir = out_case_dir / PROCESSED_SIMULATIONS_DIR_PATH
    preprop.out_simulations_dir_path == out_sim_dir
    assert out_sim_dir.exists()
    assert len(list(listdir(out_sim_dir))) == 1

    exact_sim_file = out_sim_dir / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    assert exact_sim_file.exists()


def test_pointcloud_general_structure_for_multiple_simulations(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    This testcase checks creating general structure of directories and files for multiple simulations.
    """
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME,
        CENTRAL_BOX_SIM_NAME
    ])

    out_sim_dir = preprop.out_simulations_dir_path
    assert len(list(listdir(out_sim_dir))) == 2

    first_sim_file = Path(out_sim_dir) / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    assert first_sim_file.exists()

    second_sim_file = Path(out_sim_dir) / TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)
    assert second_sim_file.exists()


def test_pointcloud_resulting_keys(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    with File(sim_file) as f:
        assert set(f.keys()) == set(
            [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY, COORDINATES_OUT_KEY]
        )

        assert set(f.attrs.keys()) == set([DTYPE_OUT_KEY, TRUNCATION_COEFFICIENTS_OUT_KEY])


def test_pointcloud_resulting_dtype_values_for_float(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    with File(sim_file) as f:
        efield = f[E_FIELD_OUT_KEY][:]
        assert efield.dtype == np.dtype([("re", np.float32), ("im", np.float32)])

        hfield = f[H_FIELD_OUT_KEY][:]
        assert hfield.dtype == np.dtype([("re", np.float32), ("im", np.float32)])

        features = f[FEATURES_OUT_KEY][:]
        assert features.dtype == np.float32
        
        dtype = f.attrs[DTYPE_OUT_KEY]
        assert isinstance(dtype, str)
        assert dtype == "float32"


def test_pointcloud_resulting_dtype_values_for_complex(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    with File(sim_file) as f:
        efield = f[E_FIELD_OUT_KEY][:]
        assert efield.dtype == np.complex64

        hfield = f[H_FIELD_OUT_KEY][:]
        assert hfield.dtype == np.complex64

        features = f[FEATURES_OUT_KEY][:]
        assert features.dtype == np.float32

        dtype = f.attrs[DTYPE_OUT_KEY]
        assert isinstance(dtype, str)
        assert dtype == "complex64"


def test_pointcloud_datasets_shapes_and_non_changable_dtypes(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    This test case checks not the values of the resulting datasets, but shapes and dtypes.
    """
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    with File(sim_file) as f:
        efield = f[E_FIELD_OUT_KEY][:]
        assert efield.shape == (729, 3, 3)

        hfield = f[H_FIELD_OUT_KEY][:]
        assert hfield.shape == (729, 3, 3)

        features = f[FEATURES_OUT_KEY][:]
        assert features.shape == (729, 3)

        coordinates = f[COORDINATES_OUT_KEY][:]
        assert coordinates.shape == (729, 3)
        assert coordinates.dtype == np.float32

        subject = f[SUBJECT_OUT_KEY][:]
        assert subject.shape == (729, 1)
        assert subject.dtype == np.bool_


def test_pointcloud_squared_coils_sphere_central_object(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Test case checks exactly the values of the resulting datasets for the 4 squared 
    coils and a central sphere.
    """
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    with File(sim_file) as f:
        efield = f[E_FIELD_OUT_KEY][:]
        
        expected_field = np.concatenate(
            [
                np.zeros((729, 3, 1), dtype=np.complex64), 
                np.ones((729, 3, 1), dtype=np.complex64),
                np.full(fill_value=2, shape=(729, 3, 1), dtype=np.complex64)
            ],
            axis=-1
        )
        assert np.equal(efield, expected_field).all()

        hfield = f[H_FIELD_OUT_KEY][:]
        assert np.equal(hfield, expected_field).all()

        features = f[FEATURES_OUT_KEY][:]
        assert len(np.where(features == 1)[0]) == 327

        subject = f[SUBJECT_OUT_KEY][:]
        len(np.where(subject)[0]) == 1


def test_pointcloud_squared_coils_central_box_object(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_BOX_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)
    with File(sim_file) as f:
        efield = f[E_FIELD_OUT_KEY][:]
        expected_field = np.concatenate(
            [
                np.zeros((729, 3, 1), dtype=np.complex64), 
                np.ones((729, 3, 1), dtype=np.complex64),
                np.full(fill_value=2, shape=(729, 3, 1), dtype=np.complex64)
            ],
            axis=-1
        )
        assert np.equal(efield, expected_field).all()

        hfield = f[H_FIELD_OUT_KEY][:]
        assert np.equal(hfield, expected_field).all()

        features = f[FEATURES_OUT_KEY][:]
        assert len(np.where(features == 1)[0]) == 327

        subject = f[SUBJECT_OUT_KEY][:]
        len(np.where(subject)[0]) == 1


def test_pointcloud_squared_coils_shifted_sphere_object(raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_shifted_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        SHIFTED_SPHERE_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=SHIFTED_SPHERE_SIM_NAME)
    with File(sim_file) as f:
        efield = f[E_FIELD_OUT_KEY][:]
        expected_field = np.concatenate(
            [
                np.zeros((729, 3, 1), dtype=np.complex64), 
                np.ones((729, 3, 1), dtype=np.complex64),
                np.full(fill_value=2, shape=(729, 3, 1), dtype=np.complex64)
            ],
            axis=-1
        )
        assert np.equal(efield, expected_field).all()

        hfield = f[H_FIELD_OUT_KEY][:]
        assert np.equal(hfield, expected_field).all()

        features = f[FEATURES_OUT_KEY][:]
        assert len(np.where(features == 1)[0]) == 327

        subject = f[SUBJECT_OUT_KEY][:]
        len(np.where(subject)[0]) == 1


def test_pointcloud_squared_coils_shifted_box_object(raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_shifted_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        SHIFTED_BOX_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=SHIFTED_BOX_SIM_NAME)
    with File(sim_file) as f:
        efield = f[E_FIELD_OUT_KEY][:]
        expected_field = np.concatenate(
            [
                np.zeros((729, 3, 1), dtype=np.complex64), 
                np.ones((729, 3, 1), dtype=np.complex64),
                np.full(fill_value=2, shape=(729, 3, 1), dtype=np.complex64)
            ],
            axis=-1
        )
        assert np.equal(efield, expected_field).all()

        hfield = f[H_FIELD_OUT_KEY][:]
        assert np.equal(hfield, expected_field).all()

        features = f[FEATURES_OUT_KEY][:]
        assert len(np.where(features == 1)[0]) == 327

        subject = f[SUBJECT_OUT_KEY][:]
        len(np.where(subject)[0]) == 1


def test_pointcloud_invalid_batch_path(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    with pytest.raises(FileNotFoundError):
        preprop = PointPreprocessing(
            "invalid_batch_path",
            raw_antenna_dir_path,
            processed_batch_dir_path,
            field_dtype=np.complex64,
            coil_thick_coef=1.0
        )


def test_pointcloud_with_empty_batch(raw_central_batch_short_term, raw_antenna_dir_path, processed_batch_dir_path):
    list(map(rmtree, raw_central_batch_short_term.iterdir()))
    with pytest.raises(FileNotFoundError):
        p = PointPreprocessing(
            raw_central_batch_short_term,
            raw_antenna_dir_path,
            processed_batch_dir_path,
            field_dtype=np.complex64,
            coil_thick_coef=1.0
        )


def test_pointcloud_with_no_antenna_dir(raw_central_batch_short_term, raw_antenna_dir_path_short_term, processed_batch_dir_path):
    rmtree(raw_antenna_dir_path_short_term)
    with pytest.raises(FileNotFoundError):
        p = PointPreprocessing(
            raw_central_batch_short_term,
            raw_antenna_dir_path_short_term,
            processed_batch_dir_path,
            field_dtype=np.complex64,
            coil_thick_coef=1.0
        )


def test_pointcloud_with_empty_antenna_dir(raw_central_batch_dir_path, raw_antenna_dir_path_short_term, processed_batch_dir_path):
    list(map(lambda x: x.unlink(), raw_antenna_dir_path_short_term.iterdir()))
    with pytest.raises(FileNotFoundError):
        p = PointPreprocessing(
            raw_central_batch_dir_path,
            raw_antenna_dir_path_short_term,
            processed_batch_dir_path,
            field_dtype=np.complex64,
            coil_thick_coef=1.0
        )


def test_multiple_batch_dirs_grid(raw_central_batch_dir_path, raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        [raw_central_batch_dir_path, raw_shifted_batch_dir_path],
        raw_antenna_dir_path,
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
    grid_preprocessor.process_simulations()

    case_name = f"grid_voxel_size_1_data_type_complex64"
    out_dir = processed_batch_dir_path / case_name
    
    assert out_dir.exists()

    assert len(list(listdir(out_dir / PROCESSED_SIMULATIONS_DIR_PATH))) == 4

    out_sim_names = listdir(out_dir / PROCESSED_SIMULATIONS_DIR_PATH)
    out_sim_names = [name.split(".")[0] for name in out_sim_names]
    assert set(out_sim_names) == set(ALL_SIM_NAMES)
