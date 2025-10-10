import pytest
import numpy as np

from magnet_pinn.preprocessing.reading_field import (
    FieldReaderFactory,
    GridReader,
    PointReader,
    E_FIELD_DATABASE_KEY,
    H_FIELD_DATABASE_KEY
)


def test_valid_grid_e_field(e_field_grid_data):
    reader = FieldReaderFactory(
        e_field_grid_data, E_FIELD_DATABASE_KEY
    ).create_reader()

    assert isinstance(reader, GridReader)

    coordinates = reader.coordinates
    assert len(coordinates) == 3
    assert len(coordinates[0]) == 121
    assert len(coordinates[1]) == 111
    assert len(coordinates[2]) == 126

    data = reader.extract_data()
    assert data.shape == (3, 121, 111, 126, 2)
    assert data.dtype == np.complex64


def test_valid_grid_h_field(h_field_grid_data):
    reader = FieldReaderFactory(
        h_field_grid_data, H_FIELD_DATABASE_KEY
    ).create_reader()

    assert isinstance(reader, GridReader)

    coordinates = reader.coordinates
    assert len(coordinates) == 3
    assert len(coordinates[0]) == 121
    assert len(coordinates[1]) == 111
    assert len(coordinates[2]) == 126

    data = reader.extract_data()
    assert data.shape == (3, 121, 111, 126, 2)
    assert data.dtype == np.complex64


def test_valid_grid_e_field_with_mixed_axis_order(e_field_grid_data_with_mixed_axis):
    reader = FieldReaderFactory(
        e_field_grid_data_with_mixed_axis, E_FIELD_DATABASE_KEY
    ).create_reader()

    assert isinstance(reader, GridReader)

    coordinates = reader.coordinates
    assert len(coordinates) == 3
    assert len(coordinates[0]) == 121
    assert len(coordinates[1]) == 111
    assert len(coordinates[2]) == 126

    data = reader.extract_data()
    assert data.shape == (3, 121, 111, 126, 2)
    assert data.dtype == np.complex64


def test_valid_grid_h_field_with_mixed_axis_order(h_field_grid_data_with_mixed_axis):
    reader = FieldReaderFactory(
        h_field_grid_data_with_mixed_axis, H_FIELD_DATABASE_KEY
    ).create_reader()

    assert isinstance(reader, GridReader)

    coordinates = reader.coordinates
    assert len(coordinates) == 3
    assert len(coordinates[0]) == 121
    assert len(coordinates[1]) == 111
    assert len(coordinates[2]) == 126

    data = reader.extract_data()
    assert data.shape == (3, 121, 111, 126, 2)
    assert data.dtype == np.complex64


def test_grid_e_field_with_inconsistent_shape(e_field_grid_data_with_inconsistent_shape):
    reader = FieldReaderFactory(
        e_field_grid_data_with_inconsistent_shape, E_FIELD_DATABASE_KEY
    ).create_reader()

    with pytest.raises(ValueError):
        reader.extract_data()


def test_grid_h_field_with_inconsistent_shape(h_field_grid_data_with_inconsistent_shape):
    reader = FieldReaderFactory(
        h_field_grid_data_with_inconsistent_shape, E_FIELD_DATABASE_KEY
    ).create_reader()

    with pytest.raises(ValueError):
        reader.extract_data()


def test_valid_grid_to_pointslist(e_field_grid_data):
    reader = FieldReaderFactory(
        e_field_grid_data, E_FIELD_DATABASE_KEY
    ).create_reader(keep_grid_output_format=False)

    assert isinstance(reader, GridReader)
    assert reader.is_grid == False


def test_valid_pointslist_e_field(e_field_pointslist_data):
    reader = FieldReaderFactory(
        e_field_pointslist_data, E_FIELD_DATABASE_KEY
    ).create_reader()

    assert isinstance(reader, PointReader)

    coordinates = reader.coordinates
    assert coordinates.shape == (100, 3)
    assert coordinates.dtype == np.float32


def test_valid_pointslist_h_field(h_field_pointslist_data):
    reader = FieldReaderFactory(
        h_field_pointslist_data, H_FIELD_DATABASE_KEY
    ).create_reader()

    assert isinstance(reader, PointReader)

    coordinates = reader.coordinates
    assert coordinates.shape == (100, 3)
    assert coordinates.dtype == np.float32


def test_invalid_path_reader(e_field_grid_data):
    with pytest.raises(FileNotFoundError):
        FieldReaderFactory(
            e_field_grid_data / "invalid_path", E_FIELD_DATABASE_KEY
        )


def test_invalid_key_reader(e_field_grid_data):
    with pytest.raises(KeyError):
        FieldReaderFactory(
            e_field_grid_data, "INVALID_KEY"
        )


def test_grid_coordinates_dtype(e_field_grid_data):
    reader = FieldReaderFactory(
        e_field_grid_data, E_FIELD_DATABASE_KEY
    ).create_reader()

    coordinates = reader.coordinates
    assert coordinates[0].dtype == np.float64
    assert coordinates[1].dtype == np.float64
    assert coordinates[2].dtype == np.float64


def test_pointslist_coordinates_consistency(e_field_pointslist_data):
    e_reader = FieldReaderFactory(
        e_field_pointslist_data, E_FIELD_DATABASE_KEY
    ).create_reader()
    
    h_reader = FieldReaderFactory(
        e_field_pointslist_data, H_FIELD_DATABASE_KEY
    ).create_reader()

    assert np.array_equal(e_reader.coordinates, h_reader.coordinates)


def test_grid_coordinates_are_consistent_across_files(e_field_grid_data):
    reader = FieldReaderFactory(
        e_field_grid_data, E_FIELD_DATABASE_KEY
    ).create_reader()
    
    assert len(reader.files_list) == 2
    
    coordinates = reader.coordinates
    assert len(coordinates) == 3
    assert coordinates[0].shape[0] == 121
    assert coordinates[1].shape[0] == 111
    assert coordinates[2].shape[0] == 126


def test_grid_e_and_h_fields_have_same_coordinates(e_field_grid_data, h_field_grid_data):
    e_reader = FieldReaderFactory(
        e_field_grid_data, E_FIELD_DATABASE_KEY
    ).create_reader()
    
    h_reader = FieldReaderFactory(
        h_field_grid_data, H_FIELD_DATABASE_KEY
    ).create_reader()

    e_coords = e_reader.coordinates
    h_coords = h_reader.coordinates
    
    assert np.array_equal(e_coords[0], h_coords[0])
    assert np.array_equal(e_coords[1], h_coords[1])
    assert np.array_equal(e_coords[2], h_coords[2])


def test_grid_to_pointslist_coordinates_shape(e_field_grid_data):
    reader = FieldReaderFactory(
        e_field_grid_data, E_FIELD_DATABASE_KEY
    ).create_reader(keep_grid_output_format=False)
    
    coordinates = reader.coordinates
    
    expected_n_points = 121 * 111 * 126
    assert coordinates.shape == (expected_n_points, 3)
    assert coordinates.dtype == np.float32


def test_grid_mismatched_coordinates_raises_exception(tmp_path):
    from tests.preprocessing.helpers import create_grid_field
    from magnet_pinn.preprocessing.reading_field import FIELD_DIR_PATH
    
    field_path = tmp_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)
    
    bounds1 = np.array([[-240, -220, -250], [240, 220, 250]])
    bounds2 = np.array([[-250, -220, -250], [240, 220, 250]])
    
    create_grid_field(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds1,
        0
    )
    
    create_grid_field(
        field_path / "e-field (f=297.2) [AC2].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds2,
        0
    )
    
    with pytest.raises(Exception, match="Different positions"):
        FieldReaderFactory(tmp_path, E_FIELD_DATABASE_KEY).create_reader()


def test_pointslist_mismatched_coordinates_raises_exception(tmp_path):
    from tests.preprocessing.helpers import create_pointslist_field
    from magnet_pinn.preprocessing.reading_field import FIELD_DIR_PATH, POSITIONS_DATABASE_KEY
    from h5py import File
    
    field_path = tmp_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)
    
    file1 = field_path / "e-field (f=297.2) [AC1].h5"
    file2 = field_path / "e-field (f=297.2) [AC2].h5"
    
    create_pointslist_field(file1, E_FIELD_DATABASE_KEY)
    create_pointslist_field(file2, E_FIELD_DATABASE_KEY)
    
    with File(file2, "a") as f:
        del f[POSITIONS_DATABASE_KEY]
        f.create_dataset(
            POSITIONS_DATABASE_KEY,
            data=np.array(
                np.ones(100),
                dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')]
            )
        )
    
    with pytest.raises(Exception, match="Different positions"):
        FieldReaderFactory(tmp_path, E_FIELD_DATABASE_KEY).create_reader()


def test_grid_single_ac_file(tmp_path):
    from tests.preprocessing.helpers import create_grid_field
    from magnet_pinn.preprocessing.reading_field import FIELD_DIR_PATH
    
    field_path = tmp_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)
    
    bounds = np.array([[-240, -220, -250], [240, 220, 250]])
    
    create_grid_field(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds,
        0
    )
    
    reader = FieldReaderFactory(tmp_path, E_FIELD_DATABASE_KEY).create_reader()
    
    assert len(reader.files_list) == 1
    
    data = reader.extract_data()
    assert data.shape == (3, 121, 111, 126, 1)
    assert data.dtype == np.complex64


def test_pointslist_single_ac_file(tmp_path):
    from tests.preprocessing.helpers import create_pointslist_field
    from magnet_pinn.preprocessing.reading_field import FIELD_DIR_PATH
    
    field_path = tmp_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)
    
    create_pointslist_field(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY
    )
    
    reader = FieldReaderFactory(tmp_path, E_FIELD_DATABASE_KEY).create_reader()
    
    assert isinstance(reader, PointReader)
    assert len(reader.files_list) == 1
    
    coordinates = reader.coordinates
    assert coordinates.shape == (100, 3)
    assert coordinates.dtype == np.float32






