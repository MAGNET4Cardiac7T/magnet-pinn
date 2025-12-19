from typing import Any, List, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pytest

from magnet_pinn.preprocessing.reading_field import (
    E_FIELD_DATABASE_KEY,
    FIELD_DIR_PATH,
    FieldReader,
    FieldReaderFactory,
    GridReader,
    H_FIELD_DATABASE_KEY,
    PointReader,
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

    data = cast(npt.NDArray[np.complex64], reader.extract_data())
    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 3, 121, 111, 126)
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

    data = cast(npt.NDArray[np.complex64], reader.extract_data())
    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 3, 121, 111, 126)
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

    data = cast(npt.NDArray[np.complex64], reader.extract_data())
    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 3, 121, 111, 126)
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

    data = cast(npt.NDArray[np.complex64], reader.extract_data())
    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 3, 121, 111, 126)
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
    assert not reader.is_grid


def test_grid_pointslist_extract_data(tmp_path):
    from tests.preprocessing.helpers import create_grid_field

    field_path = tmp_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    bounds = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)

    create_grid_field(
        field_path / "e-field (f=1.0) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (2, 2, 2),
        bounds,
        0
    )

    reader = FieldReaderFactory(tmp_path, E_FIELD_DATABASE_KEY).create_reader(keep_grid_output_format=False)

    data = reader.extract_data()
    assert data.shape == (1, 3, 8)
    assert data.dtype == np.complex64


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
        (10, 10, 10),
        bounds1,
        0
    )

    create_grid_field(
        field_path / "e-field (f=297.2) [AC2].h5",
        E_FIELD_DATABASE_KEY,
        (10, 10, 10),
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
        (10, 10, 10),
        bounds,
        0
    )

    reader = FieldReaderFactory(tmp_path, E_FIELD_DATABASE_KEY).create_reader()

    assert len(reader.files_list) == 1

    data = reader.extract_data()
    assert data.shape == (1, 3, 10, 10, 10)
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


def test_pointslist_compose_field_pattern_valid(e_field_pointslist_data):
    reader = FieldReaderFactory(
        e_field_pointslist_data, E_FIELD_DATABASE_KEY
    ).create_reader()

    compose_field_pattern_func = PointReader.__dict__["_compose_field_pattern"].fget
    pattern = compose_field_pattern_func(reader, (reader.coordinates.shape[0],))
    assert pattern == "ax batch -> batch ax"


def test_pointslist_compose_field_pattern_invalid(e_field_pointslist_data):
    reader = FieldReaderFactory(
        e_field_pointslist_data, E_FIELD_DATABASE_KEY
    ).create_reader()

    compose_field_pattern_func = PointReader.__dict__["_compose_field_pattern"].fget
    with pytest.raises(ValueError):
        compose_field_pattern_func(reader, (reader.coordinates.shape[0] - 1,))


def test_pointslist_compose_field_components(e_field_pointslist_data):
    reader = FieldReaderFactory(
        e_field_pointslist_data, E_FIELD_DATABASE_KEY
    ).create_reader()

    batch_size = reader.coordinates.shape[0]
    field_components = [
        np.ones((batch_size, 3), dtype=np.complex64),
        np.full((batch_size, 3), 2, dtype=np.complex64)
    ]

    data = reader._compose_field_components(field_components)
    assert data.shape == (2, 3, batch_size)
    assert data.dtype == np.complex64
    assert np.all(data[0] == 1)
    assert np.all(data[1] == 2)


def test_field_reader_base_methods_are_noops():
    class DummyFieldReader(FieldReader):
        def __init__(self) -> None:
            self.files_list: List[str] = []
            self.field_type = "dummy"
            self._test_coordinates: npt.NDArray[np.float64] = np.zeros((1, 3))

        @property
        def coordinates(self) -> Any:
            base_coordinates = FieldReader.__dict__["coordinates"].fget
            return base_coordinates(self)

        def _read_coordinates(
            self, file_path: str
        ) -> Union[Tuple[Any, ...], npt.NDArray[Any]]:
            FieldReader._read_coordinates(self, file_path)
            return self._test_coordinates

        def _check_coordinates(
            self, other_coordinates: Union[Tuple[Any, ...], npt.NDArray[Any]]
        ) -> bool:
            FieldReader._check_coordinates(self, other_coordinates)
            return True

        def _compose_field_pattern(self, data_shape: Tuple[Any, ...]) -> str:
            FieldReader._compose_field_pattern(self, data_shape)
            return "ax batch -> batch ax"

        def _compose_field_components(
            self, field_components: List[npt.NDArray[Any]]
        ) -> npt.NDArray[np.complex64]:
            FieldReader._compose_field_components(self, field_components)
            return np.array(field_components, dtype=np.complex64)

    reader = DummyFieldReader()

    assert reader.coordinates is None
    assert np.array_equal(
        reader._read_coordinates("any"), reader._test_coordinates
    )
    assert reader._check_coordinates(reader._test_coordinates)
    assert reader._compose_field_pattern((1,)) == "ax batch -> batch ax"
    components = reader._compose_field_components([np.zeros((1, 3))])
    assert components.shape == (1, 1, 3)
    assert components.dtype == np.complex64
