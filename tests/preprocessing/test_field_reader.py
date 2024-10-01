import pytest

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


def test_valid_pointslist_e_field(e_field_pointslist_data):
    reader = FieldReaderFactory(
        e_field_pointslist_data, E_FIELD_DATABASE_KEY
    ).create_reader()

    assert isinstance(reader, PointReader)


def test_valid_pointslist_h_field(h_field_pointslist_data):
    reader = FieldReaderFactory(
        h_field_pointslist_data, H_FIELD_DATABASE_KEY
    ).create_reader()

    assert isinstance(reader, PointReader)


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
