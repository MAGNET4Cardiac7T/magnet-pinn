import pytest

from magnet_pinn.preprocessing.reading_field import (
    FieldReaderFactory,
    GridReader,
    PointReader,
    E_FIELD_DATABASE_KEY,
    H_FIELD_DATABASE_KEY
)


def test_grid_e_field_valid_arguments(e_field_grid_data):
    reader = FieldReaderFactory(
        e_field_grid_data, E_FIELD_DATABASE_KEY
    ).create_reader()

    assert isinstance(reader, GridReader)


def test_grid_h_field_valid_arguments(h_field_grid_data):
    reader = FieldReaderFactory(
        h_field_grid_data, H_FIELD_DATABASE_KEY
    ).create_reader()

    assert isinstance(reader, GridReader)


def test_pointslist_e_field_valid_arguments(e_field_pointslist_data):
    reader = FieldReaderFactory(
        e_field_pointslist_data, E_FIELD_DATABASE_KEY
    ).create_reader()

    assert isinstance(reader, PointReader)


def test_pointslist_h_field_valid_arguments(h_field_pointslist_data):
    reader = FieldReaderFactory(
        h_field_pointslist_data, H_FIELD_DATABASE_KEY
    ).create_reader()

    assert isinstance(reader, PointReader)


def test_invalid_path(e_field_grid_data):
    with pytest.raises(FileNotFoundError):
        FieldReaderFactory(
            e_field_grid_data / "invalid_path", E_FIELD_DATABASE_KEY
        )


def test_invalid_key(e_field_grid_data):
    with pytest.raises(KeyError):
        FieldReaderFactory(
            e_field_grid_data, "INVALID_KEY"
        )
