import pytest
from trimesh import Trimesh

from magnet_pinn.preprocessing.reading_properties import (
    PropertyReader, FEATURE_NAMES, FILE_COLUMN_NAME
)


def test_materials_file_absense(grid_simulation_path):
    with pytest.raises(FileNotFoundError):
        reader = PropertyReader(
            grid_simulation_path
        )


def test_properties_frame_without_valid_columns(property_data_invalid_columns):
    with pytest.raises(ValueError):
        reader = PropertyReader(
            property_data_invalid_columns
        )


def test_properties_frame_with_invalid_file(property_data_invalid_file_name):
    reader = PropertyReader(
        property_data_invalid_file_name
    )

    with pytest.raises(FileNotFoundError):
        reader.read_meshes()


def test_properties_valid(property_data_valid):
    reader = PropertyReader(
        property_data_valid
    )

    assert reader.properties.shape[0] == 1
    assert reader.properties.shape[1] == 4
    assert set([FILE_COLUMN_NAME] + FEATURE_NAMES).issubset(reader.properties.columns)

    meshes = reader.read_meshes()
    assert len(meshes) == 1
    mesh = meshes[0]
    assert isinstance(mesh, Trimesh)
    mesh.vertices.shape == (2500, 3)
    mesh.faces.shape == (4802, 3)
