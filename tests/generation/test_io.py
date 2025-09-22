import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

import trimesh

from magnet_pinn.generator.io import (
    Writer, MeshWriter, PARENT_BLOB_FILE_NAME, 
    CHILD_BLOB_FILE_NAME, TUBE_FILE_NAME, MATERIALS_FILE_NAME
)
from magnet_pinn.generator.typing import MeshPhantom, PropertyPhantom, PropertyItem


class ConcreteWriter(Writer):
    def write(self, item):
        pass


def create_property_item(conductivity=0.5, permittivity=80.0, density=1000.0):
    return PropertyItem(
        conductivity=conductivity,
        permittivity=permittivity,
        density=density
    )


def create_mesh_phantom(simple_mesh, num_children=2, num_tubes=1):
    parent = simple_mesh
    children = [simple_mesh for _ in range(num_children)]
    tubes = [simple_mesh for _ in range(num_tubes)]
    
    return MeshPhantom(parent=parent, children=children, tubes=tubes)


def create_property_phantom(num_children=2, num_tubes=1):
    parent = create_property_item(conductivity=0.1)
    children = [create_property_item(conductivity=0.2 + i*0.1) for i in range(num_children)]
    tubes = [create_property_item(conductivity=0.8 + i*0.1) for i in range(num_tubes)]
    
    return PropertyPhantom(parent=parent, children=children, tubes=tubes)


def test_writer_initialization_with_default_directory():
    writer = ConcreteWriter()
    assert writer.dir == Path("data/raw/tissue_meshes")
    assert writer.dir.exists()


def test_writer_initialization_with_custom_directory_as_string(generation_output_dir_path):
    custom_dir = str(generation_output_dir_path / "custom_output")
    writer = ConcreteWriter(output_dir=custom_dir)
    
    assert writer.dir == Path(custom_dir)
    assert writer.dir.exists()


def test_writer_initialization_with_custom_directory_as_path(generation_output_dir_path):
    custom_dir = generation_output_dir_path / "custom_output"
    writer = ConcreteWriter(output_dir=custom_dir)
    
    assert writer.dir == custom_dir
    assert writer.dir.exists()


def test_writer_initialization_creates_nested_directories(generation_output_dir_path):
    nested_dir = generation_output_dir_path / "level1" / "level2" / "level3"
    writer = ConcreteWriter(output_dir=nested_dir)
    
    assert writer.dir == nested_dir
    assert writer.dir.exists()
    assert writer.dir.is_dir()


def test_writer_initialization_with_existing_directory(generation_output_dir_path):
    existing_dir = generation_output_dir_path / "existing"
    existing_dir.mkdir()
    
    writer = ConcreteWriter(output_dir=existing_dir)
    
    assert writer.dir == existing_dir
    assert writer.dir.exists()


def test_writer_abstract_write_method_raises_not_implemented():
    writer = ConcreteWriter()
    with pytest.raises(NotImplementedError):
        Writer.write(writer, None)


def test_mesh_writer_initialization_inherits_from_writer(generation_output_dir_path):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    
    assert isinstance(writer, Writer)
    assert writer.dir == generation_output_dir_path
    assert writer.dir.exists()


def test_mesh_writer_write_creates_parent_stl_file(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=0, num_tubes=0)
    property_phantom = create_property_phantom(num_children=0, num_tubes=0)
    
    writer.write(mesh_phantom, property_phantom)
    
    parent_file = generation_output_dir_path / PARENT_BLOB_FILE_NAME
    assert parent_file.exists()
    assert parent_file.suffix == ".stl"


def test_mesh_writer_write_creates_children_stl_files(generation_output_dir_path, simple_mesh):
    num_children = 3
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=num_children, num_tubes=0)
    property_phantom = create_property_phantom(num_children=num_children, num_tubes=0)
    
    writer.write(mesh_phantom, property_phantom)
    
    for i in range(num_children):
        child_file = generation_output_dir_path / CHILD_BLOB_FILE_NAME.format(i=i+1)
        assert child_file.exists()
        assert child_file.suffix == ".stl"


def test_mesh_writer_write_creates_tubes_stl_files(generation_output_dir_path, simple_mesh):
    num_tubes = 2
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=0, num_tubes=num_tubes)
    property_phantom = create_property_phantom(num_children=0, num_tubes=num_tubes)
    
    writer.write(mesh_phantom, property_phantom)
    
    for i in range(num_tubes):
        tube_file = generation_output_dir_path / TUBE_FILE_NAME.format(i=i+1)
        assert tube_file.exists()
        assert tube_file.suffix == ".stl"


def test_mesh_writer_write_creates_materials_csv_file(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=1, num_tubes=1)
    property_phantom = create_property_phantom(num_children=1, num_tubes=1)
    
    writer.write(mesh_phantom, property_phantom)
    
    materials_file = generation_output_dir_path / MATERIALS_FILE_NAME
    assert materials_file.exists()
    assert materials_file.suffix == ".txt"


def test_mesh_writer_write_materials_csv_contains_correct_columns(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=1, num_tubes=1)
    property_phantom = create_property_phantom(num_children=1, num_tubes=1)
    
    writer.write(mesh_phantom, property_phantom)
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    expected_columns = {'conductivity', 'permittivity', 'density', 'file'}
    assert set(df.columns) == expected_columns


def test_mesh_writer_write_materials_csv_contains_correct_number_of_rows(generation_output_dir_path, simple_mesh):
    num_children = 2
    num_tubes = 3
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=num_children, num_tubes=num_tubes)
    property_phantom = create_property_phantom(num_children=num_children, num_tubes=num_tubes)
    
    writer.write(mesh_phantom, property_phantom)
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    expected_rows = 1 + num_children + num_tubes
    assert len(df) == expected_rows


def test_mesh_writer_write_materials_csv_contains_correct_filenames(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=2, num_tubes=1)
    property_phantom = create_property_phantom(num_children=2, num_tubes=1)
    
    writer.write(mesh_phantom, property_phantom)
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    filenames = set(df['file'].tolist())
    expected_filenames = {
        PARENT_BLOB_FILE_NAME,
        CHILD_BLOB_FILE_NAME.format(i=1),
        CHILD_BLOB_FILE_NAME.format(i=2),
        TUBE_FILE_NAME.format(i=1)
    }
    assert filenames == expected_filenames


def test_mesh_writer_write_materials_csv_contains_correct_property_values(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=1, num_tubes=0)
    property_phantom = create_property_phantom(num_children=1, num_tubes=0)
    
    writer.write(mesh_phantom, property_phantom)
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    
    parent_row = df[df['file'] == PARENT_BLOB_FILE_NAME].iloc[0]
    assert parent_row['conductivity'] == property_phantom.parent.conductivity
    assert parent_row['permittivity'] == property_phantom.parent.permittivity
    assert parent_row['density'] == property_phantom.parent.density
    
    child_row = df[df['file'] == CHILD_BLOB_FILE_NAME.format(i=1)].iloc[0]
    assert child_row['conductivity'] == property_phantom.children[0].conductivity
    assert child_row['permittivity'] == property_phantom.children[0].permittivity
    assert child_row['density'] == property_phantom.children[0].density


def test_mesh_writer_write_with_empty_children_and_tubes(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=0, num_tubes=0)
    property_phantom = create_property_phantom(num_children=0, num_tubes=0)
    
    writer.write(mesh_phantom, property_phantom)
    
    assert (generation_output_dir_path / PARENT_BLOB_FILE_NAME).exists()
    assert (generation_output_dir_path / MATERIALS_FILE_NAME).exists()
    
    assert not (generation_output_dir_path / CHILD_BLOB_FILE_NAME.format(i=1)).exists()
    assert not (generation_output_dir_path / TUBE_FILE_NAME.format(i=1)).exists()
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    assert len(df) == 1
    assert df.iloc[0]['file'] == PARENT_BLOB_FILE_NAME


def test_mesh_writer_write_with_large_number_of_components(generation_output_dir_path, simple_mesh):
    num_children = 3
    num_tubes = 2
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=num_children, num_tubes=num_tubes)
    property_phantom = create_property_phantom(num_children=num_children, num_tubes=num_tubes)
    
    writer.write(mesh_phantom, property_phantom)
    
    assert (generation_output_dir_path / PARENT_BLOB_FILE_NAME).exists()
    for i in range(num_children):
        assert (generation_output_dir_path / CHILD_BLOB_FILE_NAME.format(i=i+1)).exists()
    for i in range(num_tubes):
        assert (generation_output_dir_path / TUBE_FILE_NAME.format(i=i+1)).exists()
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    assert len(df) == 1 + num_children + num_tubes


def test_mesh_writer_save_mesh_private_method_exports_stl_file(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh = simple_mesh
    prop = create_property_item()
    filename = "test_mesh.stl"
    
    result = writer._save_mesh(mesh, prop, filename)
    
    assert (generation_output_dir_path / filename).exists()
    
    expected_keys = {'conductivity', 'permittivity', 'density', 'file'}
    assert set(result.keys()) == expected_keys
    assert result['file'] == filename
    assert result['conductivity'] == prop.conductivity
    assert result['permittivity'] == prop.permittivity
    assert result['density'] == prop.density


def test_mesh_writer_save_mesh_private_method_preserves_original_property_values(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh = simple_mesh
    prop = create_property_item(conductivity=0.123, permittivity=45.67, density=890.0)
    filename = "test_mesh.stl"
    
    original_conductivity = prop.conductivity
    original_permittivity = prop.permittivity
    original_density = prop.density
    
    result = writer._save_mesh(mesh, prop, filename)
    
    assert prop.conductivity == original_conductivity
    assert prop.permittivity == original_permittivity
    assert prop.density == original_density
    assert not hasattr(prop, 'file')
    assert result['file'] == filename


def test_mesh_writer_write_handles_property_phantom_with_different_property_values(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=2, num_tubes=1)
    
    parent_prop = PropertyItem(conductivity=0.1, permittivity=10.0, density=100.0)
    child1_prop = PropertyItem(conductivity=0.2, permittivity=20.0, density=200.0)
    child2_prop = PropertyItem(conductivity=0.3, permittivity=30.0, density=300.0)
    tube_prop = PropertyItem(conductivity=0.4, permittivity=40.0, density=400.0)
    
    property_phantom = PropertyPhantom(
        parent=parent_prop,
        children=[child1_prop, child2_prop],
        tubes=[tube_prop]
    )
    
    writer.write(mesh_phantom, property_phantom)
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    
    parent_row = df[df['file'] == PARENT_BLOB_FILE_NAME].iloc[0]
    assert parent_row['conductivity'] == 0.1
    assert parent_row['permittivity'] == 10.0
    assert parent_row['density'] == 100.0
    
    child1_row = df[df['file'] == CHILD_BLOB_FILE_NAME.format(i=1)].iloc[0]
    assert child1_row['conductivity'] == 0.2
    assert child1_row['permittivity'] == 20.0
    assert child1_row['density'] == 200.0


def test_mesh_writer_write_overwrites_existing_files(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=1, num_tubes=0)
    property_phantom = create_property_phantom(num_children=1, num_tubes=0)
    
    writer.write(mesh_phantom, property_phantom)
    initial_parent_stat = (generation_output_dir_path / PARENT_BLOB_FILE_NAME).stat()
    initial_materials_stat = (generation_output_dir_path / MATERIALS_FILE_NAME).stat()
    
    new_property_phantom = PropertyPhantom(
        parent=PropertyItem(conductivity=999.0, permittivity=999.0, density=999.0),
        children=[PropertyItem(conductivity=888.0, permittivity=888.0, density=888.0)],
        tubes=[]
    )
    
    writer.write(mesh_phantom, new_property_phantom)
    
    new_materials_stat = (generation_output_dir_path / MATERIALS_FILE_NAME).stat()
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    parent_row = df[df['file'] == PARENT_BLOB_FILE_NAME].iloc[0]
    assert parent_row['conductivity'] == 999.0


def test_mesh_writer_write_handles_trimesh_export_error(generation_output_dir_path):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    
    invalid_mesh = Mock(spec=trimesh.Trimesh)
    invalid_mesh.export.side_effect = Exception("Export failed")
    
    mesh_phantom = MeshPhantom(parent=invalid_mesh, children=[], tubes=[])
    property_phantom = PropertyPhantom(parent=create_property_item(), children=[], tubes=[])
    
    with pytest.raises(Exception, match="Export failed"):
        writer.write(mesh_phantom, property_phantom)


def test_mesh_writer_write_creates_readable_stl_files(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=1, num_tubes=0)
    property_phantom = create_property_phantom(num_children=1, num_tubes=0)
    
    writer.write(mesh_phantom, property_phantom)
    
    parent_mesh = trimesh.load(generation_output_dir_path / PARENT_BLOB_FILE_NAME)
    assert isinstance(parent_mesh, trimesh.Trimesh)
    assert len(parent_mesh.vertices) > 0
    assert len(parent_mesh.faces) > 0
    
    child_mesh = trimesh.load(generation_output_dir_path / CHILD_BLOB_FILE_NAME.format(i=1))
    assert isinstance(child_mesh, trimesh.Trimesh)
    assert len(child_mesh.vertices) > 0
    assert len(child_mesh.faces) > 0


def test_mesh_writer_write_materials_csv_is_properly_formatted(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=1, num_tubes=1)
    property_phantom = create_property_phantom(num_children=1, num_tubes=1)
    
    writer.write(mesh_phantom, property_phantom)
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    
    assert pd.api.types.is_numeric_dtype(df['conductivity'])
    assert pd.api.types.is_numeric_dtype(df['permittivity'])
    assert pd.api.types.is_numeric_dtype(df['density'])
    
    assert pd.api.types.is_object_dtype(df['file'])
    
    assert not df.isnull().any().any()


def test_writer_initialization_with_none_directory_raises_error():
    with pytest.raises(TypeError):
        ConcreteWriter(output_dir=None)


def test_mesh_writer_write_with_mismatched_phantom_lengths(generation_output_dir_path, simple_mesh):
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=2, num_tubes=1)
    property_phantom = create_property_phantom(num_children=1, num_tubes=2)
    
    writer = MeshWriter(output_dir=generation_output_dir_path)
    
    writer.write(mesh_phantom, property_phantom)
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    expected_rows = 1 + min(2, 1) + min(1, 2)
    assert len(df) == expected_rows




def test_mesh_writer_private_save_mesh_method_preserves_original_property(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh = simple_mesh
    original_prop = create_property_item(conductivity=1.23, permittivity=45.6, density=789.0)
    
    result = writer._save_mesh(mesh, original_prop, "test.stl")
    
    assert original_prop.conductivity == 1.23
    assert original_prop.permittivity == 45.6
    assert original_prop.density == 789.0
    assert result['conductivity'] == 1.23
    assert result['permittivity'] == 45.6
    assert result['density'] == 789.0


def test_mesh_writer_write_with_zero_property_values(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=0, num_tubes=0)
    
    zero_prop = PropertyItem(conductivity=0.0, permittivity=0.0, density=0.0)
    property_phantom = PropertyPhantom(parent=zero_prop, children=[], tubes=[])
    
    writer.write(mesh_phantom, property_phantom)
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    assert df.iloc[0]['conductivity'] == 0.0
    assert df.iloc[0]['permittivity'] == 0.0
    assert df.iloc[0]['density'] == 0.0


def test_mesh_writer_write_with_negative_property_values(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=0, num_tubes=0)
    
    negative_prop = PropertyItem(conductivity=-1.0, permittivity=-2.0, density=-3.0)
    property_phantom = PropertyPhantom(parent=negative_prop, children=[], tubes=[])
    
    writer.write(mesh_phantom, property_phantom)
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    assert df.iloc[0]['conductivity'] == -1.0
    assert df.iloc[0]['permittivity'] == -2.0
    assert df.iloc[0]['density'] == -3.0


def test_mesh_writer_write_with_very_large_property_values(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=0, num_tubes=0)
    
    large_prop = PropertyItem(conductivity=1e10, permittivity=1e15, density=1e20)
    property_phantom = PropertyPhantom(parent=large_prop, children=[], tubes=[])
    
    writer.write(mesh_phantom, property_phantom)
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    assert df.iloc[0]['conductivity'] == 1e10
    assert df.iloc[0]['permittivity'] == 1e15
    assert df.iloc[0]['density'] == 1e20


def test_mesh_writer_filename_formatting_correctness():
    assert PARENT_BLOB_FILE_NAME == "parent.stl"
    assert CHILD_BLOB_FILE_NAME.format(i=1) == "child_blob_1.stl"
    assert CHILD_BLOB_FILE_NAME.format(i=10) == "child_blob_10.stl"
    assert TUBE_FILE_NAME.format(i=1) == "tube_1.stl"
    assert TUBE_FILE_NAME.format(i=5) == "tube_5.stl"
    assert MATERIALS_FILE_NAME == "materials.txt"


def test_writer_directory_path_handling_with_special_characters(generation_output_dir_path):
    special_dir = generation_output_dir_path / "special-dir_with.dots"
    writer = ConcreteWriter(output_dir=special_dir)
    
    assert writer.dir == special_dir
    assert writer.dir.exists()


def test_mesh_writer_write_preserves_mesh_geometry_in_export(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    original_mesh = simple_mesh
    original_vertices = original_mesh.vertices.copy()
    original_faces = original_mesh.faces.copy()
    
    mesh_phantom = MeshPhantom(parent=original_mesh, children=[], tubes=[])
    property_phantom = PropertyPhantom(parent=create_property_item(), children=[], tubes=[])
    
    writer.write(mesh_phantom, property_phantom)
    
    loaded_mesh = trimesh.load(generation_output_dir_path / PARENT_BLOB_FILE_NAME)
    assert np.allclose(loaded_mesh.vertices, original_vertices, atol=1e-6)
    assert np.array_equal(loaded_mesh.faces, original_faces)




def test_mesh_writer_write_with_corrupted_property_object(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=0, num_tubes=0)
    
    corrupted_prop = PropertyItem(conductivity=0.1, permittivity=10.0, density=100.0)
    del corrupted_prop.__dict__['conductivity']
    
    property_phantom = PropertyPhantom(parent=corrupted_prop, children=[], tubes=[])
    
    writer.write(mesh_phantom, property_phantom)
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    assert 'conductivity' not in df.columns or df.iloc[0].get('conductivity') != df.iloc[0].get('conductivity')


def test_mesh_writer_write_validates_mesh_export_success(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=1, num_tubes=0)
    property_phantom = create_property_phantom(num_children=1, num_tubes=0)
    
    writer.write(mesh_phantom, property_phantom)
    
    parent_file = generation_output_dir_path / PARENT_BLOB_FILE_NAME
    child_file = generation_output_dir_path / CHILD_BLOB_FILE_NAME.format(i=1)
    
    assert parent_file.exists()
    assert child_file.exists()
    assert parent_file.stat().st_size > 0
    assert child_file.stat().st_size > 0


def test_mesh_writer_write_property_isolation(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=1, num_tubes=0)
    
    original_prop = PropertyItem(conductivity=1.23, permittivity=45.6, density=789.0)
    property_phantom = PropertyPhantom(parent=original_prop, children=[original_prop], tubes=[])
    
    writer.write(mesh_phantom, property_phantom)
    
    assert not hasattr(original_prop, 'file')
    assert original_prop.conductivity == 1.23
    assert original_prop.permittivity == 45.6
    assert original_prop.density == 789.0





def test_mesh_writer_save_mesh_with_property_dict_manipulation(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh = simple_mesh
    prop = create_property_item()
    
    prop.__dict__['extra_field'] = 'test_value'
    
    result = writer._save_mesh(mesh, prop, "test.stl")
    
    assert result['extra_field'] == 'test_value'
    assert result['file'] == 'test.stl'


def test_mesh_writer_write_with_property_item_without_standard_attributes(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=0, num_tubes=0)
    
    minimal_prop = PropertyItem(conductivity=1.0, permittivity=2.0, density=3.0)
    delattr(minimal_prop, 'permittivity')
    
    property_phantom = PropertyPhantom(parent=minimal_prop, children=[], tubes=[])
    
    writer.write(mesh_phantom, property_phantom)
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    assert 'conductivity' in df.columns
    assert 'density' in df.columns
    assert 'permittivity' not in df.columns


def test_mesh_writer_write_with_dataframe_creation_edge_case(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh_phantom = create_mesh_phantom(simple_mesh, num_children=1, num_tubes=0)
    
    parent_prop = PropertyItem(conductivity=1.0, permittivity=2.0, density=3.0)
    child_prop = PropertyItem(conductivity=4.0, permittivity=5.0, density=6.0)
    child_prop.__dict__['extra_column'] = 'only_in_child'
    
    property_phantom = PropertyPhantom(parent=parent_prop, children=[child_prop], tubes=[])
    
    writer.write(mesh_phantom, property_phantom)
    
    df = pd.read_csv(generation_output_dir_path / MATERIALS_FILE_NAME)
    assert len(df) == 2
    assert df.iloc[1]['extra_column'] == 'only_in_child'
    assert pd.isna(df.iloc[0]['extra_column'])








def test_mesh_writer_save_mesh_with_existing_file_attribute_in_property(generation_output_dir_path, simple_mesh):
    writer = MeshWriter(output_dir=generation_output_dir_path)
    mesh = simple_mesh
    prop = create_property_item()
    prop.__dict__['file'] = 'old_filename.stl'
    
    result = writer._save_mesh(mesh, prop, "new_filename.stl")
    
    assert result['file'] == 'new_filename.stl'
    assert prop.file == 'old_filename.stl'
