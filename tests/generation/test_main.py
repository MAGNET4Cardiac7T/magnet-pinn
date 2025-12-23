from unittest.mock import Mock, patch
from argparse import Namespace
import subprocess
import sys

import numpy as np

from magnet_pinn.generator.__main__ import (
    create_property_sampler,
    create_tissue_phantom,
    create_custom_phantom,
    create_workflow,
    generate_single_phantom,
    main,
)
from magnet_pinn.generator.samplers import PropertySampler
from magnet_pinn.generator.phantoms import Tissue, CustomPhantom
from magnet_pinn.generator.transforms import Compose


def test_create_property_sampler_with_default_values():
    args = Namespace(
        density_min=100.0,
        density_max=200.0,
        conductivity_min=0.5,
        conductivity_max=1.5,
        permittivity_min=10.0,
        permittivity_max=20.0,
    )

    sampler = create_property_sampler(args)

    assert isinstance(sampler, PropertySampler)
    assert sampler.properties_cfg["density"]["min"] == 100.0
    assert sampler.properties_cfg["density"]["max"] == 200.0
    assert sampler.properties_cfg["conductivity"]["min"] == 0.5
    assert sampler.properties_cfg["conductivity"]["max"] == 1.5
    assert sampler.properties_cfg["permittivity"]["min"] == 10.0
    assert sampler.properties_cfg["permittivity"]["max"] == 20.0


def test_create_property_sampler_with_custom_values():
    args = Namespace(
        density_min=500.0,
        density_max=1500.0,
        conductivity_min=0.1,
        conductivity_max=2.0,
        permittivity_min=1.0,
        permittivity_max=80.0,
    )

    sampler = create_property_sampler(args)

    assert isinstance(sampler, PropertySampler)
    assert sampler.properties_cfg["density"]["min"] == 500.0
    assert sampler.properties_cfg["density"]["max"] == 1500.0
    assert sampler.properties_cfg["conductivity"]["min"] == 0.1
    assert sampler.properties_cfg["conductivity"]["max"] == 2.0
    assert sampler.properties_cfg["permittivity"]["min"] == 1.0
    assert sampler.properties_cfg["permittivity"]["max"] == 80.0


def test_create_tissue_phantom_with_default_parameters():
    args = Namespace(
        num_children_blobs=5,
        initial_blob_radius=50.0,
        x_min=-10.0,
        x_max=10.0,
        y_min=-10.0,
        y_max=10.0,
        z_min=-10.0,
        z_max=10.0,
        blob_radius_decrease=0.3,
        num_tubes=8,
        relative_tube_max_radius=0.1,
        relative_tube_min_radius=0.01,
    )

    phantom = create_tissue_phantom(args)

    assert isinstance(phantom, Tissue)
    assert phantom.num_children_blobs == 5
    assert phantom.initial_blob_radius == 50.0
    assert phantom.num_tubes == 8
    assert np.array_equal(
        phantom.initial_blob_center_extent,
        np.array([[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]]),
    )


def test_create_tissue_phantom_with_custom_parameters():
    args = Namespace(
        num_children_blobs=10,
        initial_blob_radius=100.0,
        x_min=-5.0,
        x_max=5.0,
        y_min=-5.0,
        y_max=5.0,
        z_min=-50.0,
        z_max=50.0,
        blob_radius_decrease=0.5,
        num_tubes=15,
        relative_tube_max_radius=0.2,
        relative_tube_min_radius=0.05,
    )

    phantom = create_tissue_phantom(args)

    assert isinstance(phantom, Tissue)
    assert phantom.num_children_blobs == 10
    assert phantom.initial_blob_radius == 100.0
    assert phantom.num_tubes == 15


def test_create_custom_phantom_with_default_parameters(
    simple_mesh, generation_output_dir_path
):
    stl_path = generation_output_dir_path / "test_mesh.stl"
    simple_mesh.export(stl_path)

    args = Namespace(
        stl_mesh_path=stl_path,
        num_children_blobs=3,
        blob_radius_decrease=0.3,
        num_tubes=10,
        relative_tube_max_radius=0.1,
        relative_tube_min_radius=0.01,
        sample_children_only_inside=False,
    )

    phantom = create_custom_phantom(args)

    assert isinstance(phantom, CustomPhantom)
    assert phantom.num_children_blobs == 3
    assert phantom.num_tubes == 10


def test_create_custom_phantom_with_custom_parameters(
    simple_mesh, generation_output_dir_path
):
    stl_path = generation_output_dir_path / "custom_mesh.stl"
    simple_mesh.export(stl_path)

    args = Namespace(
        stl_mesh_path=stl_path,
        num_children_blobs=7,
        blob_radius_decrease=0.4,
        num_tubes=12,
        relative_tube_max_radius=0.15,
        relative_tube_min_radius=0.02,
        sample_children_only_inside=True,
    )

    phantom = create_custom_phantom(args)

    assert isinstance(phantom, CustomPhantom)
    assert phantom.num_children_blobs == 7
    assert phantom.num_tubes == 12


def test_create_workflow_mode_none():
    args = Namespace(transforms_mode="none")

    workflow = create_workflow(args)

    assert isinstance(workflow, Compose)
    assert len(workflow.transforms) == 0


def test_create_workflow_mode_default():
    args = Namespace(transforms_mode="default")

    workflow = create_workflow(args)

    assert isinstance(workflow, Compose)
    assert len(workflow.transforms) == 3


def test_create_workflow_mode_all():
    args = Namespace(transforms_mode="all")

    workflow = create_workflow(args)

    assert isinstance(workflow, Compose)
    assert len(workflow.transforms) == 5


@patch("magnet_pinn.generator.__main__.MeshWriter")
@patch("magnet_pinn.generator.__main__.default_rng")
@patch("magnet_pinn.generator.__main__.ToMesh")
def test_generate_single_phantom_tissue_type(
    mock_to_mesh, mock_rng, mock_writer, generation_output_dir_path
):
    mock_phantom_generator = Mock()
    mock_raw_structures = Mock()
    mock_raw_structures.children = [Mock(), Mock()]
    mock_raw_structures.tubes = [Mock()]
    mock_phantom_generator.generate.return_value = mock_raw_structures

    mock_phantom_meshes = Mock()
    mock_to_mesh_instance = Mock()
    mock_to_mesh_instance.return_value = mock_phantom_meshes
    mock_to_mesh.return_value = mock_to_mesh_instance

    mock_workflow = Mock()
    mock_processed_meshes = Mock()
    mock_processed_meshes.parent.vertices = np.array([[0, 0, 0]])
    mock_workflow.return_value = mock_processed_meshes

    mock_property_sampler = Mock()
    mock_properties = Mock()
    mock_property_sampler.sample_like.return_value = mock_properties

    mock_rng_instance = Mock()
    mock_rng.return_value = mock_rng_instance

    mock_writer_instance = Mock()
    mock_writer.return_value = mock_writer_instance

    args = Namespace(phantom_type="tissue")
    output_dir = generation_output_dir_path / "output"
    seed = 42

    result = generate_single_phantom(
        phantom_generator=mock_phantom_generator,
        property_sampler=mock_property_sampler,
        workflow=mock_workflow,
        args=args,
        output_dir=output_dir,
        seed=seed,
    )

    mock_phantom_generator.generate.assert_called_once_with(seed=seed)
    mock_to_mesh_instance.assert_called_once_with(mock_raw_structures)
    mock_workflow.assert_called_once_with(mock_phantom_meshes)
    mock_rng.assert_called_once_with(seed)
    mock_property_sampler.sample_like.assert_called_once_with(
        mock_processed_meshes, rng=mock_rng_instance
    )
    mock_writer.assert_called_once_with(output_dir)
    mock_writer_instance.write.assert_called_once_with(
        mock_processed_meshes, mock_properties
    )
    assert result == output_dir


@patch("magnet_pinn.generator.__main__.MeshWriter")
@patch("magnet_pinn.generator.__main__.default_rng")
@patch("magnet_pinn.generator.__main__.ToMesh")
def test_generate_single_phantom_custom_type(
    mock_to_mesh, mock_rng, mock_writer, generation_output_dir_path
):
    mock_phantom_generator = Mock()
    mock_raw_structures = Mock()
    mock_raw_structures.children = [Mock()]
    mock_raw_structures.tubes = [Mock(), Mock()]
    mock_phantom_generator.generate.return_value = mock_raw_structures

    mock_phantom_meshes = Mock()
    mock_to_mesh_instance = Mock()
    mock_to_mesh_instance.return_value = mock_phantom_meshes
    mock_to_mesh.return_value = mock_to_mesh_instance

    mock_workflow = Mock()
    mock_processed_meshes = Mock()
    mock_processed_meshes.parent.vertices = np.array([[1, 2, 3], [4, 5, 6]])
    mock_workflow.return_value = mock_processed_meshes

    mock_property_sampler = Mock()
    mock_properties = Mock()
    mock_property_sampler.sample_like.return_value = mock_properties

    mock_rng_instance = Mock()
    mock_rng.return_value = mock_rng_instance

    mock_writer_instance = Mock()
    mock_writer.return_value = mock_writer_instance

    args = Namespace(phantom_type="custom", child_blobs_batch_size=500000)
    output_dir = generation_output_dir_path / "custom_output"
    seed = 123

    result = generate_single_phantom(
        phantom_generator=mock_phantom_generator,
        property_sampler=mock_property_sampler,
        workflow=mock_workflow,
        args=args,
        output_dir=output_dir,
        seed=seed,
    )

    mock_phantom_generator.generate.assert_called_once_with(
        seed=seed, child_blobs_batch_size=500000
    )
    mock_to_mesh_instance.assert_called_once_with(mock_raw_structures)
    mock_workflow.assert_called_once_with(mock_phantom_meshes)
    mock_rng.assert_called_once_with(seed)
    mock_property_sampler.sample_like.assert_called_once_with(
        mock_processed_meshes, rng=mock_rng_instance
    )
    mock_writer.assert_called_once_with(output_dir)
    mock_writer_instance.write.assert_called_once_with(
        mock_processed_meshes, mock_properties
    )
    assert result == output_dir


@patch("magnet_pinn.generator.__main__.print_report")
@patch("magnet_pinn.generator.__main__.generate_single_phantom")
@patch("magnet_pinn.generator.__main__.create_workflow")
@patch("magnet_pinn.generator.__main__.create_property_sampler")
@patch("magnet_pinn.generator.__main__.create_tissue_phantom")
@patch("magnet_pinn.generator.__main__.validate_arguments")
@patch("magnet_pinn.generator.__main__.parse_arguments")
def test_main_tissue_phantom_success(
    mock_parse_args,
    mock_validate,
    mock_create_tissue,
    mock_create_sampler,
    mock_create_workflow,
    mock_generate,
    mock_print_report,
    generation_output_dir_path,
):
    args = Namespace(phantom_type="tissue", output=generation_output_dir_path, seed=42)
    mock_parse_args.return_value = args

    mock_phantom = Mock()
    mock_create_tissue.return_value = mock_phantom

    mock_sampler = Mock()
    mock_create_sampler.return_value = mock_sampler

    mock_workflow = Mock()
    mock_create_workflow.return_value = mock_workflow

    mock_output_path = generation_output_dir_path / "result"
    mock_generate.return_value = mock_output_path

    result = main()

    assert result == 0
    mock_parse_args.assert_called_once()
    mock_validate.assert_called_once_with(args)
    mock_create_tissue.assert_called_once_with(args)
    mock_create_sampler.assert_called_once_with(args)
    mock_create_workflow.assert_called_once_with(args)
    mock_generate.assert_called_once_with(
        phantom_generator=mock_phantom,
        property_sampler=mock_sampler,
        workflow=mock_workflow,
        args=args,
        output_dir=args.output,
        seed=args.seed,
    )
    mock_print_report.assert_called_once_with(args, mock_output_path)


@patch("magnet_pinn.generator.__main__.print_report")
@patch("magnet_pinn.generator.__main__.generate_single_phantom")
@patch("magnet_pinn.generator.__main__.create_workflow")
@patch("magnet_pinn.generator.__main__.create_property_sampler")
@patch("magnet_pinn.generator.__main__.create_custom_phantom")
@patch("magnet_pinn.generator.__main__.validate_arguments")
@patch("magnet_pinn.generator.__main__.parse_arguments")
def test_main_custom_phantom_success(
    mock_parse_args,
    mock_validate,
    mock_create_custom,
    mock_create_sampler,
    mock_create_workflow,
    mock_generate,
    mock_print_report,
    generation_output_dir_path,
):
    args = Namespace(phantom_type="custom", output=generation_output_dir_path, seed=100)
    mock_parse_args.return_value = args

    mock_phantom = Mock()
    mock_create_custom.return_value = mock_phantom

    mock_sampler = Mock()
    mock_create_sampler.return_value = mock_sampler

    mock_workflow = Mock()
    mock_create_workflow.return_value = mock_workflow

    mock_output_path = generation_output_dir_path / "custom_result"
    mock_generate.return_value = mock_output_path

    result = main()

    assert result == 0
    mock_parse_args.assert_called_once()
    mock_validate.assert_called_once_with(args)
    mock_create_custom.assert_called_once_with(args)
    mock_create_sampler.assert_called_once_with(args)
    mock_create_workflow.assert_called_once_with(args)
    mock_generate.assert_called_once_with(
        phantom_generator=mock_phantom,
        property_sampler=mock_sampler,
        workflow=mock_workflow,
        args=args,
        output_dir=args.output,
        seed=args.seed,
    )
    mock_print_report.assert_called_once_with(args, mock_output_path)


@patch("magnet_pinn.generator.__main__.validate_arguments")
@patch("magnet_pinn.generator.__main__.parse_arguments")
def test_main_validation_error(mock_parse_args, mock_validate):
    args = Namespace(phantom_type="tissue")
    mock_parse_args.return_value = args
    mock_validate.side_effect = ValueError("Invalid configuration")

    result = main()

    assert result == 1
    mock_parse_args.assert_called_once()
    mock_validate.assert_called_once_with(args)


@patch("magnet_pinn.generator.__main__.create_tissue_phantom")
@patch("magnet_pinn.generator.__main__.validate_arguments")
@patch("magnet_pinn.generator.__main__.parse_arguments")
def test_main_unknown_phantom_type(mock_parse_args, mock_validate, mock_create_tissue):
    args = Namespace(phantom_type="unknown_type")
    mock_parse_args.return_value = args

    result = main()

    assert result == 1
    mock_parse_args.assert_called_once()
    mock_validate.assert_called_once_with(args)
    mock_create_tissue.assert_not_called()


@patch("magnet_pinn.generator.__main__.generate_single_phantom")
@patch("magnet_pinn.generator.__main__.create_workflow")
@patch("magnet_pinn.generator.__main__.create_property_sampler")
@patch("magnet_pinn.generator.__main__.create_tissue_phantom")
@patch("magnet_pinn.generator.__main__.validate_arguments")
@patch("magnet_pinn.generator.__main__.parse_arguments")
def test_main_generation_error(
    mock_parse_args,
    mock_validate,
    mock_create_tissue,
    mock_create_sampler,
    mock_create_workflow,
    mock_generate,
    generation_output_dir_path,
):
    args = Namespace(phantom_type="tissue", output=generation_output_dir_path, seed=42)
    mock_parse_args.return_value = args

    mock_phantom = Mock()
    mock_create_tissue.return_value = mock_phantom

    mock_sampler = Mock()
    mock_create_sampler.return_value = mock_sampler

    mock_workflow = Mock()
    mock_create_workflow.return_value = mock_workflow

    mock_generate.side_effect = RuntimeError("Generation failed")

    result = main()

    assert result == 1
    mock_parse_args.assert_called_once()
    mock_validate.assert_called_once_with(args)
    mock_create_tissue.assert_called_once_with(args)
    mock_create_sampler.assert_called_once_with(args)
    mock_create_workflow.assert_called_once_with(args)
    mock_generate.assert_called_once()


def test_main_module_direct_execution():
    result = subprocess.run(
        [sys.executable, "-m", "magnet_pinn.generator"], capture_output=True, timeout=5
    )

    assert result.returncode == 2
