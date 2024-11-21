import pytest
import numpy as np
from math import pi

from magnet_pinn.preprocessing.voxelizing_mesh import MeshVoxelizer


def test_unit_sphere_mesh_fills_grid(sphere_unit_mesh):
    """
    We expect a spehere mesh with radius equal to the voxel size. The mesh should fill the grid.
    """
    center = sphere_unit_mesh.center
    left_bound = sphere_unit_mesh.bounds[0]
    right_bound = sphere_unit_mesh.bounds[1]
    left_x_bound = left_bound[0]
    right_x_bound = right_bound[0]

    voxel_size = radius = np.abs(left_x_bound - center[0])
    steps = ((right_x_bound - left_x_bound) / voxel_size).astype(int) + 1
    grid_x = np.linspace(left_x_bound, right_x_bound, steps)
    grid_y = np.linspace(left_bound[1], right_bound[1], steps)
    grid_z = np.linspace(left_bound[2], right_bound[2], steps)

    voxelizer = MeshVoxelizer(voxel_size, grid_x, grid_y, grid_z)
    result = voxelizer.process_mesh(sphere_unit_mesh)
    supposed_voxels = np.ones((steps, steps, steps))
    
    assert 4/3 * pi * np.power((radius + np.sqrt(3) * voxel_size), 3) > np.sum(np.power(result, 3)) > 4/3 * pi * np.power((radius - np.sqrt(3) * voxel_size), 3)
    assert result.shape == (steps, steps, steps)
    assert np.sum(result) == steps ** 3
    assert np.equal(result, supposed_voxels).all()


def test_unit_sphere_mesh_fills_center_of_grid(sphere_unit_mesh):
    """
    We use a sphere mesh with radius equal to the voxel size but 
    this time the grid size is increased by 1 voxel.
    """
    center = sphere_unit_mesh.center
    left_bound = sphere_unit_mesh.bounds[0]
    right_bound = sphere_unit_mesh.bounds[1]

    voxel_size = radius = np.abs(left_bound[0] - center[0])
    left_bound -= voxel_size
    right_bound += voxel_size
    steps = ((right_bound[0] - left_bound[0]) / voxel_size).astype(int) + 1
    grid_x = np.linspace(left_bound[0], right_bound[0], steps)
    grid_y = np.linspace(left_bound[1], right_bound[1], steps)
    grid_z = np.linspace(left_bound[2], right_bound[2], steps)

    voxelizer = MeshVoxelizer(voxel_size, grid_x, grid_y, grid_y)
    result = voxelizer.process_mesh(sphere_unit_mesh)
    supposed_voxels = np.zeros((steps, steps, steps))
    supposed_voxels[1:-1, 1:-1, 1:-1] = 1

    assert 4/3 * pi * np.power((radius + np.sqrt(3) * voxel_size), 3) > np.sum(np.power(result, 3)) > 4/3 * pi * np.power((radius - np.sqrt(3) * voxel_size), 3)
    assert result.shape == (steps, steps, steps)
    assert np.sum(result) == (steps - 2) ** 3
    assert np.equal(result, supposed_voxels).all()


def test_unit_sphere_mesh_grid_is_smaller(sphere_unit_mesh):
    """
    We use a sphere primitive mesh. In this test case we would use a 
    voxel size of 0.5 of a raduis and grid size of 3.
    """
    center = sphere_unit_mesh.center
    left_x_bound = sphere_unit_mesh.bounds[0][0]
    radius = np.abs(left_x_bound - center[0])

    voxel_size = 0.5 * radius
    steps = 3
    left_bound = center - voxel_size
    right_bound = center + voxel_size
    grid_x = np.linspace(left_bound[0], right_bound[0], steps)
    grid_y = np.linspace(left_bound[1], right_bound[1], steps)
    grid_z = np.linspace(left_bound[2], right_bound[2], steps)

    voxelizer = MeshVoxelizer(voxel_size, grid_x, grid_y, grid_z)
    result = voxelizer.process_mesh(sphere_unit_mesh)
    supposed_voxels = np.ones((steps, steps, steps))

    assert 4/3 * pi * np.power((radius + np.sqrt(3) * voxel_size), 3) > np.sum(np.power(result, 3)) > 4/3 * pi * np.power((radius - np.sqrt(3) * voxel_size), 3)
    assert result.shape == (steps, steps, steps)
    assert np.sum(result) == steps ** 3
    assert np.equal(result, supposed_voxels).all()


def test_unit_sphere_mesh_grid_includes_object_in_the_center(sphere_unit_mesh):
    """
    We use a sphere primitive mesh. In this test case the voxel grid is a sprehe diameter. 
    We take grid of size 3 so the only existing voxel is the center one.
    """
    center = sphere_unit_mesh.center
    left_x_bound = sphere_unit_mesh.bounds[0][0]
    radius = np.abs(left_x_bound - center[0])

    voxel_size = 2 * radius
    steps = 3
    left_bound = center - voxel_size
    right_bound = center + voxel_size
    grid_x = np.linspace(left_bound[0], right_bound[0], steps)
    grid_y = np.linspace(left_bound[1], right_bound[1], steps)
    grid_z = np.linspace(left_bound[2], right_bound[2], steps)

    voxelizer = MeshVoxelizer(voxel_size, grid_x, grid_y, grid_z)
    result = voxelizer.process_mesh(sphere_unit_mesh)
    supposed_voxels = np.zeros((steps, steps, steps))
    supposed_voxels[steps // 2, steps // 2, steps // 2] = 1

    assert 4/3 * pi * np.power((radius + np.sqrt(3) * voxel_size), 3) > np.sum(np.power(result, 3)) > 4/3 * pi * np.power((radius - np.sqrt(3) * voxel_size), 3)
    assert result.shape == (steps, steps, steps)
    assert np.sum(result) == 1
    assert np.equal(result, supposed_voxels).all()


def test_invalid_grid_with_broken_spacing():
    """
    Checks if a mesh voxelizer class is able to detect invalid grid.
    """
    valid_grid = np.linspace(0, 5, 6)
    invalid_grid = np.linspace(0, 5, 8)
    with pytest.raises(ValueError):
        MeshVoxelizer(1, invalid_grid, valid_grid, valid_grid)
        MeshVoxelizer(1, valid_grid, invalid_grid, valid_grid)
        MeshVoxelizer(1, valid_grid, valid_grid, invalid_grid)


def test_invalid_grid_with_broken_bounds():
    """
    Check if a mesh voxelizer class is able to detect unsorted or equal grid values.
    """
    valid_grid = np.linspace(0, 5, 6)
    invalid_equal_grid = np.array(6 * [0])
    invalid_unsorted_grid = np.random.shuffle(valid_grid)

    with pytest.raises(ValueError):
        MeshVoxelizer(1, invalid_equal_grid, valid_grid, valid_grid)
        MeshVoxelizer(1, valid_grid, invalid_equal_grid, valid_grid)
        MeshVoxelizer(1, valid_grid, valid_grid, invalid_equal_grid)
        MeshVoxelizer(1, invalid_unsorted_grid, valid_grid, valid_grid)
        MeshVoxelizer(1, valid_grid, invalid_unsorted_grid, valid_grid)
        MeshVoxelizer(1, valid_grid, valid_grid, invalid_unsorted_grid)


def test_unit_box_mesh_fills_grid(box_unit_mesh):
    """
    We used a squared box mesh. The voxel size is exactly the 
    same as half of the box side. The grid would have 3 voxels in a row.
    So the as a result we will have the whole grid filled.
    """
    bounds = np.array(box_unit_mesh.bounds)
    voxel_size = np.abs(bounds[1][0] - bounds[0][0]) / 2
    steps = 3
    grid_x = np.linspace(bounds[0][0], bounds[1][0], steps)
    grid_y = np.linspace(bounds[0][1], bounds[1][1], steps)
    grid_z = np.linspace(bounds[0][2], bounds[1][2], steps)

    voxelizer = MeshVoxelizer(voxel_size, grid_x, grid_y, grid_z)
    result = voxelizer.process_mesh(box_unit_mesh)
    supposed_voxels = np.ones((steps, steps, steps))

    assert np.sum(result) == steps ** 3
    assert result.shape == (steps, steps, steps)
    assert np.equal(result, supposed_voxels).all()


def test_unit_box_mesh_fills_center_of_grid(box_unit_mesh):
    """
    We used a squared box mesh. The voxel size is exactly the same as 
    the half box size. Tre grid would have 5 voxels in a row. So the as a
    result we will have 3 * 3 * 3 = 27 voxels filled in the center.
    """
    bounds = np.array(box_unit_mesh.bounds)
    left_bound = bounds[0]
    right_bound = bounds[1]
    voxel_size = np.abs(bounds[1][0] - bounds[0][0]) / 2
    steps = 5
    left_bound -= voxel_size
    right_bound += voxel_size
    grid_x = np.linspace(left_bound[0], right_bound[0], steps)
    grid_y = np.linspace(left_bound[1], right_bound[1], steps)
    grid_z = np.linspace(left_bound[2], right_bound[2], steps)

    voxelizer = MeshVoxelizer(voxel_size, grid_x, grid_y, grid_z)
    result = voxelizer.process_mesh(box_unit_mesh)
    supposed_voxels = np.zeros((steps, steps, steps))
    supposed_voxels[1:-1, 1:-1, 1:-1] = 1

    assert np.sum(result) == (steps - 2) ** 3
    assert result.shape == (steps, steps, steps)
    assert np.equal(result, supposed_voxels).all()


def test_unit_box_mesh_grid_is_smaller(box_unit_mesh):
    """
    We used a squared box mesh. The voxel size is 0.75 of the half of the 
    box side. The grid will be just 3 voxels in a row. So the as a result 
    we will have the whole grid filled.
    """
    bounds = np.array(box_unit_mesh.bounds)
    voxel_size = 0.75 * np.abs(bounds[1][0] - bounds[0][0]) / 2
    steps = 3
    left_bound = box_unit_mesh.center_mass - voxel_size
    right_bound = box_unit_mesh.center_mass + voxel_size
    grid_x = np.linspace(left_bound[0], right_bound[0], steps)
    grid_y = np.linspace(left_bound[1], right_bound[1], steps)
    grid_z = np.linspace(left_bound[2], right_bound[2], steps)

    voxelizer = MeshVoxelizer(voxel_size, grid_x, grid_y, grid_z)
    result = voxelizer.process_mesh(box_unit_mesh)
    supposed_voxels = np.ones((steps, steps, steps))

    assert np.sum(result) == steps ** 3
    assert result.shape == (steps, steps, steps)
    assert np.equal(result, supposed_voxels).all()


def test_unit_box_mesh_grid_is_bigger(box_unit_mesh):
    """
    We used a squared box mesh. The voxel size is exactly the same as the
    box side. And the grid will have just 3 voxels in a row. So the as a 
    result we will have only one voxel in the center of the grid.
    """
    bounds = np.array(box_unit_mesh.bounds)
    voxel_size = np.abs(bounds[1][0] - bounds[0][0])
    steps = 3
    left_bound = box_unit_mesh.center_mass - voxel_size
    right_bound = box_unit_mesh.center_mass + voxel_size
    grid_x = np.linspace(left_bound[0], right_bound[0], steps)
    grid_y = np.linspace(left_bound[1], right_bound[1], steps)
    grid_z = np.linspace(left_bound[2], right_bound[2], steps)

    voxelizer = MeshVoxelizer(voxel_size, grid_x, grid_y, grid_z)
    result = voxelizer.process_mesh(box_unit_mesh)
    supposed_voxels = np.zeros((steps, steps, steps))
    supposed_voxels[1, 1, 1] = 1

    assert np.sum(result) == 1
    assert result.shape == (steps, steps, steps)
    assert np.equal(result, supposed_voxels).all()
