import pytest
import numpy as np

from magnet_pinn.generator.structures import Structure3D, Blob, Tube


class ConcreteStructure(Structure3D):
    pass


def test_structure3d_initialization_with_valid_position_and_radius():
    position = np.array([1.0, 2.0, 3.0])
    radius = 5.0
    
    structure = ConcreteStructure(position=position, radius=radius)
    
    assert np.array_equal(structure.position, position)
    assert structure.radius == radius
    assert structure.position.dtype == float
    assert isinstance(structure.radius, float)


def test_structure3d_initialization_with_zero_position_and_minimal_radius():
    position = np.array([0.0, 0.0, 0.0])
    radius = np.finfo(float).eps
    
    structure = ConcreteStructure(position=position, radius=radius)
    assert np.array_equal(structure.position, position)
    assert structure.radius == radius


def test_structure3d_initialization_with_large_position_and_radius_values():
    position = np.array([1e6, -1e6, 1e6])
    radius = 1e6
    
    structure = ConcreteStructure(position=position, radius=radius)
    assert np.array_equal(structure.position, position)
    assert structure.radius == radius


def test_structure3d_converts_integer_position_and_radius_to_float():
    position = np.array([1, 2, 3])
    radius = 5
    
    structure = ConcreteStructure(position=position, radius=radius)
    assert structure.position.dtype == float
    assert isinstance(structure.radius, float)


def test_structure3d_rejects_wrong_shaped_position_array():
    with pytest.raises(ValueError, match="Position must be a 3D numpy array"):
        ConcreteStructure(position=np.array([1.0, 2.0]), radius=1.0)


def test_structure3d_rejects_wrong_dimensional_position_array():
    with pytest.raises(ValueError, match="Position must be a 3D numpy array"):
        ConcreteStructure(position=np.array([[1.0, 2.0, 3.0]]), radius=1.0)


def test_structure3d_rejects_non_numpy_array_position():
    with pytest.raises(ValueError, match="Position must be a 3D numpy array"):
        ConcreteStructure(position=[1.0, 2.0, 3.0], radius=1.0)


def test_structure3d_rejects_zero_radius():
    with pytest.raises(ValueError, match="Radius must be a positive number"):
        ConcreteStructure(position=np.array([1.0, 2.0, 3.0]), radius=0.0)


def test_structure3d_rejects_negative_radius():
    with pytest.raises(ValueError, match="Radius must be a positive number"):
        ConcreteStructure(position=np.array([1.0, 2.0, 3.0]), radius=-1.0)


def test_structure3d_rejects_slightly_negative_radius():
    with pytest.raises(ValueError, match="Radius must be a positive number"):
        ConcreteStructure(position=np.array([1.0, 2.0, 3.0]), radius=-1e-10)


def test_structure3d_rejects_non_numeric_radius():
    with pytest.raises(ValueError, match="Radius must be a positive number"):
        ConcreteStructure(position=np.array([1.0, 2.0, 3.0]), radius="invalid")


def test_blob_initialization_with_default_optional_parameters():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    blob = Blob(position=position, radius=radius)
    
    assert np.array_equal(blob.position, position)
    assert blob.radius == radius
    assert blob.relative_disruption_strength == 0.1
    assert hasattr(blob, 'empirical_max_offset')
    assert hasattr(blob, 'empirical_min_offset')
    assert hasattr(blob, 'noise')
    assert blob.perlin_scale == 0.4


def test_blob_initialization_with_custom_optional_parameters():
    position = np.array([1.0, 2.0, 3.0])
    radius = 2.5
    num_octaves = 5
    relative_disruption_strength = 0.2
    seed = 123
    perlin_scale = 0.5
    
    blob = Blob(
        position=position,
        radius=radius,
        num_octaves=num_octaves,
        relative_disruption_strength=relative_disruption_strength,
        seed=seed,
        perlin_scale=perlin_scale
    )
    
    assert np.array_equal(blob.position, position)
    assert blob.radius == radius
    assert blob.relative_disruption_strength == relative_disruption_strength
    assert blob.perlin_scale == perlin_scale


def test_blob_initialization_with_minimum_disruption_strength():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    disruption_strength = 1e-10
    
    blob = Blob(
        position=position,
        radius=radius,
        relative_disruption_strength=disruption_strength
    )
    
    assert blob.relative_disruption_strength == disruption_strength


def test_blob_initialization_with_large_disruption_strength():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    disruption_strength = 1.0
    
    blob = Blob(
        position=position,
        radius=radius,
        relative_disruption_strength=disruption_strength
    )
    
    assert blob.relative_disruption_strength == disruption_strength


def test_blob_initialization_with_minimum_octaves():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    blob = Blob(position=position, radius=radius, num_octaves=1)
    
    assert hasattr(blob, 'noise')
    assert hasattr(blob, 'empirical_max_offset')
    assert hasattr(blob, 'empirical_min_offset')


def test_blob_initialization_with_large_octaves():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    blob = Blob(position=position, radius=radius, num_octaves=20)
    
    assert hasattr(blob, 'noise')


def test_blob_initialization_with_zero_seed():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    blob = Blob(position=position, radius=radius, seed=0)
    
    assert hasattr(blob, 'noise')


def test_blob_initialization_with_negative_seed():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    blob = Blob(position=position, radius=radius, seed=-42)
    
    assert hasattr(blob, 'noise')


def test_blob_initialization_with_minimal_perlin_scale():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    blob = Blob(position=position, radius=radius, perlin_scale=1e-10)
    
    assert blob.perlin_scale == 1e-10


def test_blob_initialization_with_negative_perlin_scale():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    blob = Blob(position=position, radius=radius, perlin_scale=-0.5)
    
    assert blob.perlin_scale == -0.5


def test_blob_calls_fibonacci_points_generation_during_initialization():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    blob = Blob(position=position, radius=radius)
    
    # Verify that empirical offsets were calculated (which means fibonacci points were generated)
    assert hasattr(blob, 'empirical_max_offset')
    assert hasattr(blob, 'empirical_min_offset')
    assert isinstance(blob.empirical_max_offset, float)
    assert isinstance(blob.empirical_min_offset, float)
    
    # Test that the fibonacci method works directly
    points = blob._generate_fibonacci_points_on_sphere(num_points=100)
    assert isinstance(points, np.ndarray)
    assert points.shape == (100, 3)
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-10)


def test_blob_calculates_empirical_offsets_during_initialization():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    blob = Blob(position=position, radius=radius, seed=42)
    
    assert isinstance(blob.empirical_max_offset, (float, np.floating))
    assert isinstance(blob.empirical_min_offset, (float, np.floating))
    assert blob.empirical_max_offset >= blob.empirical_min_offset


def test_blob_fibonacci_points_generation_with_default_count():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    points = blob._generate_fibonacci_points_on_sphere(num_points=100)
    
    assert points.shape == (100, 3)
    norms = np.linalg.norm(points, axis=1)
    assert np.allclose(norms, 1.0, rtol=1e-10)


def test_blob_fibonacci_points_generation_with_custom_count():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    points = blob._generate_fibonacci_points_on_sphere(num_points=50)
    
    assert points.shape == (50, 3)
    norms = np.linalg.norm(points, axis=1)
    assert np.allclose(norms, 1.0, rtol=1e-10)


def test_blob_fibonacci_points_generation_fails_with_single_point():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    with pytest.raises(ZeroDivisionError):
        blob._generate_fibonacci_points_on_sphere(num_points=1)


def test_blob_fibonacci_points_have_uniform_distribution():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    points = blob._generate_fibonacci_points_on_sphere(num_points=1000)
    
    x_positive = np.sum(points[:, 0] > 0)
    x_negative = np.sum(points[:, 0] < 0)
    y_positive = np.sum(points[:, 1] > 0)
    y_negative = np.sum(points[:, 1] < 0)
    z_positive = np.sum(points[:, 2] > 0)
    z_negative = np.sum(points[:, 2] < 0)
    
    assert abs(x_positive - x_negative) < 200
    assert abs(y_positive - y_negative) < 200
    assert abs(z_positive - z_negative) < 200


def test_blob_calculate_offsets_for_single_vertex():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius, seed=42)
    
    vertices = np.array([[1.0, 0.0, 0.0]])
    offsets = blob.calculate_offsets(vertices)
    
    assert offsets.shape == (1, 1)
    assert isinstance(offsets[0, 0], (float, np.floating))


def test_blob_calculate_offsets_for_multiple_vertices():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius, seed=42)
    
    vertices = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0]
    ])
    offsets = blob.calculate_offsets(vertices)
    
    assert offsets.shape == (4, 1)
    assert all(isinstance(offset[0], (float, np.floating)) for offset in offsets)


def test_blob_calculate_offsets_scaling_correctness():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    disruption_strength = 0.2
    perlin_scale = 0.5
    
    blob = Blob(
        position=position,
        radius=radius,
        relative_disruption_strength=disruption_strength,
        perlin_scale=perlin_scale,
        seed=42
    )
    
    vertices = np.array([[1.0, 0.0, 0.0]])
    offsets = blob.calculate_offsets(vertices)
    
    noise_value = blob.noise([1.0, 0.0, 0.0])
    expected_offset = noise_value * disruption_strength / perlin_scale
    
    assert np.isclose(offsets[0, 0], expected_offset, rtol=1e-10)


def test_blob_calculate_offsets_reproducibility_with_same_seed():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    seed = 123
    
    blob1 = Blob(position=position, radius=radius, seed=seed)
    blob2 = Blob(position=position, radius=radius, seed=seed)
    
    vertices = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    offsets1 = blob1.calculate_offsets(vertices)
    offsets2 = blob2.calculate_offsets(vertices)
    
    assert np.array_equal(offsets1, offsets2)


def test_blob_calculate_offsets_different_results_with_different_seeds():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    blob1 = Blob(position=position, radius=radius, seed=42)
    blob2 = Blob(position=position, radius=radius, seed=123)
    
    vertices = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.5], [0.3, 0.7, 0.1], [0.8, 0.2, 0.6],
        [0.1, 0.9, 0.4], [0.6, 0.3, 0.8], [0.4, 0.6, 0.2]
    ])
    
    offsets1 = blob1.calculate_offsets(vertices)
    offsets2 = blob2.calculate_offsets(vertices)
    
    assert not np.array_equal(offsets1, offsets2)


def test_blob_calculate_offsets_for_empty_vertices_array():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    vertices = np.empty((0, 3))
    offsets = blob.calculate_offsets(vertices)
    
    assert offsets.shape == (0, 1)


def test_tube_initialization_with_default_height():
    position = np.array([1.0, 2.0, 3.0])
    direction = np.array([1.0, 0.0, 0.0])
    radius = 0.5
    
    tube = Tube(position=position, direction=direction, radius=radius)
    
    assert np.array_equal(tube.position, position)
    assert np.allclose(tube.direction, direction)
    assert tube.radius == radius
    assert tube.height == 10000


def test_tube_initialization_with_custom_height():
    position = np.array([1.0, 2.0, 3.0])
    direction = np.array([0.0, 1.0, 0.0])
    radius = 0.5
    height = 50.0
    
    tube = Tube(position=position, direction=direction, radius=radius, height=height)
    
    assert np.array_equal(tube.position, position)
    assert np.allclose(tube.direction, direction)
    assert tube.radius == radius
    assert tube.height == height


def test_tube_normalizes_non_unit_direction_vector():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([3.0, 4.0, 0.0])
    radius = 1.0
    
    tube = Tube(position=position, direction=direction, radius=radius)
    
    expected_direction = np.array([0.6, 0.8, 0.0])
    assert np.allclose(tube.direction, expected_direction)
    assert np.isclose(np.linalg.norm(tube.direction), 1.0)


def test_tube_preserves_unit_direction_vector():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    radius = 1.0
    
    tube = Tube(position=position, direction=direction, radius=radius)
    
    assert np.allclose(tube.direction, direction)
    assert np.isclose(np.linalg.norm(tube.direction), 1.0)


def test_tube_normalizes_very_small_direction_vector():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([1e-10, 0.0, 0.0])
    radius = 1.0
    
    tube = Tube(position=position, direction=direction, radius=radius)
    
    expected_direction = np.array([1.0, 0.0, 0.0])
    assert np.allclose(tube.direction, expected_direction)
    assert np.isclose(np.linalg.norm(tube.direction), 1.0)


def test_tube_initialization_with_zero_height():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    radius = 1.0
    height = 0.0
    
    tube = Tube(position=position, direction=direction, radius=radius, height=height)
    
    assert tube.height == 0.0


def test_tube_initialization_with_negative_height():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    radius = 1.0
    height = -100.0
    
    tube = Tube(position=position, direction=direction, radius=radius, height=height)
    
    assert tube.height == -100.0


def test_tube_initialization_with_very_large_height():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    radius = 1.0
    height = 1e9
    
    tube = Tube(position=position, direction=direction, radius=radius, height=height)
    
    assert tube.height == 1e9


def test_tube_distance_calculation_between_parallel_tubes():
    tube1 = Tube(
        position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        radius=1.0
    )
    tube2 = Tube(
        position=np.array([0.0, 3.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        radius=1.0
    )
    
    distance = Tube.distance_to_tube(tube1, tube2)
    
    assert np.isclose(distance, 3.0)


def test_tube_distance_calculation_between_perpendicular_tubes():
    tube1 = Tube(
        position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        radius=1.0
    )
    tube2 = Tube(
        position=np.array([1.0, 1.0, 0.0]),
        direction=np.array([0.0, 1.0, 0.0]),
        radius=1.0
    )
    
    distance = Tube.distance_to_tube(tube1, tube2)
    
    assert np.isclose(distance, 0.0)


def test_tube_distance_calculation_between_skew_tubes():
    tube1 = Tube(
        position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        radius=1.0
    )
    tube2 = Tube(
        position=np.array([0.0, 1.0, 1.0]),
        direction=np.array([0.0, 1.0, 0.0]),
        radius=1.0
    )
    
    distance = Tube.distance_to_tube(tube1, tube2)
    
    assert np.isclose(distance, 1.0)


def test_tube_distance_calculation_between_identical_tubes():
    tube1 = Tube(
        position=np.array([1.0, 2.0, 3.0]),
        direction=np.array([1.0, 1.0, 1.0]),
        radius=1.0
    )
    tube2 = Tube(
        position=np.array([1.0, 2.0, 3.0]),
        direction=np.array([1.0, 1.0, 1.0]),
        radius=1.0
    )
    
    distance = Tube.distance_to_tube(tube1, tube2)
    
    assert np.isclose(distance, 0.0)


def test_tube_distance_calculation_between_parallel_offset_tubes():
    tube1 = Tube(
        position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([0.0, 0.0, 1.0]),
        radius=1.0
    )
    tube2 = Tube(
        position=np.array([2.0, 3.0, 5.0]),
        direction=np.array([0.0, 0.0, 1.0]),
        radius=1.0
    )
    
    distance = Tube.distance_to_tube(tube1, tube2)
    
    expected_distance = np.sqrt(2**2 + 3**2)
    assert np.isclose(distance, expected_distance)


def test_tube_distance_calculation_when_cross_product_is_zero():
    tube1 = Tube(
        position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        radius=1.0
    )
    tube2 = Tube(
        position=np.array([5.0, 0.0, 0.0]),
        direction=np.array([2.0, 0.0, 0.0]),
        radius=1.0
    )
    
    distance = Tube.distance_to_tube(tube1, tube2)
    
    assert np.isclose(distance, 0.0)


def test_tube_initialization_with_zero_direction_vector_produces_nan():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
        tube = Tube(position=position, direction=direction, radius=radius)
        assert np.isnan(tube.direction).all()


def test_blob_fibonacci_points_generation_with_none_parameter():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    points = blob._generate_fibonacci_points_on_sphere(num_points=None)
    
    assert isinstance(points, np.ndarray)
    assert points.shape[1] == 3
    assert points.shape[0] == 10000
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-10)


def test_blob_fibonacci_points_generation_with_zero_points():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    points = blob._generate_fibonacci_points_on_sphere(num_points=0)
    
    assert isinstance(points, np.ndarray)
    assert points.shape == (0,)  # Empty array when range(0)


def test_blob_fibonacci_points_generation_with_two_points():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    points = blob._generate_fibonacci_points_on_sphere(num_points=2)
    
    assert isinstance(points, np.ndarray)
    assert points.shape == (2, 3)
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-10)
    expected_first_point = np.array([0.0, 1.0, 0.0])
    expected_second_point = np.array([0.0, -1.0, 0.0])
    assert np.allclose(points[0], expected_first_point, atol=1e-10)
    assert np.allclose(points[1], expected_second_point, atol=1e-10)


def test_blob_calculate_offsets_with_nan_vertices():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    vertices = np.array([[np.nan, 0.0, 0.0], [0.0, np.nan, 0.0]])
    
    with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
        blob.calculate_offsets(vertices)


def test_blob_calculate_offsets_with_inf_vertices():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    vertices = np.array([[np.inf, 0.0, 0.0], [0.0, -np.inf, 0.0]])
    
    with pytest.raises(OverflowError, match="cannot convert float infinity to integer"):
        blob.calculate_offsets(vertices)





def test_tube_distance_calculation_with_identical_position_and_direction():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    
    tube1 = Tube(position=position, direction=direction, radius=1.0)
    tube2 = Tube(position=position, direction=direction, radius=2.0)
    
    distance = Tube.distance_to_tube(tube1, tube2)
    assert distance == 0.0


def test_tube_distance_calculation_with_very_small_direction_vectors():
    tube1 = Tube(
        position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1e-15, 0.0, 0.0]),
        radius=1.0
    )
    tube2 = Tube(
        position=np.array([0.0, 1.0, 0.0]),
        direction=np.array([0.0, 1e-15, 0.0]),
        radius=1.0
    )
    
    distance = Tube.distance_to_tube(tube1, tube2)
    assert isinstance(distance, float)
    assert distance >= 0.0


def test_tube_direction_normalization_with_large_vector():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([1e6, 2e6, 3e6])
    
    tube = Tube(position=position, direction=direction, radius=1.0)
    
    assert np.allclose(np.linalg.norm(tube.direction), 1.0)
    expected_direction = direction / np.linalg.norm(direction)
    assert np.allclose(tube.direction, expected_direction)


def test_tube_initialization_with_negative_components_in_direction():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([-1.0, -2.0, -3.0])
    
    tube = Tube(position=position, direction=direction, radius=1.0)
    
    assert np.allclose(np.linalg.norm(tube.direction), 1.0)
    expected_direction = direction / np.linalg.norm(direction)
    assert np.allclose(tube.direction, expected_direction)


def test_blob_initialization_with_zero_relative_disruption_strength():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    blob = Blob(position=position, radius=radius, relative_disruption_strength=0.0)
    
    assert blob.relative_disruption_strength == 0.0
    assert hasattr(blob, 'empirical_max_offset')
    assert hasattr(blob, 'empirical_min_offset')


def test_blob_initialization_with_negative_octaves():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    with pytest.raises(ValueError):
        Blob(position=position, radius=radius, num_octaves=-1)


def test_blob_initialization_with_zero_octaves():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    with pytest.raises(ValueError):
        Blob(position=position, radius=radius, num_octaves=0)


def test_blob_initialization_with_very_large_perlin_scale():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    blob = Blob(position=position, radius=radius, perlin_scale=1e6)
    
    assert blob.perlin_scale == 1e6


def test_blob_initialization_with_zero_perlin_scale():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    with pytest.raises(ValueError, match="perlin_scale cannot be zero as it causes division by zero"):
        Blob(position=position, radius=radius, perlin_scale=0.0)


def test_tube_distance_calculation_between_antiparallel_tubes():
    tube1 = Tube(
        position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        radius=1.0
    )
    tube2 = Tube(
        position=np.array([0.0, 2.0, 0.0]),
        direction=np.array([-1.0, 0.0, 0.0]),
        radius=1.0
    )
    
    distance = Tube.distance_to_tube(tube1, tube2)
    assert np.isclose(distance, 2.0)


def test_tube_distance_calculation_with_very_close_parallel_tubes():
    tube1 = Tube(
        position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        radius=1.0
    )
    tube2 = Tube(
        position=np.array([0.0, 1e-10, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        radius=1.0
    )
    
    distance = Tube.distance_to_tube(tube1, tube2)
    assert np.isclose(distance, 1e-10, atol=1e-15)
