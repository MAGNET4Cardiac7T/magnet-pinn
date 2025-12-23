import numpy as np
import pytest

from magnet_pinn.generator.utils import (
    spheres_packable,
    generate_fibonacci_points_on_sphere,
)


def test_spheres_packable_single_sphere_fits_exactly():
    assert spheres_packable(
        radius_outer=1.0, radius_inner=1.0, num_inner=1, safety_margin=0.0
    )


def test_spheres_packable_single_sphere_fits_with_room():
    assert spheres_packable(radius_outer=2.0, radius_inner=1.0, num_inner=1)


def test_spheres_packable_single_sphere_too_large():
    assert not spheres_packable(radius_outer=1.0, radius_inner=1.5, num_inner=1)


def test_spheres_packable_single_sphere_with_safety_margin():
    assert not spheres_packable(
        radius_outer=1.0, radius_inner=1.0, num_inner=1, safety_margin=0.1
    )


def test_spheres_packable_two_spheres_optimal_configuration():
    assert spheres_packable(
        radius_outer=1.0, radius_inner=0.5, num_inner=2, safety_margin=0.0
    )


def test_spheres_packable_two_spheres_barely_fit():
    assert spheres_packable(radius_outer=1.0, radius_inner=0.49, num_inner=2)


def test_spheres_packable_two_spheres_too_large():
    assert not spheres_packable(radius_outer=1.0, radius_inner=0.6, num_inner=2)


def test_spheres_packable_three_spheres_optimal_configuration():
    critical_ratio = 2 * np.sqrt(3) - 3
    assert spheres_packable(
        radius_outer=1.0,
        radius_inner=critical_ratio * 0.95,
        num_inner=3,
        safety_margin=0.0,
    )


def test_spheres_packable_three_spheres_exceeds_limit():
    critical_ratio = 2 * np.sqrt(3) - 3
    assert not spheres_packable(
        radius_outer=1.0,
        radius_inner=critical_ratio * 1.05,
        num_inner=3,
        safety_margin=0.0,
    )


def test_spheres_packable_four_spheres_optimal_configuration():
    critical_ratio = np.sqrt(6) - 2
    assert spheres_packable(
        radius_outer=1.0,
        radius_inner=critical_ratio * 0.95,
        num_inner=4,
        safety_margin=0.0,
    )


def test_spheres_packable_four_spheres_exceeds_limit():
    critical_ratio = np.sqrt(6) - 2
    assert not spheres_packable(
        radius_outer=1.0,
        radius_inner=critical_ratio * 1.05,
        num_inner=4,
        safety_margin=0.0,
    )


def test_spheres_packable_five_spheres_optimal_configuration():
    critical_ratio = np.sqrt(2) - 1
    assert spheres_packable(
        radius_outer=1.0,
        radius_inner=critical_ratio * 0.95,
        num_inner=5,
        safety_margin=0.0,
    )


def test_spheres_packable_five_spheres_exceeds_limit():
    critical_ratio = np.sqrt(2) - 1
    assert not spheres_packable(
        radius_outer=1.0,
        radius_inner=critical_ratio * 1.05,
        num_inner=5,
        safety_margin=0.0,
    )


def test_spheres_packable_six_spheres_optimal_configuration():
    critical_ratio = np.sqrt(2) - 1
    assert spheres_packable(
        radius_outer=1.0,
        radius_inner=critical_ratio * 0.95,
        num_inner=6,
        safety_margin=0.0,
    )


def test_spheres_packable_six_spheres_exceeds_limit():
    critical_ratio = np.sqrt(2) - 1
    assert not spheres_packable(
        radius_outer=1.0,
        radius_inner=critical_ratio * 1.05,
        num_inner=6,
        safety_margin=0.0,
    )


def test_spheres_packable_seven_spheres_always_false():
    assert not spheres_packable(radius_outer=10.0, radius_inner=0.1, num_inner=7)


def test_spheres_packable_zero_spheres_always_false():
    assert not spheres_packable(radius_outer=1.0, radius_inner=0.1, num_inner=0)


def test_spheres_packable_negative_spheres_always_false():
    assert not spheres_packable(radius_outer=1.0, radius_inner=0.1, num_inner=-1)


def test_spheres_packable_with_default_safety_margin():
    assert spheres_packable(radius_outer=2.0, radius_inner=1.0, num_inner=1)


def test_spheres_packable_with_large_safety_margin():
    assert not spheres_packable(
        radius_outer=1.0, radius_inner=0.9, num_inner=1, safety_margin=0.5
    )


def test_spheres_packable_with_zero_safety_margin():
    assert spheres_packable(
        radius_outer=1.0, radius_inner=1.0, num_inner=1, safety_margin=0.0
    )


def test_spheres_packable_with_negative_safety_margin():
    assert spheres_packable(
        radius_outer=1.0, radius_inner=1.1, num_inner=1, safety_margin=-0.1
    )


def test_spheres_packable_very_small_radii():
    assert spheres_packable(radius_outer=1e-6, radius_inner=5e-7, num_inner=1)


def test_spheres_packable_very_large_radii():
    assert spheres_packable(radius_outer=1e6, radius_inner=5e5, num_inner=1)


def test_spheres_packable_zero_outer_radius():
    assert not spheres_packable(radius_outer=0.0, radius_inner=0.1, num_inner=1)


def test_spheres_packable_zero_inner_radius():
    assert spheres_packable(radius_outer=1.0, radius_inner=0.0, num_inner=1)


def test_spheres_packable_equal_radii_single_sphere():
    assert spheres_packable(
        radius_outer=1.0, radius_inner=1.0, num_inner=1, safety_margin=0.0
    )


def test_spheres_packable_mathematical_precision_three_spheres():
    critical_ratio = 2 * np.sqrt(3) - 3
    outer_radius = 10.0
    inner_radius = critical_ratio * outer_radius
    assert spheres_packable(
        radius_outer=outer_radius,
        radius_inner=inner_radius,
        num_inner=3,
        safety_margin=0.0,
    )


def test_spheres_packable_mathematical_precision_four_spheres():
    critical_ratio = np.sqrt(6) - 2
    outer_radius = 5.0
    inner_radius = critical_ratio * outer_radius
    assert spheres_packable(
        radius_outer=outer_radius,
        radius_inner=inner_radius,
        num_inner=4,
        safety_margin=0.0,
    )


def test_spheres_packable_boundary_case_two_spheres():
    assert spheres_packable(
        radius_outer=2.0, radius_inner=1.0, num_inner=2, safety_margin=0.0
    )
    assert not spheres_packable(
        radius_outer=2.0, radius_inner=1.001, num_inner=2, safety_margin=0.0
    )


def test_spheres_packable_scaling_invariance():
    scale_factor = 100.0
    assert spheres_packable(
        radius_outer=1.0, radius_inner=0.4, num_inner=2
    ) == spheres_packable(
        radius_outer=scale_factor, radius_inner=0.4 * scale_factor, num_inner=2
    )


def test_spheres_packable_consistency_five_and_six_spheres():
    critical_ratio = np.sqrt(2) - 1
    outer_radius = 3.0
    inner_radius = critical_ratio * outer_radius * 0.9

    result_5 = spheres_packable(
        radius_outer=outer_radius,
        radius_inner=inner_radius,
        num_inner=5,
        safety_margin=0.0,
    )
    result_6 = spheres_packable(
        radius_outer=outer_radius,
        radius_inner=inner_radius,
        num_inner=6,
        safety_margin=0.0,
    )

    assert result_5 == result_6


def test_spheres_packable_edge_case_tiny_inner_spheres():
    assert spheres_packable(radius_outer=1.0, radius_inner=1e-10, num_inner=6)


def test_spheres_packable_safety_margin_effect_on_boundary():
    critical_ratio = np.sqrt(6) - 2
    outer_radius = 1.0
    inner_radius = critical_ratio * outer_radius

    assert spheres_packable(
        radius_outer=outer_radius,
        radius_inner=inner_radius,
        num_inner=4,
        safety_margin=0.0,
    )
    assert not spheres_packable(
        radius_outer=outer_radius,
        radius_inner=inner_radius,
        num_inner=4,
        safety_margin=0.01,
    )


def test_spheres_packable_fractional_safety_margin():
    assert not spheres_packable(
        radius_outer=1.0, radius_inner=0.99, num_inner=1, safety_margin=0.02
    )


def test_spheres_packable_multiple_scenarios_consistency():
    test_cases = [
        (1.0, 0.3, 1, True),
        (1.0, 0.3, 2, True),
        (1.0, 0.3, 3, True),
        (2.0, 0.4, 4, True),
        (1.0, 0.5, 4, False),
    ]

    for outer, inner, num, expected in test_cases:
        result = spheres_packable(radius_outer=outer, radius_inner=inner, num_inner=num)
        assert result == expected


def test_spheres_packable_exact_mathematical_boundary_two_spheres():
    assert spheres_packable(
        radius_outer=1.0, radius_inner=0.5, num_inner=2, safety_margin=0.0
    )


def test_spheres_packable_exact_mathematical_boundary_three_spheres():
    critical_ratio = 2 * np.sqrt(3) - 3
    assert spheres_packable(
        radius_outer=1.0, radius_inner=critical_ratio, num_inner=3, safety_margin=0.0
    )


def test_spheres_packable_exact_mathematical_boundary_four_spheres():
    critical_ratio = np.sqrt(6) - 2
    assert spheres_packable(
        radius_outer=1.0, radius_inner=critical_ratio, num_inner=4, safety_margin=0.0
    )


def test_spheres_packable_exact_mathematical_boundary_five_spheres():
    critical_ratio = np.sqrt(2) - 1
    assert spheres_packable(
        radius_outer=1.0, radius_inner=critical_ratio, num_inner=5, safety_margin=0.0
    )


def test_spheres_packable_exact_mathematical_boundary_six_spheres():
    critical_ratio = np.sqrt(2) - 1
    assert spheres_packable(
        radius_outer=1.0, radius_inner=critical_ratio, num_inner=6, safety_margin=0.0
    )


def test_spheres_packable_numerical_precision_boundary():
    assert not spheres_packable(
        radius_outer=1.0, radius_inner=0.500000001, num_inner=2, safety_margin=0.0
    )


def test_spheres_packable_machine_epsilon_boundary():
    assert not spheres_packable(
        radius_outer=1.0,
        radius_inner=1.0 + np.finfo(float).eps,
        num_inner=1,
        safety_margin=0.0,
    )


def test_spheres_packable_large_negative_safety_margin():
    assert spheres_packable(
        radius_outer=1.0, radius_inner=2.0, num_inner=1, safety_margin=-0.5
    )


def test_spheres_packable_float_num_inner_treated_as_integer():
    # Testing function behavior with float input (runtime coercion)
    assert not spheres_packable(
        radius_outer=1.0, radius_inner=0.1, num_inner=2.5  # type: ignore[arg-type]
    )


def test_spheres_packable_exact_zero_safety_margin_edge():
    assert spheres_packable(
        radius_outer=1.0, radius_inner=1.0, num_inner=1, safety_margin=0.0
    )


def test_spheres_packable_tiny_over_boundary_two_spheres():
    assert not spheres_packable(
        radius_outer=1.0, radius_inner=0.5000001, num_inner=2, safety_margin=0.0
    )


# Fibonacci sphere generation tests


def test_generate_fibonacci_points_on_sphere_with_default_count():
    points = generate_fibonacci_points_on_sphere(num_points=100)

    assert points.shape == (100, 3)
    norms = np.linalg.norm(points, axis=1)
    assert np.allclose(norms, 1.0, rtol=1e-10)


def test_generate_fibonacci_points_on_sphere_with_custom_count():
    points = generate_fibonacci_points_on_sphere(num_points=50)

    assert points.shape == (50, 3)
    norms = np.linalg.norm(points, axis=1)
    assert np.allclose(norms, 1.0, rtol=1e-10)


def test_generate_fibonacci_points_on_sphere_fails_with_single_point():
    with pytest.raises(ZeroDivisionError):
        generate_fibonacci_points_on_sphere(num_points=1)


def test_generate_fibonacci_points_on_sphere_have_uniform_distribution():
    points = generate_fibonacci_points_on_sphere(num_points=1000)

    x_positive = np.sum(points[:, 0] > 0)
    x_negative = np.sum(points[:, 0] < 0)
    y_positive = np.sum(points[:, 1] > 0)
    y_negative = np.sum(points[:, 1] < 0)
    z_positive = np.sum(points[:, 2] > 0)
    z_negative = np.sum(points[:, 2] < 0)

    assert abs(int(x_positive) - int(x_negative)) < 200
    assert abs(int(y_positive) - int(y_negative)) < 200
    assert abs(int(z_positive) - int(z_negative)) < 200


def test_generate_fibonacci_points_on_sphere_with_zero_points():
    points = generate_fibonacci_points_on_sphere(num_points=0)
    assert points.shape == (0, 3)


def test_generate_fibonacci_points_on_sphere_with_two_points():
    points = generate_fibonacci_points_on_sphere(num_points=2)
    assert points.shape == (2, 3)

    # Verify exact coordinates for 2 points case
    expected_points = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    assert np.allclose(points, expected_points, atol=1e-10)


def test_generate_fibonacci_points_on_sphere_with_default_parameter():
    points = generate_fibonacci_points_on_sphere()
    assert points.shape == (10000, 3)
    norms = np.linalg.norm(points, axis=1)
    assert np.allclose(norms, 1.0, rtol=1e-10)
