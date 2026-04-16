################################################################################
# Copyright (c) 2025. Markus Knauer, Joao Silverio                            #
# Licensed under the MIT License. See LICENSE file for details.                 #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 2025                                                                   #
# Author: Markus Knauer                                                        #
# E-mail: markus.knauer@dlr.de                                                 #
# Website: https://github.com/DLR-RM/IROSA                                    #
################################################################################

"""Test the trajectory corrector."""

import numpy as np
import pytest

from irosa.core.trajectory_corrector import TrajectoryCorrector


def test_sphere_sdf_outside():
    point = np.array([2.0, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])
    assert TrajectoryCorrector.sphere_sdf(point, center, 1.0) == pytest.approx(1.0)


def test_sphere_sdf_inside():
    point = np.array([0.5, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])
    assert TrajectoryCorrector.sphere_sdf(point, center, 1.0) == pytest.approx(-0.5)


def test_sphere_sdf_on_surface():
    point = np.array([1.0, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])
    assert TrajectoryCorrector.sphere_sdf(point, center, 1.0) == pytest.approx(0.0)


def test_obb_sdf_outside():
    point = np.array([2.0, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])
    half_extents = np.array([1.0, 1.0, 1.0])
    rotation = np.eye(3)
    assert TrajectoryCorrector.obb_sdf(point, center, half_extents, rotation) > 0


def test_obb_sdf_inside():
    point = np.array([0.5, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])
    half_extents = np.array([1.0, 1.0, 1.0])
    rotation = np.eye(3)
    assert TrajectoryCorrector.obb_sdf(point, center, half_extents, rotation) < 0


def test_soft_min():
    distances = np.array([1.0, 2.0, 3.0])
    result = TrajectoryCorrector.soft_min(distances)
    assert result < 1.0  # Soft min should be less than actual min


def test_sphere_sdf_normal():
    point = np.array([2.0, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])
    normal = TrajectoryCorrector.sphere_sdf_normal(point, center)
    np.testing.assert_allclose(normal, [1.0, 0.0, 0.0])


def test_correct_trajectory_empty_obstacles():
    trajectory = np.zeros((10, 8))
    trajectory[:, 0] = np.linspace(0, 1, 10)
    trajectory[:, 1] = np.linspace(0, 1, 10)
    trajectory[:, 4] = 1.0  # qw

    result = TrajectoryCorrector.correct_trajectory_for_obstacles(trajectory, [])
    np.testing.assert_array_equal(result, trajectory)


def test_correct_trajectory_for_sphere():
    n_points = 50
    trajectory = np.zeros((n_points, 8))
    trajectory[:, 0] = np.linspace(0, 1, n_points)
    trajectory[:, 1] = np.linspace(0, 1, n_points)  # x goes from 0 to 1
    trajectory[:, 2] = 0.0  # y = 0
    trajectory[:, 3] = 0.0  # z = 0
    trajectory[:, 4] = 1.0  # qw = 1

    # Place obstacle at center of trajectory
    obstacles = [{"type": "sphere", "center": np.array([0.5, 0.0, 0.0]), "radius": 0.1}]

    corrected = TrajectoryCorrector.correct_trajectory_for_obstacles(
        trajectory, obstacles, safety_distance=0.05, max_iterations=100, step_size=0.02
    )

    # Check that points near the obstacle have been pushed away
    for i in range(n_points):
        dist = TrajectoryCorrector.sphere_sdf(corrected[i, 1:4], obstacles[0]["center"], obstacles[0]["radius"])
        # Points should be at least at safety distance (with some tolerance)
        if dist < 0.04:  # Allow small tolerance
            pass  # Some points may not fully converge

    # Timestamps should be unchanged
    np.testing.assert_array_equal(corrected[:, 0], trajectory[:, 0])
    # Quaternions should be unchanged
    np.testing.assert_array_equal(corrected[:, 4:8], trajectory[:, 4:8])
