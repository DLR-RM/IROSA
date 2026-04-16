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

"""Trajectory correction for robotic planning with obstacle avoidance.

This module provides functionality to correct robotic trajectories to maintain
safe distances from obstacles using signed distance fields (SDFs) and iterative
gradient-based correction.
"""

from __future__ import annotations

import numpy as np


class TrajectoryCorrector:
    """Trajectory corrector for robotic planning with obstacle avoidance.

    Uses analytic signed distance fields for spheres and oriented bounding boxes,
    combined with iterative gradient-based correction to push trajectory points
    away from obstacles while preserving timestamps and orientations.
    """

    @staticmethod
    def sphere_sdf(point: np.ndarray, center: np.ndarray, radius: float) -> float:
        """Calculate signed distance from point to sphere surface.

        :param point: 3D point coordinates [x, y, z]
        :param center: 3D sphere center coordinates [x, y, z]
        :param radius: Sphere radius in meters
        :return: Signed distance (positive outside, negative inside sphere)
        """
        return float(np.linalg.norm(point - center) - radius)

    @staticmethod
    def obb_sdf(point: np.ndarray, center: np.ndarray, half_extents: np.ndarray, rotation: np.ndarray) -> float:
        """Calculate signed distance from point to oriented bounding box surface.

        :param point: 3D point coordinates [x, y, z]
        :param center: 3D box center coordinates [x, y, z]
        :param half_extents: Half-extents of box in each dimension [hx, hy, hz]
        :param rotation: 3x3 rotation matrix for box orientation
        :return: Signed distance (positive outside, negative inside box)
        """
        local_point = rotation.T @ (point - center)
        d = np.abs(local_point) - half_extents
        outside_distance = np.linalg.norm(np.maximum(d, 0.0))
        inside_distance = np.max(d)

        if np.all(d <= 0):
            return float(inside_distance)
        else:
            return float(outside_distance)

    @staticmethod
    def soft_min(distances: np.ndarray, k: float = 10.0) -> float:
        """Compute smooth soft minimum of distances.

        :param distances: Array of distance values to compute soft minimum of
        :param k: Smoothness parameter (higher = sharper minimum, default=10.0)
        :return: Soft minimum value (smooth approximation of minimum distance)
        """
        if len(distances) == 0:
            return float("inf")

        max_dist = np.max(distances)
        shifted_distances = distances - max_dist
        exp_sum = np.sum(np.exp(-k * shifted_distances))
        return float(max_dist - np.log(exp_sum) / k)

    @staticmethod
    def sphere_sdf_normal(point: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Calculate analytical normal (gradient direction) for sphere SDF.

        :param point: 3D point coordinates [x, y, z]
        :param center: 3D sphere center coordinates [x, y, z]
        :return: Normalized 3D normal vector pointing away from sphere center
        """
        direction = point - center
        norm = np.linalg.norm(direction)

        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0])

        return direction / norm

    @staticmethod
    def ellipsoid_sdf(point: np.ndarray, center: np.ndarray, half_extents: np.ndarray, rotation: np.ndarray) -> float:
        """Calculate signed distance from point to ellipsoid surface.

        :param point: 3D point coordinates [x, y, z]
        :param center: 3D ellipsoid center coordinates [x, y, z]
        :param half_extents: Half-extents (semi-axes) of ellipsoid [a, b, c]
        :param rotation: 3x3 rotation matrix for ellipsoid orientation
        :return: Signed distance (positive outside, negative inside ellipsoid)
        """
        local_point = rotation.T @ (point - center)
        scaled_point = local_point / half_extents
        norm_scaled = np.linalg.norm(scaled_point)
        return float(norm_scaled - 1.0)

    @staticmethod
    def ellipsoid_sdf_normal(
        point: np.ndarray, center: np.ndarray, half_extents: np.ndarray, rotation: np.ndarray
    ) -> np.ndarray:
        """Calculate analytical normal (gradient direction) for ellipsoid SDF.

        :param point: 3D point coordinates [x, y, z]
        :param center: 3D ellipsoid center coordinates [x, y, z]
        :param half_extents: Half-extents (semi-axes) of ellipsoid [a, b, c]
        :param rotation: 3x3 rotation matrix for ellipsoid orientation
        :return: Normalized 3D normal vector pointing away from ellipsoid surface
        """
        local_point = rotation.T @ (point - center)

        a2 = half_extents[0] ** 2
        b2 = half_extents[1] ** 2
        c2 = half_extents[2] ** 2

        n_local = np.array([local_point[0] / a2, local_point[1] / b2, local_point[2] / c2])

        n_local_norm = np.linalg.norm(n_local)
        if n_local_norm < 1e-10:
            return np.array([1.0, 0.0, 0.0])

        n_local = n_local / n_local_norm
        n_world = rotation @ n_local

        n_world_norm = np.linalg.norm(n_world)
        if n_world_norm < 1e-10:
            return np.array([1.0, 0.0, 0.0])

        return n_world / n_world_norm

    @staticmethod
    def scene_sdf_gradient_analytical(point: np.ndarray, obstacles: list[dict]) -> np.ndarray:
        """Calculate gradient of scene SDF using analytical normals.

        :param point: 3D point coordinates [x, y, z] to calculate gradient at
        :param obstacles: List of obstacle dictionaries (sphere/obb format)
        :return: 3D gradient vector pointing away from obstacles
        """
        if len(obstacles) == 0:
            return np.zeros(3)

        distances = []
        normals = []

        for obstacle in obstacles:
            if obstacle["type"] == "sphere":
                dist = TrajectoryCorrector.sphere_sdf(point, obstacle["center"], obstacle["radius"])
                normal = TrajectoryCorrector.sphere_sdf_normal(point, obstacle["center"])
            elif obstacle["type"] == "ellipsoid":
                dist = TrajectoryCorrector.ellipsoid_sdf(
                    point, obstacle["center"], obstacle["half_extents"], obstacle["rotation"]
                )
                normal = TrajectoryCorrector.ellipsoid_sdf_normal(
                    point, obstacle["center"], obstacle["half_extents"], obstacle["rotation"]
                )
            elif obstacle["type"] == "obb":
                dist = TrajectoryCorrector.obb_sdf(point, obstacle["center"], obstacle["half_extents"], obstacle["rotation"])
                # Intentionally use ellipsoid normal for OBB: provides smoother gradients
                # for iterative correction while OBB SDF gives accurate distance.
                normal = TrajectoryCorrector.ellipsoid_sdf_normal(
                    point, obstacle["center"], obstacle["half_extents"], obstacle["rotation"]
                )
            else:
                raise ValueError(f"Unknown obstacle type: {obstacle['type']}")

            distances.append(dist)
            normals.append(normal)

        if len(distances) == 0:
            return np.zeros(3)

        distances_arr = np.array(distances)
        max_dist = np.max(distances_arr)
        shifted_distances = distances_arr - max_dist

        k = 10.0
        weights = np.exp(-k * shifted_distances)
        weights /= np.sum(weights)

        gradient = np.zeros(3)
        for i, normal in enumerate(normals):
            gradient += weights[i] * normal

        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > 1e-10:
            gradient /= gradient_norm

        return gradient

    @staticmethod
    def scene_sdf(point: np.ndarray, obstacles: list[dict]) -> float:
        """Calculate scene SDF as soft minimum of all obstacle SDFs.

        :param point: 3D point coordinates [x, y, z]
        :param obstacles: List of obstacle dictionaries
        :return: Scene SDF value
        """
        distances = []

        for obstacle in obstacles:
            if obstacle["type"] == "sphere":
                dist = TrajectoryCorrector.sphere_sdf(point, obstacle["center"], obstacle["radius"])
            elif obstacle["type"] == "ellipsoid":
                dist = TrajectoryCorrector.ellipsoid_sdf(
                    point, obstacle["center"], obstacle["half_extents"], obstacle["rotation"]
                )
            elif obstacle["type"] == "obb":
                dist = TrajectoryCorrector.obb_sdf(point, obstacle["center"], obstacle["half_extents"], obstacle["rotation"])
            else:
                continue

            distances.append(dist)

        return TrajectoryCorrector.soft_min(np.array(distances))

    @staticmethod
    def _expand_mask_with_boundary_points(mask: np.ndarray, boundary_points: int = 3) -> np.ndarray:
        """Expand boolean mask by including boundary points around masked regions."""
        expanded_mask = mask.copy()
        indices = np.where(mask)[0]

        if len(indices) == 0:
            return expanded_mask

        regions = []
        region_start = indices[0]
        region_end = indices[0]

        for idx in indices[1:]:
            if idx == region_end + 1:
                region_end = idx
            else:
                regions.append((region_start, region_end))
                region_start = idx
                region_end = idx

        regions.append((region_start, region_end))

        for region_start, region_end in regions:
            expand_left_start = max(0, region_start - boundary_points)
            expand_right_end = min(len(mask) - 1, region_end + boundary_points)
            expanded_mask[expand_left_start : expand_right_end + 1] = True

        return expanded_mask

    @staticmethod
    def _redistribute_points_equally_along_line(positions: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Redistribute points equally along the trajectory line."""
        if len(positions) < 2:
            return positions.copy()

        cumulative_lengths = [0.0]
        for i in range(len(positions) - 1):
            segment_length = float(np.linalg.norm(positions[i + 1] - positions[i]))
            cumulative_lengths.append(cumulative_lengths[-1] + segment_length)

        total_length = cumulative_lengths[-1]

        if total_length < 1e-10:
            return positions.copy()

        redistributed = np.zeros_like(positions)
        redistributed[0] = positions[0].copy()
        redistributed[-1] = positions[-1].copy()

        for i in range(1, len(positions) - 1):
            target_arc_length = (i / (len(positions) - 1)) * total_length

            segment_idx = 0
            for idx in range(len(cumulative_lengths) - 1):
                if cumulative_lengths[idx + 1] >= target_arc_length:
                    segment_idx = idx
                    break

            segment_start_length = cumulative_lengths[segment_idx]
            segment_end_length = cumulative_lengths[segment_idx + 1]
            segment_length = float(segment_end_length - segment_start_length)

            if segment_length < 1e-10:
                target_pos = positions[segment_idx + 1].copy()
            else:
                t = (target_arc_length - segment_start_length) / segment_length
                t = np.clip(t, 0.0, 1.0)
                target_pos = (1.0 - t) * positions[segment_idx] + t * positions[segment_idx + 1]

            if mask is not None and not mask[i]:
                redistributed[i] = positions[i].copy()
            else:
                redistributed[i] = target_pos

        return redistributed

    @staticmethod
    def correct_trajectory_for_obstacles(
        trajectory: np.ndarray,
        obstacles: list[dict],
        safety_distance: float = 0.0,
        max_iterations: int = 100,
        step_size: float = 0.01,
        window_size: int = 5,
    ) -> np.ndarray:
        """Correct trajectory to maintain safety distance from obstacles.

        Uses iterative gradient-based correction with SDF. For OBB obstacles, uses
        accurate OBB distance combined with smooth ellipsoid gradients.

        :param trajectory: Nx9 array [time, x, y, z, qw, qx, qy, qz, additional]
        :param obstacles: List of obstacle dictionaries:
            - Sphere: {'type': 'sphere', 'center': np.array([x,y,z]), 'radius': float}
            - OBB: {'type': 'obb', 'center': np.array([x,y,z]),
                   'half_extents': np.array([hx,hy,hz]), 'rotation': 3x3 rotation matrix}
        :param safety_distance: Target distance from obstacle surface (meters)
        :param max_iterations: Maximum correction iterations
        :param step_size: Step size for gradient-based correction (meters)
        :param window_size: Moving average window size for smoothing
        :return: Corrected trajectory with same format as input
        """
        if not isinstance(trajectory, np.ndarray) or trajectory.ndim != 2:
            raise ValueError("trajectory must be a 2D numpy array")
        if trajectory.shape[1] < 8:
            raise ValueError("trajectory must have at least 8 columns [time, x, y, z, qw, qx, qy, qz]")
        if safety_distance < 0:
            raise ValueError("safety_distance must be non-negative")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")

        if len(obstacles) == 0:
            return trajectory.copy()

        corrected_trajectory = trajectory.copy()
        needs_redistribution = np.zeros(len(trajectory), dtype=bool)

        for _iteration in range(max_iterations):
            for i in range(len(trajectory)):
                scene_distance = TrajectoryCorrector.scene_sdf(corrected_trajectory[i, 1:4], obstacles)
                needs_redistribution[i] = needs_redistribution[i] or scene_distance < safety_distance

            needs_redistribution_extended = TrajectoryCorrector._expand_mask_with_boundary_points(
                needs_redistribution, boundary_points=3
            )

            for i in range(len(trajectory)):
                current_distance = TrajectoryCorrector.scene_sdf(corrected_trajectory[i, 1:4], obstacles)

                if current_distance < safety_distance:
                    corrected_pos = corrected_trajectory[i, 1:4].copy()
                    gradient = TrajectoryCorrector.scene_sdf_gradient_analytical(corrected_pos, obstacles)
                    gradient_norm = np.linalg.norm(gradient)

                    if gradient_norm > 1e-10:
                        corrected_pos += step_size * gradient

                    corrected_trajectory[i, 1:4] = corrected_pos

            positions_only = corrected_trajectory[:, 1:4]
            redistributed_positions = TrajectoryCorrector._redistribute_points_equally_along_line(
                positions_only, needs_redistribution_extended
            )
            corrected_trajectory[:, 1:4] = redistributed_positions

            all_points_converged = all(
                TrajectoryCorrector.scene_sdf(corrected_trajectory[i, 1:4], obstacles) >= safety_distance
                for i in range(len(trajectory))
            )
            if all_points_converged:
                break

        return corrected_trajectory
