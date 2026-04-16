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

"""KMP wrapper for LLM-based robot skill adaptation."""

from __future__ import annotations

import logging
import pathlib
import pickle as pkl
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import numpy as np

from irosa.core.trajectory_corrector import TrajectoryCorrector
from irosa.exceptions import DataValidationError, ToolArgsValidationError
from irosa.models.kmp_core import Kmp
from irosa.robots.robot import Robot

logger = logging.getLogger(__name__)


class KMPWrapper:
    """Wrapper for Kernelized Movement Primitives enabling LLM-based skill adaptation.

    Manages demonstration data loading, KMP model training, trajectory generation,
    via-point management, obstacle avoidance, and robot interaction.

    Demonstration Data Format:
        3D numpy array with shape (Nr_Demos, Nr_Points, Nr_Variables), where:
        - Nr_Variables: [time, x, y, z, qw, qx, qy, qz] (8D) or [time, x, y, z] (4D)
        - Time dimension is normalized to [0, 1]
    """

    kmp_saves_path: pathlib.Path = pathlib.Path(__file__).parent.parent.parent / "kmp_saves"

    def __init__(
        self,
        demonstration_path: pathlib.Path | None = None,
        robot: Robot | None = None,
        t_lim: float = 1.0,
        test_mode: bool = False,
        force_retrain: bool = False,
        base_kwargs: dict[str, Any] | None = None,
        demonstrations: np.ndarray | None = None,
    ):
        """Initialize KMP wrapper.

        :param demonstration_path: Path to demonstration data file (.npz or .pickle)
        :param robot: Robot instance to control
        :param t_lim: Time limit for trajectory execution
        :param test_mode: Enable test mode with reduced parameters
        :param force_retrain: If True, train immediately
        :param base_kwargs: Optional KMP configuration overrides
        :param demonstrations: Optional pre-loaded demonstrations array
        """
        self.demonstration_path = demonstration_path
        self.robot = robot
        self.test_mode = test_mode
        self.t_lim = t_lim
        self.robot_stopped = False
        self._stop_lock = threading.Lock()

        self.kmp: Kmp | None = None
        self.demonstrations: np.ndarray | None = demonstrations
        self.kmp_kwargs: dict[str, Any] | None = None
        self.via_points: dict | None = None
        self.base_kwargs = base_kwargs
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.x_in: np.ndarray | None = None
        self.trajectory: np.ndarray | None = None

        if robot is not None:
            robot.on_setup(test_mode=test_mode)

        self._init_dummy_trajectory()

        if force_retrain and (demonstration_path is not None or demonstrations is not None):
            self.train_skill(force_retrain=True)

        if self.robot is not None and self.trajectory is not None:
            self.robot.init_predicting_frequency(trajectory=self.trajectory)

    @contextmanager
    def safe_robot_lock(self):
        """Context manager for safe robot locking."""
        if self.robot is not None:
            with self.robot.robot_lock:
                yield
        else:
            yield

    def _init_dummy_trajectory(self) -> None:
        """Initialize a dummy trajectory."""
        n_points = 10
        pose = np.array([[0.6, 0.0, 0.4, 1, 0, 0, 0], [0.6, 0.0, 0.4, 1, 0, 0, 0]])
        self.mean = np.repeat(pose, [1, n_points - 1], axis=0)
        self.cov = np.eye(7, dtype=np.float64)
        self.x_in = np.linspace(0, self.t_lim, n_points)
        self.update_trajectory()

    def update_trajectory(self, delta_t: float = 0.05) -> None:
        """Update the trajectory from current predictions."""
        if self.x_in is not None and self.mean is not None:
            with self.safe_robot_lock():
                self.trajectory = np.column_stack((self.x_in, self.mean))
                if self.robot is not None:
                    self.robot.on_trajectory_updated(self.trajectory, self.mean, delta_t=delta_t)
            logger.info("Trajectory updated")

    def get_data(self, demo_filename: str = "") -> np.ndarray:
        """Load demonstration data from file (.npz or .pickle).

        :return: 3D demonstrations array (Nr_Demos, Nr_Points, Nr_Variables)
        """
        if demo_filename:
            path = pathlib.Path(demo_filename)
        elif self.demonstration_path is not None:
            path = self.demonstration_path
        else:
            raise ValueError("No demonstration path provided.")

        demonstrations = self._load_demonstrations_from_file(path)

        processed = []
        for demo in demonstrations:
            if demo.shape[1] not in [4, 8]:
                raise DataValidationError(f"Only 4 or 8 values are allowed, not {demo.shape[1]}")
            if demo.shape[1] == 4:
                quaternions = np.tile([1, 0, 0, 0], (demo.shape[0], 1))
                demo = np.concatenate([demo, quaternions], axis=1)
            processed.append(demo)

        return np.array(processed)

    @staticmethod
    def _load_demonstrations_from_file(datadir: pathlib.Path) -> np.ndarray:
        """Load and normalize demonstration data from .npz or .pickle file."""
        if datadir.suffix == ".npz":
            data = np.load(datadir)
            demonstrations_list = data["demonstrations"]
        else:
            with open(datadir, "rb") as f:
                demonstrations_list = pkl.load(f)

        processed = []
        for demo in demonstrations_list:
            if isinstance(demo, list):
                demo = np.array(demo)
            demo = demo.copy()
            T_max = np.max(demo[:, 0])
            demo[:, 0] /= T_max
            processed.append(demo)

        return np.array(processed)

    def _get_kmp_kwargs(self) -> dict[str, Any]:
        """Get KMP configuration parameters."""
        if self.base_kwargs is not None:
            base_kwargs = self.base_kwargs.copy()
        else:
            base_kwargs = {
                "gmm_n_components": 12,
                "N": 500,
                "l": 0.1,
                "h": 1.0,
                "lambda1": 0.1,
                "lambda2": 1,
                "alpha": 1,
                "kernel_function": "matern2",
            }

        if self.test_mode:
            base_kwargs.update({"gmm_n_components": 6, "N": 20})

        return base_kwargs

    def train_skill(self, demo_filename: str = "", save_name: str = "", force_retrain: bool = False) -> None:
        """Train KMP skill from demonstration data.

        :param demo_filename: Optional demonstration filename
        :param save_name: Optional name for saved skill file
        :param force_retrain: Always retrain even if cache exists
        """
        if not force_retrain and self._check_cache_exists(save_name):
            logger.info("Cached KMP found. Loading from cache.")
            if self._load_from_cache(save_name):
                self.update_trajectory()
                return

        if self.demonstrations is None:
            self.demonstrations = self.get_data(demo_filename)

        self._create_new_kmp_model()
        self._save_to_cache(save_name)
        self.update_trajectory()
        logger.info("Successfully trained new skill!")

    def _create_new_kmp_model(self) -> None:
        """Create and train a new KMP model."""
        self.kmp_kwargs = self._get_kmp_kwargs()
        self.kmp = Kmp(**self.kmp_kwargs)
        self.x_in = np.linspace(0, self.t_lim, self.kmp.N)
        self.fit_kmp()

    def fit_kmp(self) -> None:
        """Fit KMP to loaded demonstrations."""
        if self.demonstrations is None or len(self.demonstrations) == 0:
            raise RuntimeError("Demonstrations cannot be None or empty!")
        assert self.kmp is not None
        assert self.x_in is not None

        dataset = np.vstack(list(self.demonstrations))
        X = dataset[:, :1]
        Y = dataset[:, 1:]
        self.kmp.fit(X, Y, x_in=self.x_in)
        self.mean, self.cov = self.kmp.predict_with_uncertainty(self.x_in)

    def is_trained(self) -> bool:
        """Check if the KMP model has been trained."""
        return self.kmp is not None and self.mean is not None and self.demonstrations is not None

    def _ensure_trained_model(self, method_name: str) -> None:
        """Ensure model is trained, auto-training if necessary."""
        if not self.is_trained():
            logger.info("Model not trained for %s(). Auto-training...", method_name)
            self.train_skill()

    def get_parameters(self) -> dict:
        """Get current KMP parameters."""
        if self.kmp_kwargs is None:
            return self._get_kmp_kwargs()
        return self.kmp_kwargs

    def update_kernel_length(self, kernel_length: float) -> None:
        """Update kernel length scale and refit."""
        self._ensure_trained_model("update_kernel_length")
        assert self.kmp is not None
        assert self.kmp_kwargs is not None
        with self.safe_robot_lock():
            self.kmp.l = kernel_length
            self.kmp_kwargs["l"] = kernel_length
            self.fit_kmp()

    def add_viapoints(self, input_via, output_via, gamma: float = 1e-8) -> None:
        """Add via-points to the KMP trajectory.

        :param input_via: Time positions where via-points are added
        :param output_via: Output values (e.g. x,y,z)
        :param gamma: Covariance parameter for constraints
        """
        self._ensure_trained_model("add_viapoints")
        assert self.kmp is not None
        assert self.x_in is not None

        try:
            input_via = self.kmp.check_via_point_structure(input_via)
            output_via = self.kmp.check_via_point_structure(output_via)

            if len(input_via) != len(output_via):
                raise ValueError(f"len of input {len(input_via)} must match output {len(output_via)}")

            # Adjust dimensions if needed
            if self.demonstrations is not None and len(self.demonstrations) > 0:
                expected_dims = self.demonstrations[0].shape[1] - 1
                output_via = self._adjust_viapoint_dimensions(output_via, expected_dims)

            with self.safe_robot_lock():
                self.kmp.add_viapoints(input_via=input_via, output_via=output_via, gamma=gamma)
                self.mean, self.cov = self.kmp.predict_with_uncertainty(self.x_in)
                self.via_points = self.kmp.via_points

            self.update_trajectory()
            if self.robot is not None:
                self.robot.on_viapoints_added([list(v[0:3]) for v in output_via])
        except Exception as e:
            raise ToolArgsValidationError(f"Error adding via points: {e}") from e

    def _adjust_viapoint_dimensions(self, output_via, expected_output_dims: int):
        """Adjust via-point dimensions to match KMP output."""
        provided_dims = len(output_via[0]) if output_via and len(output_via) > 0 else 0
        if provided_dims == expected_output_dims:
            return output_via
        if provided_dims == 7 and expected_output_dims == 8:
            return [[*list(point), 0.0] for point in output_via]
        if provided_dims == 8 and expected_output_dims == 7:
            return [point[:7] for point in output_via]
        return output_via

    def remove_viapoints(self, input_via) -> None:
        """Remove via-points from the KMP trajectory."""
        self._ensure_trained_model("remove_viapoints")
        assert self.kmp is not None
        assert self.x_in is not None
        try:
            input_via = self.kmp.check_via_point_structure(input_via)

            removed_positions: list[list] = []
            if self.robot is not None:
                for entry in input_via:
                    idx = self.kmp.via_points["input"].index(entry)
                    removed_positions.append(list(self.kmp.via_points["output"][idx][0:3]))

            with self.safe_robot_lock():
                self.kmp.update_K()
                self.kmp.remove_viapoints(input_via=input_via)
                self.mean, self.cov = self.kmp.predict_with_uncertainty(self.x_in)
                self.via_points = self.kmp.via_points
                self.update_trajectory()

            if self.robot is not None and removed_positions:
                self.robot.on_viapoints_removed(removed_positions)
        except Exception as e:
            raise ToolArgsValidationError(f"Error removing via points: {e}") from e

    def edit_viapoints(self, input_via_old, input_via_new, output_via_new, gamma: float = 1e-8) -> None:
        """Edit existing via-points."""
        self._ensure_trained_model("edit_viapoints")
        assert self.kmp is not None
        assert self.x_in is not None
        try:
            input_via_old = self.kmp.check_via_point_structure(input_via_old)
            input_via_new = self.kmp.check_via_point_structure(input_via_new)
            output_via_new = self.kmp.check_via_point_structure(output_via_new)

            removed_positions: list[list] = []
            if self.robot is not None:
                for entry in input_via_old:
                    idx = self.kmp.via_points["input"].index([entry])
                    removed_positions.append(list(self.kmp.via_points["output"][idx][0:3]))

            with self.safe_robot_lock():
                self.kmp.update_K()
                self.kmp.edit_viapoints(
                    input_via_old=input_via_old,
                    input_via_new=input_via_new,
                    output_via_new=output_via_new,
                    gamma=gamma,
                    replace=True,
                )
                self.mean, self.cov = self.kmp.predict_with_uncertainty(self.x_in)
                self.via_points = self.kmp.via_points
                self.update_trajectory()

            if self.robot is not None:
                if removed_positions:
                    self.robot.on_viapoints_removed(removed_positions)
                self.robot.on_viapoints_added([list(v[0:3]) for v in output_via_new])
        except Exception as e:
            raise ToolArgsValidationError(f"Error editing via points: {e}") from e

    def get_viapoints(self) -> tuple:
        """Get all existing via-points."""
        self._ensure_trained_model("get_viapoints")
        assert self.kmp is not None
        return self.kmp.via_points["input"], self.kmp.via_points["output"], self.kmp.via_points["gamma"]

    def add_repulsion_point(
        self,
        position: np.ndarray | list,
        radius: float | None = None,
        safety_margin: float = 0.0,
        dimensions: list | None = None,
    ) -> None:
        """Add a repulsion point for obstacle avoidance.

        Uses TrajectoryCorrector to correct trajectory, then adds via-points
        at corrected positions.

        :param position: Obstacle center [x, y, z]
        :param radius: Obstacle radius (for spheres)
        :param safety_margin: Additional safety margin beyond surface
        :param dimensions: [width, length, height] for box-shaped obstacles
        """
        self._ensure_trained_model("add_repulsion_point")
        assert self.trajectory is not None
        try:
            if isinstance(position, list):
                position = np.array(position)
            if len(position) != 3:
                position = position[:3]

            center = position.copy()
            self.update_trajectory()

            obstacles: list[dict[str, Any]]
            if dimensions is not None:
                half_extents = np.array([dimensions[1], dimensions[0], dimensions[2]]) / 2.0
                rotation = np.eye(3)
                obstacles = [{"type": "obb", "center": center, "half_extents": half_extents, "rotation": rotation}]
            else:
                if radius is None:
                    raise ValueError("Either radius or dimensions must be provided")
                obstacles = [{"type": "sphere", "center": center, "radius": radius}]

            corrected = TrajectoryCorrector.correct_trajectory_for_obstacles(
                self.trajectory, obstacles, safety_distance=safety_margin, max_iterations=5000, step_size=0.005
            )

            update_input, update_output = [], []
            position_changes = np.linalg.norm(self.trajectory[:, 1:4] - corrected[:, 1:4], axis=1)

            for i, change in enumerate(position_changes):
                if change > 0.005:
                    time_point = corrected[i, 0]
                    corrected_position = corrected[i, 1:4]
                    corrected_orientation = corrected[i, 4:8]
                    update_input.append([time_point])
                    via_point = np.concatenate([corrected_position, corrected_orientation])
                    update_output.append(via_point)

            if update_input:
                logger.info("Adding %d via-points to avoid obstacle at %s", len(update_input), center)
                self.add_viapoints(input_via=np.array(update_input), output_via=np.array(update_output))
            else:
                logger.info("Trajectory already avoids obstacle")

        except Exception as e:
            raise ToolArgsValidationError(f"Error adding repulsion point: {e}") from e

    def run(self) -> None:
        """Execute the current trajectory on the robot."""
        if self.robot is not None and self.trajectory is not None:
            self.robot.run(trajectory=self.trajectory)
        else:
            logger.info("No robot configured - trajectory not executed")

    def stop_robot(self) -> None:
        """Safely stop the robot."""
        if self.robot is not None:
            with self._stop_lock:
                if not self.robot_stopped:
                    self.robot_stopped = True
                    self.robot.robot_lock.acquire()

    def restart_robot(self) -> None:
        """Restart the robot."""
        if self.robot is not None:
            with self._stop_lock:
                if self.robot_stopped:
                    self.robot.robot_lock.release()
                    self.robot_stopped = False

    # -- Cache system --

    def _save_to_cache(self, filename: str | None = None) -> None:
        """Save KMP to cache."""
        cache_dir = self.kmp_saves_path
        cache_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"kmp_save_{timestamp}.kmp"
        if not filename.endswith(".kmp"):
            filename += ".kmp"

        cache_data: dict[str, Any] = {
            "kmp": self.kmp,
            "kmp_kwargs": self.kmp_kwargs,
            "demonstrations": self.demonstrations,
            "t_lim": self.t_lim,
            "x_in": self.x_in,
            "mean": self.mean,
            "cov": self.cov,
            "via_points": self.via_points,
        }
        if self.demonstration_path is not None:
            cache_data["demonstration_path"] = str(self.demonstration_path)

        with (cache_dir / filename).open("wb") as f:
            pkl.dump(cache_data, f)
        logger.info("KMP saved to %s", cache_dir / filename)

    def _load_from_cache(self, filename: str | None = None) -> bool:
        """Load KMP from cache. Returns True if successful."""
        try:
            cache_dir = self.kmp_saves_path
            if filename:
                file_path = cache_dir / filename
                if not file_path.suffix:
                    file_path = file_path.with_suffix(".kmp")
            else:
                cache_files = list(cache_dir.glob("*.kmp"))
                if not cache_files:
                    return False
                import os

                file_path = max(cache_files, key=os.path.getmtime)

            if not file_path.exists():
                return False

            with file_path.open("rb") as f:
                data = pkl.load(f)

            for attr in ["kmp", "kmp_kwargs", "demonstrations", "t_lim", "x_in", "mean", "cov", "via_points"]:
                if attr in data:
                    setattr(self, attr, data[attr])

            if self.x_in is not None and self.mean is not None:
                self.trajectory = np.column_stack((self.x_in, self.mean))

            logger.info("KMP loaded from %s", file_path)
            return True
        except Exception as e:
            logger.error("Error loading KMP from cache: %s", e)
            return False

    def _check_cache_exists(self, filename: str | None = None) -> bool:
        """Check if cache file exists."""
        cache_dir = self.kmp_saves_path
        if not cache_dir.exists():
            return False
        if filename:
            path = cache_dir / filename
            if not path.suffix:
                path = path.with_suffix(".kmp")
            return path.exists()
        return len(list(cache_dir.glob("*.kmp"))) > 0
